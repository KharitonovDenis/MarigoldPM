# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------

import logging
import numpy as np
import torch
from PIL import Image
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import BaseOutput
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Dict, List, Optional, Sequence, Union

from .util.batchsize import find_batch_size
from .util.ensemble import ensemble_depth
from .util.image_util import (
    chw2hwc,
    colorize_depth_maps,
    get_tv_resample_method,
    resize_max_res,
)


class MarigoldPMVideoOutput(BaseOutput):
    """
    Output class for MarigoldPM Video Monocular Depth Estimation pipeline.

    Args:
        depth_np_list (`List[np.ndarray]`):
            List of predicted depth maps, with depth values in the range of [0, 1].
        depth_colored_list (`List[PIL.Image.Image]`):
            List of colorized depth maps, with the shape of [H, W, 3] and values in [0, 255].
    """

    depth_np_list: List[np.ndarray]
    depth_colored_list: List[Image.Image]


class MarigoldPMDepthPipeline(DiffusionPipeline):
    """
    Pipeline for MarigoldPM Monocular Video Depth Estimation.

    This pipeline extends MarigoldDepthPipeline with accelerated frame-by-frame video depth
    estimation via prompt switching. The core idea is to reuse intermediate denoising states
    from the previous frame as the initialization for the next frame, leveraging the high
    spatial correlation between adjacent video frames.

    For a sequence of N frames with M total inference steps, each frame i starts denoising
    from step t_i (where 0 = t_0 <= t_1 <= ... <= t_{N-1} <= M-1). The latent state at
    the switching point is passed to the next frame as its initialization.

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the prediction latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and predictions.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
        scale_invariant (`bool`, *optional*):
            A model property specifying whether the predicted depth maps are scale-invariant.
        shift_invariant (`bool`, *optional*):
            A model property specifying whether the predicted depth maps are shift-invariant.
        default_denoising_steps (`int`, *optional*):
            The minimum number of denoising diffusion steps for reasonable predictions.
        default_processing_resolution (`int`, *optional*):
            The recommended processing resolution from the model config.
    """

    latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        scale_invariant: Optional[bool] = True,
        shift_invariant: Optional[bool] = True,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.register_to_config(
            scale_invariant=scale_invariant,
            shift_invariant=shift_invariant,
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        self.scale_invariant = scale_invariant
        self.shift_invariant = shift_invariant
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

        self.empty_text_embed = None

    @torch.no_grad()
    def __call__(
        self,
        input_images: Sequence[Union[Image.Image, torch.Tensor]],
        denoising_steps: Optional[int] = None,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        init_steps: Optional[List[int]] = None,
    ) -> MarigoldPMVideoOutput:
        """
        Function invoked when calling the pipeline for video depth estimation.

        Args:
            input_images (`Sequence[Image.Image | torch.Tensor]`):
                Sequence of input RGB (or gray-scale) images representing video frames.
            denoising_steps (`int`, *optional*, defaults to `None`):
                Total number of denoising diffusion steps. The default value `None` results
                in automatic selection from the model config.
            processing_res (`int`, *optional*, defaults to `None`):
                Effective processing resolution. When set to `0`, processes at the original
                image resolution. The default value `None` resolves to the optimal value
                from the model config.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize the prediction to match the input resolution.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and predictions.
                Can be one of `bilinear`, `bicubic` or `nearest`.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size. If set to 0, will auto-detect.
            generator (`torch.Generator`, *optional*, defaults to `None`)
                Random generator for initial noise generation (only used for the first frame).
            color_map (`str`, *optional*, defaults to `"Spectral"`):
                Colormap used to colorize the depth map. Pass `None` to skip.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display a progress bar of diffusion denoising.
            init_steps (`List[int]`, *optional*, defaults to `None`):
                List of initial denoising step indices for each frame. Length must match
                the number of input images. Each value must be in [0, denoising_steps).
                If None, defaults to linear ramp: [0, M//N, 2*M//N, ..., (N-1)*M//N].

        Returns:
            `MarigoldPMVideoOutput`: Output class containing:
            - **depth_np_list** (`List[np.ndarray]`) List of depth maps in [0, 1]
            - **depth_colored_list** (`List[PIL.Image.Image]`) List of colorized depth maps
        """
        n_frames = len(input_images)
        if n_frames == 0:
            raise ValueError("input_images must contain at least one image")

        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution

        assert processing_res >= 0

        # Check if denoising step is reasonable
        self._check_inference_step(denoising_steps)

        # Default init_steps: linear ramp across frames
        if init_steps is None:
            init_steps = [i * denoising_steps // n_frames for i in range(n_frames)]
        else:
            if len(init_steps) != n_frames:
                raise ValueError(
                    f"init_steps length ({len(init_steps)}) must match number of frames ({n_frames})"
                )
            for i, s in enumerate(init_steps):
                if not (0 <= s < denoising_steps):
                    raise ValueError(
                        f"init_steps[{i}] = {s} must be in [0, {denoising_steps})"
                    )

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Preprocess all images -----------------
        rgb_norm_list = []
        original_sizes = []
        for input_image in input_images:
            if isinstance(input_image, Image.Image):
                input_image = input_image.convert("RGB")
                rgb = pil_to_tensor(input_image)
                rgb = rgb.unsqueeze(0)
            elif isinstance(input_image, torch.Tensor):
                rgb = input_image
            else:
                raise TypeError(f"Unknown input type: {type(input_image)}")

            original_sizes.append(rgb.shape[-2:])

            if processing_res > 0:
                rgb = resize_max_res(
                    rgb,
                    max_edge_resolution=processing_res,
                    resample_method=resample_method,
                )

            rgb_norm = rgb / 255.0 * 2.0 - 1.0  # [0, 255] -> [-1, 1]
            rgb_norm = rgb_norm.to(self.dtype)
            assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0
            rgb_norm_list.append(rgb_norm)

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()

        # ----------------- Video denoising with prompt switching -----------------
        depth_pred_list = []

        # Set timesteps once for all frames
        self.scheduler.set_timesteps(denoising_steps, device=self.device)
        timesteps = self.scheduler.timesteps  # [M]

        # Build mapping from step index to timestep value
        # timesteps[i] gives the timestep value at denoising step i
        step_to_timestep = {i: t for i, t in enumerate(timesteps)}

        # Initialize: first frame starts from random noise
        # We'll determine the latent shape from the first frame
        first_rgb_latent = self.encode_rgb(rgb_norm_list[0].to(self.device))
        next_init = torch.randn(
            first_rgb_latent.shape,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )

        if show_progress_bar:
            iterable = tqdm(
                range(n_frames),
                total=n_frames,
                leave=True,
                desc="Video frame denoising",
            )
        else:
            iterable = range(n_frames)

        for frame_idx in iterable:
            rgb_norm = rgb_norm_list[frame_idx].to(self.device)
            rgb_latent = self.encode_rgb(rgb_norm)

            # Ensure latent shapes match (in case of resolution differences)
            if rgb_latent.shape != next_init.shape:
                # Resize next_init to match current frame latent shape
                next_init = torch.nn.functional.interpolate(
                    next_init,
                    size=rgb_latent.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            t_i = init_steps[frame_idx]

            # Determine the target timestep to start from
            # The scheduler timesteps are ordered from largest noise to smallest
            # timesteps[t_i] is the actual timestep value at step index t_i
            start_timestep_val = step_to_timestep[t_i]

            # Get the slice of timesteps from t_i onwards
            frame_timesteps = timesteps[t_i:]

            # Use next_init as the starting latent for this frame
            target_latent = next_init.clone()

            # Denoising loop for this frame
            if show_progress_bar:
                frame_iter = tqdm(
                    enumerate(frame_timesteps),
                    total=len(frame_timesteps),
                    leave=False,
                    desc=f"  Frame {frame_idx}",
                )
            else:
                frame_iter = enumerate(frame_timesteps)

            for step_offset, t in frame_iter:
                current_step_idx = t_i + step_offset

                # Check if this is the switching point for the next frame
                if frame_idx + 1 < n_frames:
                    next_frame_start = init_steps[frame_idx + 1]
                    if current_step_idx == next_frame_start:
                        # Save current latent as initialization for next frame
                        next_init = target_latent.clone()

                unet_input = torch.cat([rgb_latent, target_latent], dim=1)

                noise_pred = self.unet(
                    unet_input, t, encoder_hidden_states=self.empty_text_embed
                ).sample

                target_latent = self.scheduler.step(
                    noise_pred, t, target_latent, generator=generator
                ).prev_sample

            # Decode depth for this frame
            depth = self.decode_depth(target_latent)
            depth = torch.clip(depth, -1.0, 1.0)
            depth = (depth + 1.0) / 2.0

            # Resize back to original resolution if needed
            if match_input_res and processing_res > 0:
                orig_h, orig_w = original_sizes[frame_idx]
                depth = resize(
                    depth,
                    (orig_h, orig_w),
                    interpolation=resample_method,
                    antialias=True,
                )

            # Convert to numpy
            depth_np = depth.squeeze().cpu().numpy()
            depth_np = depth_np.clip(0, 1)
            depth_pred_list.append(depth_np)

        torch.cuda.empty_cache()

        # ----------------- Colorize depth maps -----------------
        depth_colored_list = []
        if color_map is not None:
            for depth_np in depth_pred_list:
                depth_colored = colorize_depth_maps(
                    depth_np, 0, 1, cmap=color_map
                ).squeeze()
                depth_colored = (depth_colored * 255).astype(np.uint8)
                depth_colored_hwc = chw2hwc(depth_colored)
                depth_colored_img = Image.fromarray(depth_colored_hwc)
                depth_colored_list.append(depth_colored_img)
        else:
            depth_colored_list = [None] * len(depth_pred_list)

        return MarigoldPMVideoOutput(
            depth_np_list=depth_pred_list,
            depth_colored_list=depth_colored_list,
        )

    def _check_inference_step(self, n_step: int) -> None:
        """
        Check if denoising step is reasonable.
        Args:
            n_step (`int`): denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if "trailing" != self.scheduler.config.timestep_spacing:
                logging.warning(
                    f"The loaded `DDIMScheduler` is configured with `timestep_spacing="
                    f'"{self.scheduler.config.timestep_spacing}"`; the recommended setting is `"trailing"`. '
                    f"This change is backward-compatible and yields better results. "
                    f"Consider using `prs-eth/marigold-depth-v1-1` for the best experience."
                )
            else:
                if n_step > 10:
                    logging.warning(
                        f"Setting too many denoising steps ({n_step}) may degrade the prediction; consider relying on "
                        f"the default values."
                    )
            if not self.scheduler.config.rescale_betas_zero_snr:
                logging.warning(
                    f"The loaded `DDIMScheduler` is configured with `rescale_betas_zero_snr="
                    f"{self.scheduler.config.rescale_betas_zero_snr}`; the recommended setting is True. "
                    f"Consider using `prs-eth/marigold-depth-v1-1` for the best experience."
                )
        elif isinstance(self.scheduler, LCMScheduler):
            logging.warning(
                "DeprecationWarning: LCMScheduler will not be supported in the future. "
                "Consider using `prs-eth/marigold-depth-v1-1` for the best experience."
            )
            if n_step > 10:
                logging.warning(
                    f"Setting too many denoising steps ({n_step}) may degrade the prediction; consider relying on "
                    f"the default values."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt.
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
    ) -> torch.Tensor:
        """
        Perform a single prediction without ensembling (standard single-image inference).

        Args:
            rgb_in (`torch.Tensor`): Input RGB image.
            num_inference_steps (`int`): Number of diffusion denoising steps.
            generator (`torch.Generator`): Random generator for initial noise.
            show_pbar (`bool`): Display a progress bar.

        Returns:
            `torch.Tensor`: Predicted depth.
        """
        device = self.device
        rgb_in = rgb_in.to(device)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        rgb_latent = self.encode_rgb(rgb_in)

        target_latent = torch.randn(
            rgb_latent.shape,
            device=device,
            dtype=self.dtype,
            generator=generator,
        )

        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        ).to(device)

        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            unet_input = torch.cat([rgb_latent, target_latent], dim=1)
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample
            target_latent = self.scheduler.step(
                noise_pred, t, target_latent, generator=generator
            ).prev_sample

        depth = self.decode_depth(target_latent)
        depth = torch.clip(depth, -1.0, 1.0)
        depth = (depth + 1.0) / 2.0
        return depth

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`): Input RGB image.

        Returns:
            `torch.Tensor`: Image latent.
        """
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        rgb_latent = mean * self.latent_scale_factor
        return rgb_latent

    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`): Depth latent.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        depth_latent = depth_latent / self.latent_scale_factor
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean
