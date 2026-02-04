# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import numpy as np
import torch

from cosmos_transfer2._src.imaginaire.auxiliary.guardrail.common import presets as guardrail_presets
from cosmos_transfer2._src.imaginaire.flags import SMOKE
from cosmos_transfer2._src.imaginaire.lazy_config.lazy import LazyConfig
from cosmos_transfer2._src.imaginaire.utils import distributed, log, misc
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.experiment.experiment_list import EXPERIMENTS
from cosmos_transfer2._src.transfer2.inference.inference_pipeline import ControlVideo2WorldInference
from cosmos_transfer2._src.transfer2.inference.utils import compile_tokenizer_if_enabled
from cosmos_transfer2._src.transfer2.utils import fp8_utils
from cosmos_transfer2.config import (
    MODEL_CHECKPOINTS,
    InferenceArguments,
    ModelKey,
    SetupArguments,
    is_rank0,
    path_to_str,
)


class Control2WorldInference:
    # Default state_t value (24 latent frames = 93 pixel frames)
    DEFAULT_STATE_T = 24
    
    def __init__(
        self,
        args: SetupArguments,
        batch_hint_keys: list[str],
        state_t: int = DEFAULT_STATE_T,
        use_cuda_graphs: bool = False,
        use_cfg_batching: bool = False,
        use_fp8: bool = False,
    ) -> None:
        """
        Initialize Control2World inference.
        
        Args:
            args: Setup arguments for inference
            batch_hint_keys: List of control types (e.g., ["depth"])
            state_t: Number of latent temporal frames. Controls video length:
                     - state_t=2 → 5 pixel frames
                     - state_t=4 → 13 pixel frames  
                     - state_t=7 → 25 pixel frames
                     - state_t=24 → 93 pixel frames (default)
                     Formula: pixel_frames = (state_t - 1) * 4 + 1
            use_cuda_graphs: Whether to use CUDA Graphs for inference.
                     Dramatically reduces kernel launch overhead but requires
                     fixed input shapes (controlled via state_t).
            use_cfg_batching: Whether to use CFG batching optimization.
                     Batches conditioned/unconditioned passes and caches control hints
                     for ~35-45% speedup. Should produce identical outputs.
            use_fp8: Whether to use FP8 precision for inference.
                     Leverages FP8 Tensor Cores on Blackwell GPUs for ~2x speedup.
                     Requires Transformer Engine and compute capability >= 8.9.
                     Falls back to BF16 if not available.
        """
        log.debug(f"{args.__class__.__name__}({args})({batch_hint_keys})")
        self.setup_args = args
        self.batch_hint_keys = batch_hint_keys
        self.state_t = state_t
        self.use_cuda_graphs = use_cuda_graphs
        self.use_cfg_batching = use_cfg_batching
        self.use_fp8 = use_fp8
        
        # Check FP8 availability and create recipe
        if self.use_fp8:
            if not fp8_utils.is_fp8_available():
                log.warning("FP8 requested but not available - falling back to BF16")
                self.use_fp8 = False
                self.fp8_recipe = None
            else:
                log.info("FP8 inference enabled on Blackwell GPU")
                self.fp8_recipe = fp8_utils.create_fp8_recipe()
        else:
            self.fp8_recipe = None
        
        if len(self.batch_hint_keys) == 1:
            # pyrefly: ignore  # bad-argument-type
            checkpoint = MODEL_CHECKPOINTS[ModelKey(variant=self.batch_hint_keys[0])]
            self.checkpoint_list = [checkpoint.path]
            self.experiment = checkpoint.experiment
        else:
            # pyrefly: ignore  # bad-argument-type
            self.checkpoint_list = [MODEL_CHECKPOINTS[ModelKey(variant=key)].path for key in self.batch_hint_keys]
            self.experiment = "multibranch_720p_t24_spaced_layer4_cr1pt1_rectified_flow_inference"

        log.debug(f"Loading keys for batch hints {self.batch_hint_keys=}")
        torch.enable_grad(False)  # Disable gradient calculations for inference

        self.device_rank = 0

        process_group = None
        # pyrefly: ignore  # unsupported-operation
        if args.context_parallel_size > 1:
            from megatron.core import parallel_state

            distributed.init()

            # pyrefly: ignore  # bad-argument-type
            parallel_state.initialize_model_parallel(context_parallel_size=args.context_parallel_size)
            process_group = parallel_state.get_context_parallel_group()

        if args.enable_guardrails and self.device_rank == 0:
            self.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(
                offload_model_to_cpu=args.offload_guardrail_models
            )
            self.video_guardrail_runner = guardrail_presets.create_video_guardrail_runner(
                offload_model_to_cpu=args.offload_guardrail_models
            )
        else:
            # pyrefly: ignore  # bad-assignment
            self.text_guardrail_runner = None
            # pyrefly: ignore  # bad-assignment
            self.video_guardrail_runner = None

        self.benchmark_timer = misc.TrainingTimer()
        
        # Build experiment override options, including state_t if non-default
        exp_override_opts = list(EXPERIMENTS[self.experiment].command_args)  # Copy to avoid mutation
        if state_t != self.DEFAULT_STATE_T:
            exp_override_opts.append(f"model.config.state_t={state_t}")
            log.info(f"Overriding state_t to {state_t} (pixel frames: {(state_t - 1) * 4 + 1})")
        
        # Initialize the inference class
        self.inference_pipeline = ControlVideo2WorldInference(
            registered_exp_name=EXPERIMENTS[self.experiment].registered_exp_name,
            checkpoint_paths=self.checkpoint_list,
            s3_credential_path="",
            exp_override_opts=exp_override_opts,
            process_group=process_group,
            use_cp_wan=args.enable_parallel_tokenizer,
            wan_cp_grid=args.parallel_tokenizer_grid,
            benchmark_timer=self.benchmark_timer if args.benchmark else None,
            use_cuda_graphs=use_cuda_graphs,
        )
        # Set CFG batching optimization flag on pipeline
        self.inference_pipeline.use_cfg_batching = self.use_cfg_batching
        # Set FP8 parameters on pipeline
        self.inference_pipeline.use_fp8 = self.use_fp8
        self.inference_pipeline.fp8_recipe = self.fp8_recipe
        
        if use_cuda_graphs:
            log.info("CUDA Graphs enabled - kernel launches will be captured and replayed")

        compile_tokenizer_if_enabled(self.inference_pipeline, args.compile_tokenizer.value)

        if self.device_rank == 0:
            log.info(f"Found {len(self.batch_hint_keys)} hint keys across all samples")
            if len(self.batch_hint_keys) > 1:
                log.warning(
                    "Loading the multicontrol model. Multicontrol inference is not strictly equal to single control"
                )

            args.output_dir.mkdir(parents=True, exist_ok=True)
            config_path = args.output_dir / "config.yaml"
            # pyrefly: ignore  # bad-argument-type
            LazyConfig.save_yaml(self.inference_pipeline.config, config_path)
            log.info(f"Saved config to {config_path}")

    def generate(self, samples: list[InferenceArguments], output_dir: Path) -> list[str]:
        if SMOKE:
            samples = samples[:1]

        sample_names = [sample.name for sample in samples]
        log.info(f"Generating {len(samples)} samples: {sample_names}")

        output_paths: list[str] = []
        for i_sample, sample in enumerate(samples):
            log.info(f"[{i_sample + 1}/{len(samples)}] Processing sample {sample.name}")
            output_path = self._generate_sample(sample, output_dir, sample_id=i_sample)
            if output_path is not None:
                output_paths.append(output_path)

        if is_rank0() and self.setup_args.benchmark:
            log.info("=" * 50)
            log.info("BENCHMARK RESULTS")
            log.info("=" * 50)
            log.info(f"Benchmark runs:")
            for key, value in self.benchmark_timer.results.items():
                log.info(f"{key}: {value} seconds")
            log.info(f"Average times:")
            for key, value in self.benchmark_timer.compute_average_results().items():
                log.info(f"{key}: {value:.2f} seconds")
            log.info("=" * 50)
        return output_paths

    def generate_batch(
        self,
        samples: list[InferenceArguments],
        output_dir: Path,
        batch_size: int = 2,
    ) -> list[str]:
        """
        Process samples in GPU-parallel batches for improved throughput.

        Unlike generate() which processes samples sequentially, this method
        batches multiple samples together and processes them in parallel on the GPU.

        Requirements:
        - All samples in a batch must have the same resolution
        - All samples must have the same number of frames
        - All samples must use the same hint_keys (control types)

        Args:
            samples: List of InferenceArguments to process
            output_dir: Directory to save outputs
            batch_size: Number of samples to process in parallel (default: 2)

        Returns:
            List of output file paths
        """
        if SMOKE:
            samples = samples[:1]
            batch_size = 1

        sample_names = [sample.name for sample in samples]
        log.info(f"Batch generating {len(samples)} samples with batch_size={batch_size}: {sample_names}")

        output_paths: list[str] = []

        # Process in batches
        for batch_start in range(0, len(samples), batch_size):
            batch_end = min(batch_start + batch_size, len(samples))
            batch_samples = samples[batch_start:batch_end]

            log.info(f"Processing batch {batch_start // batch_size + 1}: samples {batch_start + 1}-{batch_end}")

            batch_outputs = self._generate_batch(batch_samples, output_dir, batch_start)
            output_paths.extend([p for p in batch_outputs if p is not None])

        if is_rank0() and self.setup_args.benchmark:
            log.info("=" * 50)
            log.info("BATCH BENCHMARK RESULTS")
            log.info("=" * 50)
            for key, value in self.benchmark_timer.results.items():
                log.info(f"{key}: {value} seconds")
            log.info("=" * 50)

        return output_paths

    def _generate_batch(
        self,
        batch_samples: list[InferenceArguments],
        output_dir: Path,
        batch_offset: int = 0,
    ) -> list[str | None]:
        """
        Generate videos for a batch of samples in parallel.

        Args:
            batch_samples: List of samples to process together
            output_dir: Output directory
            batch_offset: Offset for sample IDs (for benchmark timing)

        Returns:
            List of output paths (or None for failed samples)
        """
        if not batch_samples:
            return []

        import torch
        batch_size = len(batch_samples)
        log.info(f"Running batch inference for {batch_size} samples")

        # Validate batch compatibility
        torch.cuda.nvtx.range_push("BATCH_PREP_VALIDATE")
        first_sample = batch_samples[0]
        for sample in batch_samples[1:]:
            if sample.resolution != first_sample.resolution:
                raise ValueError(f"All samples must have same resolution. Got {sample.resolution} vs {first_sample.resolution}")
            if sample.max_frames != first_sample.max_frames:
                raise ValueError(f"All samples must have same max_frames. Got {sample.max_frames} vs {first_sample.max_frames}")
            if set(sample.hint_keys) != set(first_sample.hint_keys):
                raise ValueError(f"All samples must have same hint_keys. Got {sample.hint_keys} vs {first_sample.hint_keys}")

        # Run text guardrails for all samples first
        if self.device_rank == 0:
            for sample in batch_samples:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / sample.name
                open(f"{output_path}.json", "w").write(sample.model_dump_json())

                if self.text_guardrail_runner is not None:
                    if not guardrail_presets.run_text_guardrail(sample.prompt, self.text_guardrail_runner):
                        log.critical(f"Guardrail blocked prompt for {sample.name}")
                        if not self.setup_args.keep_going:
                            raise Exception(f"Guardrail blocked: {sample.prompt}")

        # Build control weights (same for all samples in batch for now)
        control_weight = ""
        for key in self.batch_hint_keys:
            control_weight += first_sample.control_weight_dict.get(key, "0.0") + ","
        control_weight = control_weight[:-1]
        torch.cuda.nvtx.range_pop()

        # Run batched inference through the pipeline
        with self.benchmark_timer("generate_img2world_batch"):
            output_videos, control_video_dicts, fps_list = self.inference_pipeline.generate_img2world_batch(
                video_paths=[path_to_str(s.video_path) for s in batch_samples],
                prompts=[s.prompt for s in batch_samples],
                negative_prompts=[s.negative_prompt for s in batch_samples],
                guidance=first_sample.guidance,  # Same for batch
                seeds=[s.seed for s in batch_samples],
                resolution=first_sample.resolution,
                control_weight=control_weight,
                hint_key=first_sample.hint_keys,
                input_control_video_paths_list=[s.control_modalities for s in batch_samples],
                num_steps=first_sample.num_steps,
                max_frames=first_sample.max_frames,
            )

        # Save outputs
        import torch
        torch.cuda.nvtx.range_push("SAVE_BATCH_OUTPUTS")
        output_paths: list[str | None] = []
        for i, (sample, output_video, control_video_dict, fps) in enumerate(
            zip(batch_samples, output_videos, control_video_dicts, fps_list)
        ):
            output_path = output_dir / sample.name
            ext = "mp4" if output_video.shape[1] > 1 else "jpg"  # Check temporal dim

            if self.device_rank == 0:
                # Normalize and save
                output_video_norm = (1.0 + output_video) / 2
                save_img_or_video(output_video_norm, str(output_path), fps=fps)

                # Save control videos
                for key in control_video_dict:
                    control_norm = (1.0 + control_video_dict[key]) / 2
                    save_img_or_video(control_norm, f"{output_path}_control_{key}", fps=fps)

                # Save prompt
                with open(f"{output_path}.txt", "w") as f:
                    f.write(sample.prompt)

                log.success(f"Generated video saved to {output_path}.{ext}")
                output_paths.append(f"{output_path}.{ext}")
            else:
                output_paths.append(None)
        torch.cuda.nvtx.range_pop()

        torch.cuda.empty_cache()
        return output_paths

    def _generate_sample(self, sample: InferenceArguments, output_dir: Path, sample_id: int = 0) -> str | None:
        log.debug(f"{sample.__class__.__name__}({sample})")
        output_path = output_dir / sample.name

        assert sample.prompt is not None
        prompt: str = sample.prompt

        assert sample.negative_prompt is not None
        negative_prompt: str = sample.negative_prompt

        if self.device_rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            open(f"{output_path}.json", "w").write(sample.model_dump_json())
            log.info(f"Saved arguments to {output_path}.json")

            with self.benchmark_timer("text_guardrail"):
                # run text guardrail on the prompt
                if self.text_guardrail_runner is not None:
                    log.info("Running guardrail check on prompt...")

                    if not guardrail_presets.run_text_guardrail(prompt, self.text_guardrail_runner):
                        message = f"Guardrail blocked generation. Prompt: {prompt}"
                        log.critical(message)
                        if self.setup_args.keep_going:
                            return None
                        else:
                            raise Exception(message)
                    else:
                        log.success("Passed guardrail on prompt")

                    if not guardrail_presets.run_text_guardrail(
                        negative_prompt,
                        self.text_guardrail_runner,
                    ):
                        message = f"Guardrail blocked generation. Negative prompt: {negative_prompt}"
                        log.critical(message)
                        if self.setup_args.keep_going:
                            return None
                        else:
                            raise Exception(message)
                    else:
                        log.success("Passed guardrail on negative prompt")
                elif self.text_guardrail_runner is None:
                    log.warning("Guardrail checks on prompt are disabled")

        input_control_video_paths = sample.control_modalities
        log.info(f"Processing the following paths: {input_control_video_paths}")

        sigma_max = None if sample.sigma_max is None else float(sample.sigma_max)

        # control_weight is a string because of multi-control
        control_weight = ""
        for key in self.batch_hint_keys:
            # pyrefly: ignore  # missing-attribute
            control_weight += sample.control_weight_dict.get(key, "0.0") + ","
        control_weight = control_weight[:-1]

        if self.setup_args.benchmark:
            torch.cuda.synchronize()

        with self.benchmark_timer("generate_img2world"):
            # Run model inference
            output_video, control_video_dict, mask_video_dict, fps, _ = self.inference_pipeline.generate_img2world(
                # pyrefly: ignore  # bad-argument-type
                video_path=path_to_str(sample.video_path),
                prompt=prompt,
                negative_prompt=negative_prompt,
                image_context_path=path_to_str(sample.image_context_path),
                context_frame_idx=sample.context_frame_index,
                max_frames=sample.max_frames,
                guidance=sample.guidance,
                seed=sample.seed,
                resolution=sample.resolution,
                control_weight=control_weight,
                sigma_max=sigma_max,
                hint_key=sample.hint_keys,
                # pyrefly: ignore  # bad-argument-type
                input_control_video_paths=input_control_video_paths,
                show_control_condition=sample.show_control_condition,
                seg_control_prompt=sample.seg_control_prompt,
                show_input=sample.show_input,
                keep_input_resolution=not sample.not_keep_input_resolution,
                preset_blur_strength=sample.preset_blur_strength,
                preset_edge_threshold=sample.preset_edge_threshold,
                num_conditional_frames=sample.num_conditional_frames,
                num_video_frames_per_chunk=sample.num_video_frames_per_chunk,
                num_steps=sample.num_steps,
            )
            if self.setup_args.benchmark:
                torch.cuda.synchronize()

        if output_video.shape[2] == 1:
            ext = "jpg"
        else:
            ext = "mp4"

        # Save video/image
        if self.device_rank == 0:
            torch.cuda.nvtx.range_push("SAVE_SEQUENTIAL_OUTPUT")
            output_video = (1.0 + output_video[0]) / 2
            for key in control_video_dict:
                control_video_dict[key] = (1.0 + control_video_dict[key][0]) / 2
                save_img_or_video(control_video_dict[key], f"{output_path}_control_{key}", fps=fps)
                log.info(f"{key} control video saved to {output_path}_control_{key}.{ext}")

            with self.benchmark_timer("video_guardrail"):
                for key in mask_video_dict:
                    save_img_or_video(mask_video_dict[key], f"{output_path}_mask_{key}", fps=fps)
                    log.info(f"Mask for {key} saved to {output_path}_mask_{key}.{ext}")
                # run video guardrail on the video
                if self.video_guardrail_runner is not None:
                    log.info("Running guardrail check on video...")
                    frames = (output_video * 255.0).clamp(0.0, 255.0).to(torch.uint8)
                    frames = frames.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)  # (T, H, W, C)
                    processed_frames = guardrail_presets.run_video_guardrail(frames, self.video_guardrail_runner)
                    if processed_frames is None:
                        if self.setup_args.keep_going:
                            torch.cuda.nvtx.range_pop()
                            return None
                        else:
                            raise Exception("Guardrail blocked video2world generation.")
                    else:
                        log.success("Passed guardrail on generated video")

                    # Convert processed frames back to tensor format
                    processed_video = torch.from_numpy(processed_frames).float().permute(3, 0, 1, 2) / 255.0
                    output_video = processed_video.to(output_video.device, dtype=output_video.dtype)
                else:
                    log.warning("Guardrail checks on video are disabled")

            # Remove batch dimension and normalize to [0, 1] range
            save_img_or_video(output_video, str(output_path), fps=fps)
            # save prompt
            prompt_save_path = f"{output_path}.txt"
            with open(prompt_save_path, "w") as f:
                f.write(sample.prompt)
            log.success(f"Generated video saved to {output_path}.{ext}")
            torch.cuda.nvtx.range_pop()

        if sample_id == 0 and self.setup_args.benchmark:
            # discard first warmup sample from timing
            self.benchmark_timer.reset()

        torch.cuda.empty_cache()
        return f"{output_path}.{ext}"
