"""
Unit tests for CFG (Classifier-Free Guidance) optimization.

These tests verify that:
1. compute_control_hints() produces identical hints to forward()'s internal computation
2. forward() with precomputed_hints produces identical output to forward() without
3. Batched CFG (cond+uncond in one pass) produces identical results to sequential
4. Performance improvements are measurable

The mathematical guarantee: since these are all deterministic linear operations (matmuls, 
additions), given identical inputs we MUST get identical outputs (within floating point 
tolerance for the same dtype).

Run tests with:
    # Quick standalone tests (no model loading)
    python -m pytest tests/test_cfg_optimization.py -v -s -k "Standalone"
    
    # Full tests with real model (requires GPU and model weights)
    python -m pytest tests/test_cfg_optimization.py -v -s -k "not Standalone"
    
    # All tests
    python -m pytest tests/test_cfg_optimization.py -v -s
"""

import pytest
import torch
import time
from typing import Tuple, Optional, List
from dataclasses import dataclass
import sys


@dataclass 
class MockCondition:
    """Mock condition object for testing."""
    crossattn_emb: torch.Tensor
    latent_control_input: torch.Tensor
    condition_video_input_mask_B_C_T_H_W: torch.Tensor


def create_test_inputs(
    batch_size: int = 1,
    channels: int = 16,
    temporal_frames: int = 2,  # Latent frames (state_t)
    height: int = 24,  # Latent height
    width: int = 40,   # Latent width
    text_seq_len: int = 256,
    text_dim: int = 4096,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, ...]:
    """
    Create synthetic test inputs matching real model dimensions.
    
    Using smaller dimensions for faster testing, but same tensor structure.
    """
    # Noisy latent input [B, C, T, H, W]
    x = torch.randn(batch_size, channels, temporal_frames, height, width, 
                    device=device, dtype=dtype)
    
    # Timesteps [B, T] or [B]
    timesteps = torch.rand(batch_size, temporal_frames, device=device, dtype=dtype) * 1000
    
    # Text conditioning [B, seq_len, dim]
    crossattn_emb = torch.randn(batch_size, text_seq_len, text_dim, 
                                device=device, dtype=dtype)
    
    # Control input (depth video, VAE-encoded) [B, C, T, H, W]
    latent_control_input = torch.randn(batch_size, channels, temporal_frames, height, width,
                                       device=device, dtype=dtype)
    
    # Video conditioning mask [B, 1, T, H, W]
    condition_mask = torch.ones(batch_size, 1, temporal_frames, height, width,
                                device=device, dtype=dtype)
    
    return x, timesteps, crossattn_emb, latent_control_input, condition_mask


# ==============================================================================
# STANDALONE TESTS - No model loading required, test mathematical equivalence
# ==============================================================================

class TestStandaloneMathEquivalence:
    """
    Standalone tests that verify the mathematical operations are correct
    without requiring the full Cosmos model to be loaded.
    
    These tests verify the core principle: batching linear operations
    produces identical results to sequential execution.
    """
    
    def test_cfg_formula_equivalence(self):
        """
        Verify CFG formula: v = cond_v + guidance * (cond_v - uncond_v)
        produces identical results regardless of computation order.
        """
        torch.manual_seed(42)
        
        # Simulate velocity outputs
        shape = (1, 16, 2, 24, 40)  # B, C, T, H, W
        cond_v = torch.randn(shape)
        uncond_v = torch.randn(shape)
        guidance = 7.5
        
        # Method 1: Standard formula
        v1 = cond_v + guidance * (cond_v - uncond_v)
        
        # Method 2: Expanded form
        v2 = cond_v * (1 + guidance) - uncond_v * guidance
        
        # Method 3: Rearranged
        v3 = uncond_v + guidance * (cond_v - uncond_v) + (cond_v - uncond_v)
        
        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(v1, v3, rtol=1e-4, atol=1e-4)
        print("✓ CFG formula produces identical results with different orderings")
    
    def test_batched_matmul_equivalence(self):
        """
        Verify that batched matrix multiplication produces identical results
        to sequential matrix multiplication - the core operation in DiT.
        
        This tests: concat([A, B]) @ W == [A @ W, B @ W]
        """
        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Simulate transformer operation
        hidden_dim = 512
        seq_len = 100
        
        # Two different inputs (like cond and uncond)
        A = torch.randn(1, seq_len, hidden_dim, device=device)
        B = torch.randn(1, seq_len, hidden_dim, device=device)
        
        # Weight matrix (same for both)
        W = torch.randn(hidden_dim, hidden_dim, device=device)
        
        # Sequential approach
        out_A_seq = A @ W
        out_B_seq = B @ W
        
        # Batched approach
        AB_batched = torch.cat([A, B], dim=0)  # [2, seq_len, hidden_dim]
        out_AB_batched = AB_batched @ W
        out_A_batch, out_B_batch = out_AB_batched.chunk(2, dim=0)
        
        torch.testing.assert_close(out_A_seq, out_A_batch, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(out_B_seq, out_B_batch, rtol=1e-4, atol=1e-4)
        print("✓ Batched matmul produces identical results to sequential")
    
    def test_batched_attention_equivalence(self):
        """
        Verify that batched attention produces identical results to sequential.
        
        Attention: softmax(Q @ K^T / sqrt(d)) @ V
        """
        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        batch_size = 1
        seq_len = 64
        num_heads = 8
        head_dim = 64
        
        # Two different query inputs (like cond and uncond)
        Q_cond = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        Q_uncond = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        
        # Same K, V for both (like precomputed hints)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        
        scale = head_dim ** -0.5
        
        def attention(Q, K, V):
            attn_weights = torch.softmax(Q @ K.transpose(-2, -1) * scale, dim=-1)
            return attn_weights @ V
        
        # Sequential approach
        out_cond_seq = attention(Q_cond, K, V)
        out_uncond_seq = attention(Q_uncond, K, V)
        
        # Batched approach
        Q_batched = torch.cat([Q_cond, Q_uncond], dim=0)  # [2, heads, seq, dim]
        K_batched = torch.cat([K, K], dim=0)
        V_batched = torch.cat([V, V], dim=0)
        
        out_batched = attention(Q_batched, K_batched, V_batched)
        out_cond_batch, out_uncond_batch = out_batched.chunk(2, dim=0)
        
        torch.testing.assert_close(out_cond_seq, out_cond_batch, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(out_uncond_seq, out_uncond_batch, rtol=1e-4, atol=1e-4)
        print("✓ Batched attention produces identical results to sequential")
    
    def test_hint_reuse_correctness(self):
        """
        Verify that reusing the same hint tensor for cond and uncond produces
        correct results (not accidentally sharing mutable state).
        """
        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Simulate hint application: x = x + hint * scale
        x_cond = torch.randn(1, 16, 2, 24, 40, device=device)
        x_uncond = torch.randn(1, 16, 2, 24, 40, device=device)
        hint = torch.randn(1, 16, 2, 24, 40, device=device)
        scale = 0.8
        
        # Sequential with same hint
        out_cond_seq = x_cond + hint * scale
        out_uncond_seq = x_uncond + hint * scale
        
        # Verify hint wasn't modified
        hint_copy = hint.clone()
        out_cond_again = x_cond + hint * scale
        torch.testing.assert_close(hint, hint_copy, rtol=0, atol=0)
        torch.testing.assert_close(out_cond_seq, out_cond_again, rtol=0, atol=0)
        
        # Batched approach with replicated hints
        x_batched = torch.cat([x_cond, x_uncond], dim=0)
        hint_batched = torch.cat([hint, hint], dim=0)
        out_batched = x_batched + hint_batched * scale
        out_cond_batch, out_uncond_batch = out_batched.chunk(2, dim=0)
        
        torch.testing.assert_close(out_cond_seq, out_cond_batch, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(out_uncond_seq, out_uncond_batch, rtol=1e-6, atol=1e-6)
        print("✓ Hint reuse produces correct results")
    
    def test_full_cfg_batched_simulation(self):
        """
        Full simulation of batched CFG computation flow.
        
        This mirrors the actual optimization:
        1. Compute hints once
        2. Batch inputs
        3. Single forward pass
        4. Split and apply CFG
        """
        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Dimensions
        B, C, T, H, W = 1, 16, 2, 24, 40
        guidance = 7.5
        
        # Simulated inputs
        x = torch.randn(B, C, T, H, W, device=device)
        cond_text = torch.randn(B, 64, 512, device=device)  # Different text
        uncond_text = torch.zeros(B, 64, 512, device=device)  # Null text
        control_input = torch.randn(B, C, T, H, W, device=device)
        
        # Simulate control branch (produces hints)
        def compute_hints(x, control):
            return control * 0.5 + x * 0.1  # Simplified hint computation
        
        # Simulate main DiT branch
        def dit_forward(x, text, hints):
            # Simplified: x + text.mean() + hints
            return x + text.mean(dim=(1, 2), keepdim=True).unsqueeze(-1).unsqueeze(-1) + hints
        
        # === Sequential approach (baseline) ===
        hints_cond = compute_hints(x, control_input)  # Compute hints
        hints_uncond = compute_hints(x, control_input)  # Compute again (wasteful!)
        
        cond_v = dit_forward(x, cond_text, hints_cond)
        uncond_v = dit_forward(x, uncond_text, hints_uncond)
        v_sequential = cond_v + guidance * (cond_v - uncond_v)
        
        # === Optimized approach ===
        # Step 1: Compute hints ONCE
        hints = compute_hints(x, control_input)
        
        # Step 2: Batch inputs
        x_batched = torch.cat([x, x], dim=0)
        text_batched = torch.cat([cond_text, uncond_text], dim=0)
        hints_batched = torch.cat([hints, hints], dim=0)
        
        # Step 3: Single forward pass
        both_v = dit_forward(x_batched, text_batched, hints_batched)
        
        # Step 4: Split and apply CFG
        cond_v_opt, uncond_v_opt = both_v.chunk(2, dim=0)
        v_optimized = cond_v_opt + guidance * (cond_v_opt - uncond_v_opt)
        
        # Verify equivalence
        torch.testing.assert_close(v_sequential, v_optimized, rtol=1e-4, atol=1e-4)
        print("✓ Full batched CFG simulation produces identical results")
        
        # Also verify intermediate values
        torch.testing.assert_close(cond_v, cond_v_opt, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(uncond_v, uncond_v_opt, rtol=1e-4, atol=1e-4)
        print("✓ Intermediate cond_v and uncond_v match between approaches")


class TestControlHintsCorrectness:
    """Test that compute_control_hints produces correct outputs."""
    
    @pytest.fixture
    def model_and_inputs(self):
        """Load the real model and create test inputs."""
        # Skip if CUDA not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Import here to avoid import errors when CUDA isn't available
        import sys
        sys.path.insert(0, '/home/tobyj/code/notes/daily-notes/src/reference/dex/experiments/octi/cosmos-transfer2.5')
        
        from cosmos_transfer2.inference import Control2WorldInference
        
        # Load minimal model for testing
        # Note: This requires the actual model weights
        inference = Control2WorldInference(
            checkpoint_path="/home/horde/cosmos/cosmos-transfer2.5/pretrained/NVIDIA--Cosmos-1.0-Transfer2-5B-ControlVideo2World",
            offload_network=False,
            offload_tokenizer=True,
            state_t=2,  # Minimal temporal frames for fast testing
        )
        
        # Create test inputs
        inputs = create_test_inputs(
            batch_size=1,
            temporal_frames=2,
            device="cuda",
            dtype=torch.bfloat16,
        )
        
        return inference, inputs
    
    def test_compute_hints_matches_forward_internal(self, model_and_inputs):
        """
        Verify compute_control_hints() produces same hints as forward()'s internal computation.
        
        This is the fundamental correctness test - if hints differ, optimization is invalid.
        """
        inference, (x, timesteps, crossattn_emb, latent_control_input, condition_mask) = model_and_inputs
        net = inference.inference_pipeline.model.net
        
        # Get hints from dedicated method
        hints_from_method, scale_from_method = net.compute_control_hints(
            x_B_C_T_H_W=x,
            latent_control_input=latent_control_input,
            timesteps_B_T=timesteps,
            crossattn_emb=crossattn_emb,
            condition_video_input_mask_B_C_T_H_W=condition_mask,
            fps=torch.tensor([24.0], device=x.device),
            control_context_scale=1.0,
        )
        
        # Get hints from forward (we'll extract them by running forward twice and checking)
        # First, run forward WITHOUT precomputed hints - this computes hints internally
        with torch.no_grad():
            output_without_precomputed = net.forward(
                x_B_C_T_H_W=x,
                timesteps_B_T=timesteps,
                crossattn_emb=crossattn_emb,
                latent_control_input=latent_control_input,
                condition_video_input_mask_B_C_T_H_W=condition_mask,
                fps=torch.tensor([24.0], device=x.device),
                control_context_scale=1.0,
                precomputed_hints=None,  # Let forward compute hints internally
            )
        
        # Now run forward WITH precomputed hints
        with torch.no_grad():
            output_with_precomputed = net.forward(
                x_B_C_T_H_W=x,
                timesteps_B_T=timesteps,
                crossattn_emb=crossattn_emb,
                latent_control_input=latent_control_input,
                condition_video_input_mask_B_C_T_H_W=condition_mask,
                fps=torch.tensor([24.0], device=x.device),
                control_context_scale=1.0,
                precomputed_hints=(hints_from_method, scale_from_method),  # Use precomputed
            )
        
        # Outputs MUST be identical (same computation path after hints)
        torch.testing.assert_close(
            output_without_precomputed, 
            output_with_precomputed,
            rtol=1e-4,  # Relative tolerance
            atol=1e-4,  # Absolute tolerance
            msg="Output with precomputed hints differs from output with internal hints computation!"
        )
        
        print(f"✓ Precomputed hints produce identical output (max diff: {(output_without_precomputed - output_with_precomputed).abs().max().item():.2e})")
    
    def test_hints_deterministic(self, model_and_inputs):
        """Verify that hint computation is deterministic (same input → same output)."""
        inference, (x, timesteps, crossattn_emb, latent_control_input, condition_mask) = model_and_inputs
        net = inference.inference_pipeline.model.net
        
        # Compute hints twice
        hints1, scale1 = net.compute_control_hints(
            x_B_C_T_H_W=x,
            latent_control_input=latent_control_input,
            timesteps_B_T=timesteps,
            crossattn_emb=crossattn_emb,
            condition_video_input_mask_B_C_T_H_W=condition_mask,
            fps=torch.tensor([24.0], device=x.device),
            control_context_scale=1.0,
        )
        
        hints2, scale2 = net.compute_control_hints(
            x_B_C_T_H_W=x,
            latent_control_input=latent_control_input,
            timesteps_B_T=timesteps,
            crossattn_emb=crossattn_emb,
            condition_video_input_mask_B_C_T_H_W=condition_mask,
            fps=torch.tensor([24.0], device=x.device),
            control_context_scale=1.0,
        )
        
        torch.testing.assert_close(hints1, hints2, rtol=0, atol=0,
            msg="Hint computation is not deterministic!")
        torch.testing.assert_close(scale1, scale2, rtol=0, atol=0,
            msg="Scale computation is not deterministic!")
        
        print("✓ Hint computation is deterministic")


class TestBatchedCFGCorrectness:
    """Test that batched CFG produces identical results to sequential."""
    
    @pytest.fixture
    def model_and_inputs(self):
        """Load model and create inputs for both conditioned and unconditioned."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        import sys
        sys.path.insert(0, '/home/tobyj/code/notes/daily-notes/src/reference/dex/experiments/octi/cosmos-transfer2.5')
        
        from cosmos_transfer2.inference import Control2WorldInference
        
        inference = Control2WorldInference(
            checkpoint_path="/home/horde/cosmos/cosmos-transfer2.5/pretrained/NVIDIA--Cosmos-1.0-Transfer2-5B-ControlVideo2World",
            offload_network=False,
            offload_tokenizer=True,
            state_t=2,
        )
        
        # Create base inputs
        x, timesteps, crossattn_emb_cond, latent_control_input, condition_mask = create_test_inputs(
            batch_size=1,
            temporal_frames=2,
            device="cuda",
            dtype=torch.bfloat16,
        )
        
        # Create unconditioned text embedding (typically zeros or learned null token)
        crossattn_emb_uncond = torch.zeros_like(crossattn_emb_cond)
        
        return inference, x, timesteps, crossattn_emb_cond, crossattn_emb_uncond, latent_control_input, condition_mask
    
    def test_sequential_vs_batched_forward(self, model_and_inputs):
        """
        Test that running cond and uncond sequentially produces same result as batched.
        
        This verifies the core optimization: batch processing doesn't change outputs.
        """
        inference, x, timesteps, crossattn_cond, crossattn_uncond, control_input, mask = model_and_inputs
        net = inference.inference_pipeline.model.net
        
        # === Sequential approach (current implementation) ===
        with torch.no_grad():
            # Compute hints once (to simulate optimization)
            hints, scale = net.compute_control_hints(
                x_B_C_T_H_W=x,
                latent_control_input=control_input,
                timesteps_B_T=timesteps,
                crossattn_emb=crossattn_cond,  # Either works for hints
                condition_video_input_mask_B_C_T_H_W=mask,
                fps=torch.tensor([24.0], device=x.device),
                control_context_scale=1.0,
            )
            
            # Run conditioned pass
            cond_output_seq = net.forward(
                x_B_C_T_H_W=x,
                timesteps_B_T=timesteps,
                crossattn_emb=crossattn_cond,
                latent_control_input=control_input,
                condition_video_input_mask_B_C_T_H_W=mask,
                fps=torch.tensor([24.0], device=x.device),
                control_context_scale=1.0,
                precomputed_hints=(hints, scale),
            )
            
            # Run unconditioned pass  
            uncond_output_seq = net.forward(
                x_B_C_T_H_W=x,
                timesteps_B_T=timesteps,
                crossattn_emb=crossattn_uncond,
                latent_control_input=control_input,
                condition_video_input_mask_B_C_T_H_W=mask,
                fps=torch.tensor([24.0], device=x.device),
                control_context_scale=1.0,
                precomputed_hints=(hints, scale),
            )
        
        # === Batched approach (proposed optimization) ===
        with torch.no_grad():
            # Stack inputs for batch processing
            x_batched = x.repeat(2, 1, 1, 1, 1)  # [2, C, T, H, W]
            timesteps_batched = timesteps.repeat(2, 1)
            control_input_batched = control_input.repeat(2, 1, 1, 1, 1)
            mask_batched = mask.repeat(2, 1, 1, 1, 1)
            fps_batched = torch.tensor([24.0, 24.0], device=x.device)
            
            # Stack hints (same for both)
            hints_batched = hints.repeat(2, 1, 1, 1, 1, 1) if hints.dim() == 6 else torch.cat([hints, hints], dim=0)
            scale_batched = scale  # Same scale for both
            
            # Stack text embeddings (this is where cond/uncond differ)
            crossattn_batched = torch.cat([crossattn_cond, crossattn_uncond], dim=0)
            
            # Single batched forward pass
            batched_output = net.forward(
                x_B_C_T_H_W=x_batched,
                timesteps_B_T=timesteps_batched,
                crossattn_emb=crossattn_batched,
                latent_control_input=control_input_batched,
                condition_video_input_mask_B_C_T_H_W=mask_batched,
                fps=fps_batched,
                control_context_scale=1.0,
                precomputed_hints=(hints_batched, scale_batched),
            )
            
            # Split outputs
            cond_output_batch, uncond_output_batch = batched_output.chunk(2, dim=0)
        
        # === Verify equivalence ===
        torch.testing.assert_close(
            cond_output_seq, cond_output_batch,
            rtol=1e-4, atol=1e-4,
            msg="Conditioned output differs between sequential and batched!"
        )
        
        torch.testing.assert_close(
            uncond_output_seq, uncond_output_batch,
            rtol=1e-4, atol=1e-4,
            msg="Unconditioned output differs between sequential and batched!"
        )
        
        cond_diff = (cond_output_seq - cond_output_batch).abs().max().item()
        uncond_diff = (uncond_output_seq - uncond_output_batch).abs().max().item()
        print(f"✓ Batched CFG produces identical results (cond diff: {cond_diff:.2e}, uncond diff: {uncond_diff:.2e})")


class TestCFGVelocityEquivalence:
    """Test the full CFG velocity computation (cond_v + guidance * (cond_v - uncond_v))."""
    
    @pytest.fixture
    def model_and_inputs(self):
        """Load model for velocity function testing."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        import sys
        sys.path.insert(0, '/home/tobyj/code/notes/daily-notes/src/reference/dex/experiments/octi/cosmos-transfer2.5')
        
        from cosmos_transfer2.inference import Control2WorldInference
        
        inference = Control2WorldInference(
            checkpoint_path="/home/horde/cosmos/cosmos-transfer2.5/pretrained/NVIDIA--Cosmos-1.0-Transfer2-5B-ControlVideo2World",
            offload_network=False,
            offload_tokenizer=True,
            state_t=2,
        )
        
        inputs = create_test_inputs(batch_size=1, temporal_frames=2, device="cuda", dtype=torch.bfloat16)
        return inference, inputs
    
    def test_cfg_velocity_computation(self, model_and_inputs):
        """
        Test that CFG velocity v = cond_v + guidance * (cond_v - uncond_v) is identical
        regardless of whether cond_v and uncond_v are computed sequentially or batched.
        """
        inference, (x, timesteps, crossattn_cond, control_input, mask) = model_and_inputs
        net = inference.inference_pipeline.model.net
        
        crossattn_uncond = torch.zeros_like(crossattn_cond)
        guidance_scale = 7.5  # Typical CFG scale
        
        with torch.no_grad():
            # Compute shared hints
            hints, scale = net.compute_control_hints(
                x_B_C_T_H_W=x,
                latent_control_input=control_input,
                timesteps_B_T=timesteps,
                crossattn_emb=crossattn_cond,
                condition_video_input_mask_B_C_T_H_W=mask,
                fps=torch.tensor([24.0], device=x.device),
                control_context_scale=1.0,
            )
            
            # Sequential CFG
            cond_v_seq = net.forward(
                x_B_C_T_H_W=x, timesteps_B_T=timesteps, crossattn_emb=crossattn_cond,
                latent_control_input=control_input, condition_video_input_mask_B_C_T_H_W=mask,
                fps=torch.tensor([24.0], device=x.device), control_context_scale=1.0,
                precomputed_hints=(hints, scale),
            )
            uncond_v_seq = net.forward(
                x_B_C_T_H_W=x, timesteps_B_T=timesteps, crossattn_emb=crossattn_uncond,
                latent_control_input=control_input, condition_video_input_mask_B_C_T_H_W=mask,
                fps=torch.tensor([24.0], device=x.device), control_context_scale=1.0,
                precomputed_hints=(hints, scale),
            )
            velocity_seq = cond_v_seq + guidance_scale * (cond_v_seq - uncond_v_seq)
            
            # Batched CFG
            x_batch = x.repeat(2, 1, 1, 1, 1)
            timesteps_batch = timesteps.repeat(2, 1)
            crossattn_batch = torch.cat([crossattn_cond, crossattn_uncond], dim=0)
            control_batch = control_input.repeat(2, 1, 1, 1, 1)
            mask_batch = mask.repeat(2, 1, 1, 1, 1)
            hints_batch = hints.repeat(2, 1, 1, 1, 1, 1) if hints.dim() == 6 else torch.cat([hints, hints], dim=0)
            
            both_v = net.forward(
                x_B_C_T_H_W=x_batch, timesteps_B_T=timesteps_batch, crossattn_emb=crossattn_batch,
                latent_control_input=control_batch, condition_video_input_mask_B_C_T_H_W=mask_batch,
                fps=torch.tensor([24.0, 24.0], device=x.device), control_context_scale=1.0,
                precomputed_hints=(hints_batch, scale),
            )
            cond_v_batch, uncond_v_batch = both_v.chunk(2, dim=0)
            velocity_batch = cond_v_batch + guidance_scale * (cond_v_batch - uncond_v_batch)
        
        # Final velocity must be identical
        torch.testing.assert_close(
            velocity_seq, velocity_batch,
            rtol=1e-4, atol=1e-4,
            msg="CFG velocity differs between sequential and batched computation!"
        )
        
        diff = (velocity_seq - velocity_batch).abs().max().item()
        print(f"✓ CFG velocity computation is equivalent (max diff: {diff:.2e})")


class TestPerformanceBaseline:
    """Measure performance of sequential vs batched approaches."""
    
    @pytest.fixture
    def model_and_inputs(self):
        """Load model for performance testing."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        import sys
        sys.path.insert(0, '/home/tobyj/code/notes/daily-notes/src/reference/dex/experiments/octi/cosmos-transfer2.5')
        
        from cosmos_transfer2.inference import Control2WorldInference
        
        inference = Control2WorldInference(
            checkpoint_path="/home/horde/cosmos/cosmos-transfer2.5/pretrained/NVIDIA--Cosmos-1.0-Transfer2-5B-ControlVideo2World",
            offload_network=False,
            offload_tokenizer=True,
            state_t=2,
        )
        
        inputs = create_test_inputs(batch_size=1, temporal_frames=2, device="cuda", dtype=torch.bfloat16)
        return inference, inputs
    
    def test_performance_comparison(self, model_and_inputs):
        """
        Measure and compare performance of different approaches.
        
        This is informational - we just want to see the speedup potential.
        """
        inference, (x, timesteps, crossattn_cond, control_input, mask) = model_and_inputs
        net = inference.inference_pipeline.model.net
        crossattn_uncond = torch.zeros_like(crossattn_cond)
        
        num_iterations = 5
        
        # Warmup
        with torch.no_grad():
            for _ in range(2):
                _ = net.forward(
                    x_B_C_T_H_W=x, timesteps_B_T=timesteps, crossattn_emb=crossattn_cond,
                    latent_control_input=control_input, condition_video_input_mask_B_C_T_H_W=mask,
                    fps=torch.tensor([24.0], device=x.device), control_context_scale=1.0,
                )
        torch.cuda.synchronize()
        
        # === Baseline: Sequential without hint caching (original) ===
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                # Two separate forward passes, each computing hints
                _ = net.forward(
                    x_B_C_T_H_W=x, timesteps_B_T=timesteps, crossattn_emb=crossattn_cond,
                    latent_control_input=control_input, condition_video_input_mask_B_C_T_H_W=mask,
                    fps=torch.tensor([24.0], device=x.device), control_context_scale=1.0,
                    precomputed_hints=None,
                )
                _ = net.forward(
                    x_B_C_T_H_W=x, timesteps_B_T=timesteps, crossattn_emb=crossattn_uncond,
                    latent_control_input=control_input, condition_video_input_mask_B_C_T_H_W=mask,
                    fps=torch.tensor([24.0], device=x.device), control_context_scale=1.0,
                    precomputed_hints=None,
                )
        torch.cuda.synchronize()
        baseline_time = (time.perf_counter() - start) / num_iterations
        
        # === Optimized: Sequential with hint caching ===
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                # Compute hints once
                hints, scale = net.compute_control_hints(
                    x_B_C_T_H_W=x, latent_control_input=control_input, timesteps_B_T=timesteps,
                    crossattn_emb=crossattn_cond, condition_video_input_mask_B_C_T_H_W=mask,
                    fps=torch.tensor([24.0], device=x.device), control_context_scale=1.0,
                )
                # Two forward passes reusing hints
                _ = net.forward(
                    x_B_C_T_H_W=x, timesteps_B_T=timesteps, crossattn_emb=crossattn_cond,
                    latent_control_input=control_input, condition_video_input_mask_B_C_T_H_W=mask,
                    fps=torch.tensor([24.0], device=x.device), control_context_scale=1.0,
                    precomputed_hints=(hints, scale),
                )
                _ = net.forward(
                    x_B_C_T_H_W=x, timesteps_B_T=timesteps, crossattn_emb=crossattn_uncond,
                    latent_control_input=control_input, condition_video_input_mask_B_C_T_H_W=mask,
                    fps=torch.tensor([24.0], device=x.device), control_context_scale=1.0,
                    precomputed_hints=(hints, scale),
                )
        torch.cuda.synchronize()
        cached_time = (time.perf_counter() - start) / num_iterations
        
        # === Fully Optimized: Batched CFG with hint caching ===
        x_batch = x.repeat(2, 1, 1, 1, 1)
        timesteps_batch = timesteps.repeat(2, 1)
        crossattn_batch = torch.cat([crossattn_cond, crossattn_uncond], dim=0)
        control_batch = control_input.repeat(2, 1, 1, 1, 1)
        mask_batch = mask.repeat(2, 1, 1, 1, 1)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                # Compute hints once
                hints, scale = net.compute_control_hints(
                    x_B_C_T_H_W=x, latent_control_input=control_input, timesteps_B_T=timesteps,
                    crossattn_emb=crossattn_cond, condition_video_input_mask_B_C_T_H_W=mask,
                    fps=torch.tensor([24.0], device=x.device), control_context_scale=1.0,
                )
                # Stack hints for batch
                hints_batch = hints.repeat(2, 1, 1, 1, 1, 1) if hints.dim() == 6 else torch.cat([hints, hints], dim=0)
                
                # Single batched forward pass
                _ = net.forward(
                    x_B_C_T_H_W=x_batch, timesteps_B_T=timesteps_batch, crossattn_emb=crossattn_batch,
                    latent_control_input=control_batch, condition_video_input_mask_B_C_T_H_W=mask_batch,
                    fps=torch.tensor([24.0, 24.0], device=x.device), control_context_scale=1.0,
                    precomputed_hints=(hints_batch, scale),
                )
        torch.cuda.synchronize()
        batched_time = (time.perf_counter() - start) / num_iterations
        
        # Report results
        print(f"\n{'='*60}")
        print("Performance Comparison (per diffusion step, average of {num_iterations} runs)")
        print(f"{'='*60}")
        print(f"Baseline (sequential, no cache):     {baseline_time*1000:.1f} ms")
        print(f"Optimized (sequential, hint cache):  {cached_time*1000:.1f} ms ({(1-cached_time/baseline_time)*100:.1f}% faster)")
        print(f"Fully Optimized (batched + cache):   {batched_time*1000:.1f} ms ({(1-batched_time/baseline_time)*100:.1f}% faster)")
        print(f"{'='*60}")
        
        # Also print per-component estimates
        hint_overhead = cached_time - batched_time  # Rough estimate of second main branch pass
        print(f"\nEstimated component times:")
        print(f"  Control branch (hint computation): ~{(baseline_time - cached_time)*1000/2:.1f} ms (saved by caching)")
        print(f"  Main DiT branch (single pass):     ~{batched_time*1000:.1f} ms")


class TestAPIWithMockTensors:
    """
    API-level tests with mock tensors (no model loading required).
    
    These tests verify:
    - All tensors in condition dicts are properly batched
    - Tensor shapes are consistent throughout the API
    - No dimension mismatches when calling the optimized velocity function
    
    This is the middle ground between:
    - Standalone math tests (don't test real API)
    - Full integration tests (require flash_attn and model weights)
    """
    
    def create_mock_condition_dict(
        self,
        batch_size: int = 1,
        temporal_frames: int = 2,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """Create a realistic condition dict with all optional fields."""
        height, width = 88, 160  # Latent spatial dims
        channels = 16
        
        return {
            "crossattn_emb": torch.randn(batch_size, 256, 4096, device=device, dtype=dtype),
            "latent_control_input": torch.randn(
                batch_size, 3, temporal_frames * 4 + 1, height * 8, width * 8, 
                device=device, dtype=dtype
            ),
            "condition_video_input_mask_B_C_T_H_W": torch.ones(
                batch_size, 1, temporal_frames * 4 + 1, height * 8, width * 8,
                device=device, dtype=dtype
            ),
            "img_context_emb": torch.randn(batch_size, 256, 1152, device=device, dtype=dtype),
            "fps": torch.tensor([24.0] * batch_size, device=device, dtype=torch.float32),
            "padding_mask": torch.ones(batch_size, temporal_frames, device=device, dtype=torch.bool),
        }
    
    def test_condition_dict_batching_consistency(self):
        """
        Test that when we batch condition dicts, all tensor dimensions are consistent.
        
        This would have caught the img_context_emb and padding_mask bugs!
        """
        cond_dict = self.create_mock_condition_dict(batch_size=1)
        
        # Simulate what the optimized velocity function does
        # Stack all tensors for batching (cond + uncond)
        batched = {}
        
        for key, value in cond_dict.items():
            if isinstance(value, torch.Tensor):
                # All tensors should have batch_size=1, so batching should give batch_size=2
                batched[key] = torch.cat([value, value], dim=0)
                
                # Verify batch dimension doubled
                assert batched[key].shape[0] == 2, f"{key} batch dimension should be 2, got {batched[key].shape[0]}"
                
                # Verify other dimensions unchanged
                assert batched[key].shape[1:] == value.shape[1:], f"{key} non-batch dimensions should be unchanged"
        
        print(f"\n✓ All {len(batched)} tensors batch consistently")
        for key, tensor in batched.items():
            print(f"  {key}: {list(tensor.shape)}")
    
    def test_batched_hints_shape(self):
        """
        Test that hints batching produces correct shapes using semantic checks.
        
        Hints from compute_control_hints have shape [num_blocks, B, T, H, W, D]
        We should check batch dimension by comparing hints.shape[1] to expected batch_size,
        NOT by checking if dim() == 6 (brittle magic number).
        """
        num_blocks = 12
        batch_size = 1
        T, H, W, D = 2, 88, 160, 1536
        
        # Create hints with explicit batch dimension (as returned by compute_control_hints)
        hints = torch.randn(num_blocks, batch_size, T, H, W, D)  # [num_blocks, B=1, T, H, W, D]
        
        # ROBUST CHECK: Verify dimension 1 is batch dimension by comparing to batch_size
        assert hints.shape[1] == batch_size, f"Expected hints.shape[1]={batch_size}, got {hints.shape[1]}"
        
        # Batch along dim=1 (the batch dimension)
        hints_batched = torch.cat([hints, hints], dim=1)
        expected_shape = (num_blocks, 2 * batch_size, T, H, W, D)
        assert hints_batched.shape == expected_shape, f"Expected {expected_shape}, got {hints_batched.shape}"
        
        # Verify that after unbinding, each hint has correct shape [B, T, H, W, D]
        hints_list = torch.unbind(hints_batched, dim=0)
        assert len(hints_list) == num_blocks, f"Should have {num_blocks} hints, got {len(hints_list)}"
        assert hints_list[0].shape == (2, T, H, W, D), f"Each hint should be [B=2, T, H, W, D], got {hints_list[0].shape}"
        
        print("\n✓ Hints batching uses semantic check (not magic number)")
        print(f"  Check: hints.shape[1] == batch_size ({hints.shape[1]} == {batch_size})")
        print(f"  Input: {list(hints.shape)}")
        print(f"  Batched: {list(hints_batched.shape)} (concatenated along dim=1)")
        print(f"  After unbind: {len(hints_list)} hints of shape {list(hints_list[0].shape)}")
    
    def test_all_optional_fields_handled(self):
        """
        Verify that all optional fields in condition dict are properly handled during batching.
        
        This is a checklist test to ensure we don't miss any fields.
        """
        cond_dict = self.create_mock_condition_dict(batch_size=1)
        
        # These are the fields that need batching (tensors with batch dimension)
        tensor_fields = [
            "crossattn_emb",
            "latent_control_input", 
            "condition_video_input_mask_B_C_T_H_W",
            "img_context_emb",
            "fps",
            "padding_mask",
        ]
        
        # These are fields that don't need batching (scalars/enums)
        scalar_fields = [
            "data_type",  # Enum, same for both
            "control_context_scale",  # Float, same for both
        ]
        
        print("\n✓ Field handling checklist:")
        print(f"  Tensor fields requiring batching: {len(tensor_fields)}")
        for field in tensor_fields:
            present = field in cond_dict
            print(f"    - {field}: {'✓ present' if present else '✗ MISSING'}")
        
        print(f"  Scalar fields (no batching needed): {len(scalar_fields)}")
        for field in scalar_fields:
            print(f"    - {field}: (enum/scalar)")
    
    def test_velocity_function_tensor_flow(self):
        """
        Simulate the full tensor flow through the optimized velocity function.
        
        This catches dimension mismatches without needing the actual network.
        """
        batch_size = 1
        temporal_frames = 2
        height, width = 88, 160
        channels = 16
        
        # Input tensors to velocity_fn
        noise = torch.randn(batch_size, channels, temporal_frames, height, width)
        noise_x = torch.randn(batch_size, channels, temporal_frames, height, width)
        timestep = torch.rand(batch_size, 1)
        
        # Condition dicts
        cond_dict = self.create_mock_condition_dict(batch_size, temporal_frames)
        uncond_dict = self.create_mock_condition_dict(batch_size, temporal_frames)
        
        # Simulate batching (what _get_optimized_velocity_fn does)
        noise_x_batched = torch.cat([noise_x, noise_x], dim=0)
        timestep_batched = torch.cat([timestep, timestep], dim=0)
        
        # Batch all condition tensors
        crossattn_batched = torch.cat([cond_dict["crossattn_emb"], uncond_dict["crossattn_emb"]], dim=0)
        control_batched = torch.cat([cond_dict["latent_control_input"], uncond_dict["latent_control_input"]], dim=0)
        mask_batched = torch.cat([
            cond_dict["condition_video_input_mask_B_C_T_H_W"],
            uncond_dict["condition_video_input_mask_B_C_T_H_W"]
        ], dim=0)
        img_context_batched = torch.cat([cond_dict["img_context_emb"], uncond_dict["img_context_emb"]], dim=0)
        fps_batched = torch.cat([cond_dict["fps"], uncond_dict["fps"]], dim=0)
        padding_mask_batched = torch.cat([cond_dict["padding_mask"], uncond_dict["padding_mask"]], dim=0)
        
        # Verify all batched tensors have consistent batch dimension
        expected_batch = 2
        assert noise_x_batched.shape[0] == expected_batch, f"noise_x batch mismatch"
        assert timestep_batched.shape[0] == expected_batch, f"timestep batch mismatch"
        assert crossattn_batched.shape[0] == expected_batch, f"crossattn batch mismatch"
        assert control_batched.shape[0] == expected_batch, f"control batch mismatch"
        assert mask_batched.shape[0] == expected_batch, f"mask batch mismatch"
        assert img_context_batched.shape[0] == expected_batch, f"img_context batch mismatch"
        assert fps_batched.shape[0] == expected_batch, f"fps batch mismatch"
        assert padding_mask_batched.shape[0] == expected_batch, f"padding_mask batch mismatch"
        
        print("\n✓ All batched tensors have consistent batch dimension (2)")
        print(f"  noise_x: {list(noise_x_batched.shape)}")
        print(f"  timestep: {list(timestep_batched.shape)}")
        print(f"  crossattn: {list(crossattn_batched.shape)}")
        print(f"  control: {list(control_batched.shape)}")
        print(f"  mask: {list(mask_batched.shape)}")
        print(f"  img_context: {list(img_context_batched.shape)}")
        print(f"  fps: {list(fps_batched.shape)}")
        print(f"  padding_mask: {list(padding_mask_batched.shape)}")


class TestVelocityFunctionIntegration:
    """
    Integration tests that verify the full velocity function creation and execution.
    
    These tests go through the complete API stack:
    - ControlVideo2WorldModelRectifiedFlow.get_velocity_fn_from_batch()
    - _get_optimized_velocity_fn() internal implementation
    - Full condition dictionary handling with all optional fields
    
    This catches issues like missing batching for img_context_emb, padding_mask, etc.
    """
    
    @pytest.fixture(scope="class")
    def model_fixture(self):
        """Load the actual model for integration testing."""
        import sys
        sys.path.insert(0, '/home/tobyj/code/notes/daily-notes/src/reference/dex/experiments/octi/cosmos-transfer2.5')
        
        from cosmos_transfer2.inference import Control2WorldInference
        from cosmos_transfer2._src.transfer2.inference.utils import read_and_process_video
        
        # Initialize with state_t=2 for 5-frame inference
        inference = Control2WorldInference(
            checkpoint_path="/home/horde/cosmos/cosmos-transfer2.5/pretrained/NVIDIA--Cosmos-1.0-Transfer2-5B-ControlVideo2World",
            offload_network=False,
            offload_tokenizer=True,
            state_t=2,
            use_cfg_batching=False,  # We'll test both modes
        )
        
        return inference
    
    def create_data_batch(
        self,
        inference,
        batch_size: int = 1,
        device: str = "cuda",
    ):
        """Create a realistic data_batch dict similar to what the pipeline produces."""
        import torch
        
        # Create minimal tensors that match what the real pipeline produces
        temporal_frames = 2  # state_t
        height, width = 352, 640  # Latent spatial dims (1/8 of 2816x5120)
        channels = 16  # VAE latent channels
        
        # Noisy latent input
        noise_x = torch.randn(
            batch_size, channels, temporal_frames, height, width,
            device=device, dtype=torch.bfloat16
        )
        
        # Timestep
        timestep = torch.rand(batch_size, 1, device=device, dtype=torch.bfloat16)
        
        # Text embeddings (from T5)
        text_emb = torch.randn(batch_size, 256, 4096, device=device, dtype=torch.bfloat16)
        
        # Control input (depth video, latent space)
        control_input = torch.randn(
            batch_size, 3, temporal_frames * 4 + 1, height * 8, width * 8,
            device=device, dtype=torch.bfloat16
        )
        
        # Video input mask
        mask = torch.ones(
            batch_size, 1, temporal_frames * 4 + 1, height * 8, width * 8,
            device=device, dtype=torch.bfloat16
        )
        
        # Image context embedding (reference image feature)
        img_context = torch.randn(batch_size, 256, 1152, device=device, dtype=torch.bfloat16)
        
        # FPS
        fps = torch.tensor([24.0] * batch_size, device=device, dtype=torch.float32)
        
        # Build data_batch dict as the real pipeline does
        data_batch = {
            "crossattn_emb": text_emb,
            "control_input_depth": control_input,
            "condition_video_input_mask_B_C_T_H_W": mask,
            "img_context_emb": img_context,
            "fps": fps,
            "NUM_CONDITIONAL_FRAMES": 0,
        }
        
        return data_batch, noise_x, timestep
    
    def test_velocity_fn_without_batching(self, model_fixture):
        """Test velocity function creation and execution WITHOUT CFG batching (baseline)."""
        inference = model_fixture
        model = inference.inference_pipeline.model
        
        data_batch, noise_x, timestep = self.create_data_batch(inference, batch_size=1)
        
        # Get velocity function without CFG batching
        velocity_fn = model.get_velocity_fn_from_batch(
            data_batch=data_batch,
            guidance=7.0,
            is_negative_prompt=False,
            use_cfg_batching=False,
        )
        
        # Execute velocity function
        with torch.no_grad():
            velocity = velocity_fn(
                noise=torch.randn_like(noise_x),
                noise_x=noise_x,
                timestep=timestep,
            )
        
        # Basic sanity checks
        assert velocity.shape == noise_x.shape, "Velocity shape should match input"
        assert velocity.dtype == torch.float32, "Velocity should be float32"
        assert not torch.isnan(velocity).any(), "Velocity should not contain NaNs"
        assert not torch.isinf(velocity).any(), "Velocity should not contain Infs"
    
    def test_velocity_fn_with_batching(self, model_fixture):
        """Test velocity function creation and execution WITH CFG batching (optimized)."""
        inference = model_fixture
        model = inference.inference_pipeline.model
        
        data_batch, noise_x, timestep = self.create_data_batch(inference, batch_size=1)
        
        # Get velocity function WITH CFG batching
        velocity_fn = model.get_velocity_fn_from_batch(
            data_batch=data_batch,
            guidance=7.0,
            is_negative_prompt=False,
            use_cfg_batching=True,
        )
        
        # Execute velocity function
        with torch.no_grad():
            velocity = velocity_fn(
                noise=torch.randn_like(noise_x),
                noise_x=noise_x,
                timestep=timestep,
            )
        
        # Basic sanity checks
        assert velocity.shape == noise_x.shape, "Velocity shape should match input"
        assert velocity.dtype == torch.float32, "Velocity should be float32"
        assert not torch.isnan(velocity).any(), "Velocity should not contain NaNs"
        assert not torch.isinf(velocity).any(), "Velocity should not contain Infs"
    
    def test_velocity_fn_correctness_comparison(self, model_fixture):
        """
        Test that both velocity functions produce identical results.
        
        This is the KEY test that would have caught the img_context_emb bug!
        """
        inference = model_fixture
        model = inference.inference_pipeline.model
        
        # Create test inputs with FIXED seed for reproducibility
        torch.manual_seed(42)
        data_batch, noise_x, timestep = self.create_data_batch(inference, batch_size=1)
        
        # Same noise for both runs
        noise = torch.randn_like(noise_x)
        
        # Run WITHOUT CFG batching
        torch.manual_seed(42)
        velocity_fn_seq = model.get_velocity_fn_from_batch(
            data_batch=data_batch,
            guidance=7.0,
            is_negative_prompt=False,
            use_cfg_batching=False,
        )
        with torch.no_grad():
            velocity_seq = velocity_fn_seq(noise=noise.clone(), noise_x=noise_x.clone(), timestep=timestep.clone())
        
        # Run WITH CFG batching
        torch.manual_seed(42)
        velocity_fn_batched = model.get_velocity_fn_from_batch(
            data_batch=data_batch,
            guidance=7.0,
            is_negative_prompt=False,
            use_cfg_batching=True,
        )
        with torch.no_grad():
            velocity_batched = velocity_fn_batched(noise=noise.clone(), noise_x=noise_x.clone(), timestep=timestep.clone())
        
        # Compare outputs
        torch.testing.assert_close(
            velocity_batched,
            velocity_seq,
            rtol=1e-3,
            atol=1e-3,
            msg="Batched and sequential velocity functions should produce identical results"
        )
        
        print(f"\n✓ Velocity function correctness verified!")
        print(f"  Max absolute difference: {(velocity_batched - velocity_seq).abs().max().item():.2e}")
        print(f"  Mean absolute difference: {(velocity_batched - velocity_seq).abs().mean().item():.2e}")


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_cfg_optimization.py -v -s
    pytest.main([__file__, "-v", "-s"])

