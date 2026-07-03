import torch
import torch.nn as nn

def verify_norm_sum_inequivalence():
    print("=== Verifying Inequivalence: Sum(Norm(x)) vs Norm(Sum(x)) ===\n")
    
    # 1. Setup parameters
    B, L, D = 2, 4, 128
    h_kv = 2
    G = 4          # Group size (h_q = h_kv * G)
    epsilon = 1e-6
    
    # 2. Simulate the output of Linear Projection (before Norm)
    # Shape: [B, L, h_kv, G, D]
    # This represents the raw query states for all heads
    x_raw = torch.randn(B, L, h_kv, G, D)
    
    # Initialize RMSNorm
    # Note: We use the same RMSNorm instance to ensure gamma/beta are identical
    rms_norm = nn.RMSNorm(D, eps=epsilon)
    
    # 3. Path A: Previous Implementation (Norm then Sum)
    # Logic: Linear(h_q) -> Norm -> Kernel(Sum)
    # Step A1: Apply Norm to each head individually
    x_normed = rms_norm(x_raw)
    # Step A2: Sum over the group dimension (simulating the kernel behavior)
    out_path_a = x_normed.sum(dim=3)  # Shape: [B, L, h_kv, D]
    
    # 4. Path B: Current Implementation (Sum then Norm)
    # Logic: Linear(h_kv) -> Norm
    # Assumption: The weights of Linear(h_kv) are the sum of weights of Linear(h_q)
    #             So x_projected_small = Sum(x_projected_large)
    # Step B1: Sum raw outputs first
    x_summed = x_raw.sum(dim=3)
    # Step B2: Apply Norm to the summed result
    out_path_b = rms_norm(x_summed)   # Shape: [B, L, h_kv, D]
    
    # 5. Compare
    diff = (out_path_a - out_path_b).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Input Shape (Per Group): {x_raw.shape}")
    print(f"Output Shape:            {out_path_a.shape}")
    print(f"-" * 40)
    print(f"Max Difference:  {max_diff:.6f}")
    print(f"Mean Difference: {mean_diff:.6f}")
    print(f"-" * 40)
    
    # 6. Theoretical Explanation Check
    # Example: v1=[10], v2=[10]. 
    # Norm(v1)=1, Norm(v2)=1 -> Sum=2
    # Sum(v1,v2)=[20] -> Norm([20])=1
    # 2 != 1
    
    if max_diff > 1e-4:
        print("❌ Conclusion: The two approaches are MATHEMATICALLY INEQUIVALENT.")
        print("   Reason: RMSNorm is non-linear. Summing normalized vectors does not")
        print("           preserve the unit-norm property, whereas normalizing the sum does.")
    else:
        print("✅ Conclusion: Equivalent (Unexpected)")

if __name__ == "__main__":
    torch.manual_seed(42)
    verify_norm_sum_inequivalence()
