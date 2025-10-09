import torch
import torch.nn as nn
import torch.nn.functional as F

class FlyLoRALinear(nn.Module):
    """
    FlyLoRA Linear Layer
    Implements the FlyLoRA method with implicit routing via fixed sparse random projection.
    """
    def __init__(self, in_features, out_features, r=32, k=8, sparsity_ratio=None, alpha=None, bias_lr=1e-3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r  # total rank
        self.k = k  # activated ranks
        self.alpha = alpha or (2.0 * r)  # scaling factor, default 2r as in LoRA
        self.bias_lr = bias_lr  # learning rate for bias update
        self.sparsity_ratio = sparsity_ratio or (k / r)  # sparsity ratio ρ = p/n

        # Fixed sparse random projection A ∈ R^{r×n}
        # Each row has exactly p non-zero entries sampled from N(0, 1/r^2)
        A = torch.zeros(r, in_features)
        p = max(1, int(in_features * self.sparsity_ratio))  # number of non-zero entries per row
        
        for i in range(r):
            # Randomly select p indices for non-zero entries
            indices = torch.randperm(in_features)[:p]
            # Initialize selected entries with normal distribution
            A[i, indices] = torch.randn(p) * (1.0 / r)
        
        self.register_buffer("A", A)  # frozen during training

        # Trainable up-projection B ∈ R^{m×r}
        self.B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.zeros_(self.B)

        # Expert-wise bias term for load balancing d ∈ R^r
        self.d = nn.Parameter(torch.zeros(r))
        
        # Buffer to track expert assignment counts
        self.register_buffer("assignment_count", torch.zeros(r))
        self.register_buffer("expected_frequency", torch.ones(r) * (k / r))

    @torch.no_grad()
    def update_bias(self):
        """Update expert bias for load balancing: d_i ← d_i + u·sign(̄c_i - c_i)"""
        delta = torch.sign(self.expected_frequency - self.assignment_count)
        self.d.add_(self.bias_lr * delta)
        self.assignment_count.zero_()

    def forward(self, x):
        """
        Forward pass of FlyLoRA:
        1. Project input through frozen sparse A: y = A x
        2. Add expert bias for routing: y' = y + d
        3. Select top-k experts based on |y'|
        4. Compute output using only activated experts in B
        """
        # Project input through frozen sparse A
        y = F.linear(x, self.A)  # (batch_size, r)
        
        # Add expert bias for routing
        y_biased = y + self.d  # (batch_size, r)
        
        # Select top-k experts based on magnitude
        _, indices = torch.topk(y_biased.abs(), self.k, dim=-1)  # (batch_size, k)
        
        # Create mask for activated experts
        mask = torch.zeros_like(y_biased)  # (batch_size, r)
        mask.scatter_(-1, indices, 1.0)  # set top-k positions to 1
        
        # Update assignment counts for load balancing
        if self.training:
            flat_indices = indices.flatten()
            self.assignment_count.index_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float))
        
        # Compute output using only activated experts
        # Only the columns of B corresponding to top-k indices contribute
        activated_y = y * mask  # zero out non-top-k projections
        output = F.linear(activated_y, self.B) * (self.alpha / self.r)
        
        return output