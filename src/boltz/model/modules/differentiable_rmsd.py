import torch

def differentiable_rmsd_loss(pred_coords, target_coords):
    """
    Calculates the differentiable, alignment-free RMSD (Mean Squared Error) 
    between two batches of 3D coordinates using the Kabsch algorithm.

    This function returns the (non-square-rooted) RMSD, which is the 
    Mean Squared Error (MSE) after optimal alignment. Minimizing this
    is equivalent to minimizing RMSD and is more stable for gradients.

    NOTE: This function assumes that pred_coords and target_coords contain the
    *same set* of atoms in the *same order* (e.g., all C-alpha atoms,
    or all heavy atoms, with hydrogens masked *before* calling this).

    Args:
        pred_coords (torch.Tensor): Predicted coordinates, shape [B, N, 3].
                                    This tensor should have requires_grad=True.
        target_coords (torch.Tensor): Target coordinates, shape [B, N, 3].
                                      This tensor is treated as a fixed constant.

    Returns:
        torch.Tensor: A scalar tensor representing the mean MSE loss over the batch.
                      This value is (RMSD)^2.
    """

    # 0. Check shapes
    if pred_coords.shape != target_coords.shape:
        raise ValueError(
            f"Input coordinate shapes must be identical. "
            f"Got pred={pred_coords.shape} and target={target_coords.shape}"
        )
    
    # B: Batch size, N: Number of atoms
    B, N, _ = pred_coords.shape
    device = pred_coords.device

    # 1. Center the coordinates (COM)
    # This removes translational differences.
    pred_mean = pred_coords.mean(dim=1, keepdim=True)
    # Detach target_mean, as it's a constant and doesn't need gradients
    target_mean = target_coords.mean(dim=1, keepdim=True).detach()
    
    pred_centered = pred_coords - pred_mean
    target_centered = target_coords - target_mean

    # 2. Compute the covariance matrix H = P^T * W
    # P = pred_centered, W = target_centered
    # H shape will be [B, 3, 3]
    H = pred_centered.transpose(1, 2) @ target_centered

    # 3. Compute SVD(H)
    # This is the core of the Kabsch algorithm.
    # H = U * S * Vh (where Vh = V.T)
    # SVD must be differentiable, which torch.linalg.svd is.
    try:
        U, S, Vh = torch.linalg.svd(H)
    except torch.linalg.LinAlgError:
        # Handle rare SVD convergence errors by returning a simple,
        # non-aligned MSE. This is a fallback.
        loss = (pred_centered - target_centered).pow(2).sum() / (B * N)
        return loss

    # 4. Handle reflections (det(R) < 0)
    # The optimal rotation R = V * U.T (where V = Vh.T)
    # We must check det(R) = det(V) * det(U) to ensure it's a rotation
    # and not a reflection.
    dets = torch.linalg.det(Vh) * torch.linalg.det(U)
    
    # Create a diagonal correction matrix to flip the sign of the
    # 3rd singular vector if det(R) is negative.
    # This is a differentiable way to handle an 'if' statement.
    correction_vec = torch.where(dets < 0, -1.0, 1.0)
    
    # Build a [B, 3, 3] diagonal matrix with [1, 1, correction_vec]
    correction_mat = torch.diag_embed(torch.ones(B, 3, device=device))
    correction_mat[:, 2, 2] = correction_vec

    # 5. Compute the optimal rotation matrix R = V @ correction @ U.T
    # R = Vh.T @ correction @ U.T
    R = Vh.transpose(-1, -2) @ correction_mat @ U.transpose(-1, -2)

    # 6. Apply the optimal rotation to the centered predicted coordinates
    # Our coordinates are row vectors (shape [B, N, 3]),
    # so we apply the rotation as p' = p * R
    pred_aligned = pred_centered @ R

    # 7. Calculate the (L2-norm)^2 / N
    # This is the MSE, or (RMSD)^2
    # sum(dim=-1) sums the (x-x')^2, (y-y')^2, (z-z')^2
    distances_sq = (pred_aligned - target_centered).pow(2).sum(dim=-1) # [B, N]
    
    # mean(dim=-1) averages over all N atoms
    mse_per_structure = distances_sq.mean(dim=-1) # [B]
    
    # Return the mean loss over the whole batch
    loss = mse_per_structure.mean()

    return loss

