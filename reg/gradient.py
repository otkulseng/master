import torch


def eigenvalue_perturbation_gradient(L: torch.Tensor, Q: torch.Tensor):
    """
    N is here assumed to be the indices of all order parameters.
    Args:
        L (torch.Tensor): Eigenvalues. Shape (Batch, 4N)
        Q (torch.Tensor): Eigenvectors. Shape (Batch, 4N, 4N)
    """




    # K is actually a very sparse matrix sized 4N by 4N. This performs the product KQ
    # leveraging sparsity.
    #







    
