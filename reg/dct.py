import torch
import math


def dct(x: torch.Tensor) -> torch.Tensor:
    # From signal to coefficients (expand x in cosine waves)
    N = x.shape[-1]
    n = torch.arange(N).unsqueeze(0)
    k = torch.arange(N).unsqueeze(1)

    basis = torch.cos(torch.pi / N * (n + 0.5) * k)
    X = torch.matmul(x, basis.T.to(x.dtype))
    X = X * math.sqrt(2 / N)
    X[..., 0] = X[..., 0] / math.sqrt(2.0)
    return X

def idct(X: torch.Tensor) -> torch.Tensor:
    """ From coefficients to signal (expand coefficients * cosine waves)

    Args:
        X (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    N = X.shape[-1]
    n = torch.arange(N).unsqueeze(0)
    k = torch.arange(N).unsqueeze(1)
    basis = torch.cos(torch.pi / N * (n + 0.5) * k)

    X_adj = X.clone()
    X_adj[..., 0] = X_adj[..., 0] * math.sqrt(2.0)

    x = torch.matmul(X_adj, basis.to(X_adj.dtype)) * math.sqrt(2.0 / N)
    return x

def dst(x: torch.Tensor) -> torch.Tensor:

    N = x.shape[-1]

    n = torch.arange(N).unsqueeze(0)
    k = torch.arange(1, N+1).unsqueeze(1)
    B = torch.sin(math.pi / N * (n + 0.5) * k)
    B = B * math.sqrt(2.0 / N)
    return torch.matmul(x, B.T.to(x.dtype))

def idst(X: torch.Tensor) -> torch.Tensor:
    N = X.shape[-1]
    n = torch.arange(N).unsqueeze(0)
    k = torch.arange(1, N+1).unsqueeze(1)
    B = torch.sin(math.pi / N * (n + 0.5) * k)
    B = B * math.sqrt(2.0 / N)

    return torch.matmul(X, B.to(X.dtype))