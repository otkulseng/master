import torch
import math
import matplotlib.pyplot as plt

def dct_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the DCT-II (Type-II Discrete Cosine Transform) of x along the last dimension.
    The transform is defined to be orthonormal.

    Args:
        x: Input tensor of shape (..., N)

    Returns:
        Tensor of shape (..., N) with DCT coefficients.
    """
    N = x.shape[-1]
    n = torch.arange(N, device=x.device, dtype=x.dtype).unsqueeze(0)  # shape (1, N)
    k = torch.arange(N, device=x.device, dtype=x.dtype).unsqueeze(1)  # shape (N, 1)
    # Create the cosine basis: cos(pi/N * (n + 0.5) * k)
    basis = torch.cos(math.pi / N * (n + 0.5) * k)  # shape (N, N)

    # Multiply the signal by the basis: note that for a batched input, this
    # performs the transformation along the last dimension.
    X = torch.matmul(x, basis.T)

    # Orthonormal scaling:
    factor = math.sqrt(2.0 / N)
    X = X * factor
    # Adjust the first coefficient (k=0) by 1/sqrt(2)
    X[..., 0] = X[..., 0] / math.sqrt(2)
    return X

def idct_torch(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse DCT (DCT-III) corresponding to the orthonormal DCT-II.

    Args:
        X: DCT coefficients of shape (..., N)

    Returns:
        Reconstructed signal of shape (..., N)
    """
    N = X.shape[-1]
    n = torch.arange(N, device=X.device, dtype=X.dtype).unsqueeze(0)  # shape (1, N)
    k = torch.arange(N, device=X.device, dtype=X.dtype).unsqueeze(1)  # shape (N, 1)
    basis = torch.cos(math.pi / N * (n + 0.5) * k)  # shape (N, N)

    # Undo the adjustment on the first coefficient: multiply back by sqrt(2)
    X_adj = X.clone()
    X_adj[..., 0] = X_adj[..., 0] * math.sqrt(2)

    # Compute the inverse transformation
    x = torch.matmul(X_adj, basis) * math.sqrt(2.0 / N)
    return x

def compress_dct(x: torch.Tensor, num_coeffs: int):
    """
    Compress a signal by performing the DCT, zeroing out higher frequency coefficients,
    and then reconstructing the signal using the inverse DCT.

    Args:
        x: Input signal tensor of shape (N,) or (batch, N)
        num_coeffs: Number of DCT coefficients to keep (must be <= N)

    Returns:
        x_compressed: The reconstructed signal using only the first num_coeffs cosine waves.
        X: The full set of DCT coefficients.
        X_compressed: The DCT coefficients after zeroing out high-frequency terms.
    """
    X = dct_torch(x)

    # Create a mask that keeps only the first num_coeffs coefficients.
    mask = torch.zeros_like(X)
    mask[..., :num_coeffs] = 1.0

    X_compressed = X * mask
    x_compressed = idct_torch(X_compressed)
    return x_compressed, X, X_compressed

# === Demonstration ===
if __name__ == "__main__":
    # Create a 1D equidistant signal, for example a sine wave with added noise.
    N = 1000
    t = torch.linspace(0, 2, steps=N)
    signal = torch.sin(2 * math.pi * 5 * t) + 0.5 * torch.sin(2 * math.pi * 20 * t)

    # Compute the DCT of the signal
    X = dct_torch(signal)

    # Compress the signal by keeping only the first 20 cosine coefficients.
    num_coeffs = 200
    signal_compressed, X_full, X_compressed = compress_dct(signal, num_coeffs)

    # Plot the original and compressed signals
    plt.figure(figsize=(10, 6))
    plt.plot(t.numpy(), signal.numpy(), label='Original Signal')
    plt.plot(t.numpy(), signal_compressed.numpy(), '--', label=f'Compressed (first {num_coeffs} coeffs)')
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("Signal Compression using DCT/IDCT")
    plt.savefig("ims/signal.pdf")

    # Optionally, plot the DCT coefficients
    plt.figure(figsize=(10, 4))
    plt.stem(X_full.numpy(), markerfmt=" ", label="Full DCT Coeffs")
    plt.stem(X_compressed.numpy(), markerfmt=" ", linefmt="r-", basefmt=" ", label="Compressed Coeffs")
    plt.xlabel("Coefficient index")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.title("DCT Coefficients")
    plt.savefig("ims/coefs.pdf")
