import torch
import torch.optim as optim

# Global counter to track total free energy evaluations.
function_eval_count = 0
N = 100
goal = torch.randn(N)
# Example free energy function using a fixed set of Hadamard (or Rademacher) vectors.
def free_energy(params, hadamard_vectors):
    """
    Compute the free energy given parameters and a fixed set of Hadamard vectors.
    Replace this dummy function with your actual free energy computation.
    """
    # For demonstration, use a quadratic loss.
    energy = torch.sum((params - goal)**2)
    # In practice, incorporate your stochastic trace estimation using hadamard_vectors here.
    return energy

# Initialize parameters (example: a vector of parameters).
params = torch.nn.Parameter(torch.ones(N, requires_grad=True))

# Fix a set of Hadamard vectors (here we use random ±1 values for demonstration).
# In your application, choose these vectors appropriately.
num_vectors = 50
hadamard_vectors = torch.randint(0, 2, (num_vectors, 10), dtype=torch.float32) * 2 - 1

# Create the LBFGS optimizer.
optimizer = optim.LBFGS([params], max_iter=40, history_size=10)

# Define the closure function with a counter.
def closure():
    global function_eval_count
    function_eval_count += 1
    optimizer.zero_grad()  # Clear previous gradients.
    energy = free_energy(params, hadamard_vectors)
    energy.backward()      # Compute gradients.
    return energy

# Run the optimization loop.
num_epochs = 2
for epoch in range(num_epochs):
    loss = optimizer.step(closure)
    # For logging, evaluate the free energy outside of the optimizer's internal line search.
    with torch.no_grad():
        current_energy = free_energy(params, hadamard_vectors).item()
    print(f"Epoch {epoch+1:02d}: Free Energy = {current_energy:.6f} {function_eval_count}")

print("Optimized parameters:", params.data)
print("Total free energy function evaluations:", function_eval_count)
