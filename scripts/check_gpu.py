import torch

print("PyTorch version:", torch.__version__)
print("PyTorch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

    x = torch.randn(3, 3, device="cuda")
    y = torch.randn(3, 3, device="cuda")
    z = x @ y

    print("Tensor device:", z.device)
    print("GPU test passed.")
else:
    print("CUDA is not available.")
