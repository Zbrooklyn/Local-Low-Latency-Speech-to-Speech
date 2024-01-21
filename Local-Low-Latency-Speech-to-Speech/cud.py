import torch

# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available")
    print("CuDNN is enabled:", torch.backends.cudnn.enabled)
else:
    print("GPU is not available")
