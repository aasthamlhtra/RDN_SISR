import torch
from model import rdn
import os

# Create an args object that matches your training configuration
class Args:
    def __init__(self):
        self.scale = [4]
        self.n_colors = 3
        self.G0 = 64
        self.RDNkSize = 3
        self.RDNconfig = 'B'

# Set device to GPU if available, else fallback to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create args object
args = Args()

# Instantiate the model with args
model = rdn.RDN(args)
model.to(device)

# Fix the path to go up one directory level from src to find experiment
# Use os.path.join for cross-platform compatibility
checkpoint_path = os.path.join('..', 'experiment', 'RDN_X4', 'model', 'model_latest.pt')
print(f"Looking for checkpoint at: {os.path.abspath(checkpoint_path)}")

# Load the state_dict with weights_only=True to address the warning
state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)

# Set to eval mode for inference
model.eval()

# Inspect weights
print("\n--- Model Parameters ---\n")
for name, param in model.state_dict().items():
    print(f"{name}: {param.shape}")