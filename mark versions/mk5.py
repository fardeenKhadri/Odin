import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Define the model architecture based on the saved model
class DepthProModel(nn.Module):
    def __init__(self):
        super(DepthProModel, self).__init__()
        # Example encoder and decoder structure
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.decoder_depth = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        )
        self.decoder_focallength = nn.Sequential(
            nn.Linear(128 * 256 * 256, 1)  # Adjust input size based on encoder output
        )

    def forward(self, x):
        features = self.encoder(x)
        depth = self.decoder_depth(features)
        # Flatten for the fully connected layer
        features_flat = features.view(features.size(0), -1)
        focallength_px = self.decoder_focallength(features_flat)
        return {"depth": depth, "focallength_px": focallength_px}

# Instantiate the model
model = DepthProModel()

# Load the weights
model_path = "F:/bmsit/trial/checkpoints/depth_pro.pt"
try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    print("Model weights loaded successfully.")
except Exception as e:
    print(f"Error loading model weights: {e}")

model.eval()

# Define a preprocessing transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load and preprocess an image
image_path = "F:/bmsit/c.jpg"
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)

# Run inference
with torch.no_grad():
    prediction = model(image_tensor)
    depth = prediction["depth"].squeeze().numpy()  # Remove batch and channel dimensions
    focallength_px = prediction["focallength_px"].item()  # Extract scalar

# Display input image and depth map
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(depth, cmap="viridis")
plt.title(f"Depth Map\nFocal Length: {focallength_px:.2f}px")
plt.axis("off")

plt.show()
