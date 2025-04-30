import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import os
import tifffile as tiff
import tifffile
from pathlib import Path
from torch.cuda import amp
import pandas as pd
import torch
import torchvision.transforms as T
from torch.cuda import amp
# ---------- Setup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class twoConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(twoConv, self).__init__()  # Fixed class name
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
class UNet(nn.Module):
    def __init__(self, in_channel=4, out_channel=1, features=[64, 128, 256, 512]):  # Changed in_channel to 4 for R,G,B,IR
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down Part of UNet
        for feature in features:
            self.downs.append(twoConv(in_channel, feature))
            in_channel = feature

        # Upsampling part
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(twoConv(feature*2, feature))

        self.bottleneck = twoConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channel, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        return self.final_conv(x)

def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).
    Args:
        mask (np.ndarray): 2D binary mask (0s and 1s).
    Returns:
        str: RLE-encoded string, or a single space " " if mask is all zeros.
    """
    if np.sum(mask) == 0:
        return " "  # As it seems that kaggle reject nulls. We'll handle cloud-free images with empty spaces.

    pixels = mask.flatten(order='F')  # Flatten in column-major order
    pixels = np.concatenate([[0], pixels, [0]])  # Add padding to detect transitions
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1  # Get transition indices
    runs[1::2] -= runs[::2]  # Compute run lengths
    runs[::2] -= 1  # Make it 0-indexed instead of 1-indexed

    return " ".join(map(str, runs))  # Convert to string format

# ---------- Load the model ----------
model = UNet().to(device)
model.load_state_dict(torch.load("/kaggle/input/test_2/pytorch/default/1/model_best_dice_96.25_loss_9.87.pth", map_location=device))
model.eval()



def read_tifff(file_path):
    try:
        img = tifffile.imread(file_path)
        # If image is 3D, ensure it has 4 channels (R,G,B,NIR)
        if len(img.shape) == 3:
            if img.shape[0] == 4:  # Channels first
                img = np.transpose(img, (1, 2, 0))
            elif img.shape[0] != 4 and img.shape[2] == 4:
                pass  # Already in the right format (H,W,C)
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")

        # Normalize each channel separately to [0, 1]
        img = img.astype(np.float32)
        for i in range(img.shape[2]):
            channel = img[:, :, i]
            min_val = np.min(channel)
            max_val = np.max(channel)
            if max_val > min_val:
                img[:, :, i] = (channel - min_val) / (max_val - min_val)

        return img
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        raise e
# ---------- Helper Functions ----------
def read_tiff(path):
    return np.array(tiff.imread(path))

def predict_mask(image_path, model, device):
    image_np = read_tiff(image_path)

    if image_np.ndim == 2:
        image_np = np.expand_dims(image_np, axis=-1)

    if image_np.shape[2] != 4:
        raise ValueError(f"Expected 4 channels (RGB+IR), got shape: {image_np.shape}")

    image_tensor = torch.tensor(image_np.transpose(2, 0, 1), dtype=torch.float32) / 255.0
    image_tensor = T.Resize((256, 256))(image_tensor)
    input_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad(), amp.autocast():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output)
        pred_mask = (pred_mask > 0.5).float().squeeze().cpu().numpy()
    return pred_mask

def predict_and_display(image_path, pred_mask):
    img = read_tifff(image_path)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img[:, :, :3])
    ax[0].set_title("Input Image (RGB)")

    ax[1].imshow(pred_mask, cmap='gray')
    ax[1].set_title("Predicted Mask (256x256)")
    plt.tight_layout()
    plt.show()

# ---------- Run Prediction Loop ----------
IMAGES_DIR = Path("/kaggle/input/cloud-masking-test-set-satellite-cmp25-course/test/test/data")
CSV_PATH = Path("/kaggle/input/cloud-masking-test-set-satellite-cmp25-course/sample_submission.csv")

# Load sample submission
df = pd.read_csv(CSV_PATH, dtype={"id": str})
df["segmentation"] = ""  # Add empty segmentation column

for idx, row in df.iterrows():
    image_id = row["id"]
    image_path = IMAGES_DIR / f"{image_id}.tif"

    print(f"\n🔍 Processing: {image_id}")
    pred_mask = predict_mask(image_path, model, device)

    # Encode predicted mask (resize to original if needed)
    encoded_mask = rle_encode(pred_mask)
    df.at[idx, "segmentation"] = encoded_mask

    # Show prediction
    predict_and_display(image_path, pred_mask)

# Save results
df["id"] = df["id"].astype(str)
df.to_csv("submission.csv", index=False)
print("✅ Submission saved to submission.csv")