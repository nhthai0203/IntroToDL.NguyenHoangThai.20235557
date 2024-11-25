import argparse
import torch
from torchvision import transforms
from PIL import Image
import os
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Argument parser for command-line usage
parser = argparse.ArgumentParser(description="Image Segmentation Inference Script")
parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
parser.add_argument("--checkpoint_path", type=str, default= 'model.pth', help="Path to the model checkpoint")
parser.add_argument("--output_path", type=str, default="output_segmented.png", help="Path to save the output image")

args = parser.parse_args()

# Check if the input image exists
if not os.path.exists(args.image_path):
    raise FileNotFoundError(f"Input image {args.image_path} not found.")

# Load the model
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3
)
model.load_state_dict(torch.load(args.checkpoint_path, map_location=torch.device('cpu')))
model.eval()

# Preprocess the image
transform = transforms.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

image = Image.open(args.image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Run infer
with torch.no_grad():
    output = model(input_tensor)

# Postprocess the output
output = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()  # Get class labels
output_image = Image.fromarray((output * 255).astype('uint8'))  # Scale to 0-255 for saving
output_image.save(args.output_path)

print(f"Segmented image saved to {args.output_path}")