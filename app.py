import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import gradio as gr
from model import UNet

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
model.eval()

# Transform for image
transform = T.Compose([
    T.Resize((572, 572)),
    T.ToTensor(),
])

def predict_mask(img_pil):
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()

    binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    binary_mask = Image.fromarray(binary_mask)
    return binary_mask

# Gradio interface
interface = gr.Interface(
    fn=predict_mask,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Image(type="pil", label="Predicted Mask"),
    title="UNet Image Segmentation",
    description="Upload an image and get the segmented mask."
)

interface.launch()
