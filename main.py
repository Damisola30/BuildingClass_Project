from typing import Union
from fastapi import FastAPI, File, UploadFile
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import matplotlib.pyplot as plt

# Initialize FastAPI app
app = FastAPI()
from model.model import CNN

# Load the trained model

model = CNN()
model.load_state_dict(torch.load("model/model.pth"))
model.eval()  # Set the model to evaluation mode (no training behavior)

# Define preprocessing transformations for input images
transform = transforms.Compose([
    transforms.Resize((400, 300)),  # Resize to match the training size
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
])

# Class labels corresponding to the model's output
LABELS = ["Bungalow", "High-rise", "Storey-building"]

def preprocess_image(image_bytes):
    """
    Convert the uploaded image bytes to a tensor for model prediction.
    """
    # Open and convert the image to RGB format
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
   
    # Apply preprocessing transformations
    image = transform(image)
    
    # Add a batch dimension (1, C, H, W)
    image = image.unsqueeze(0)
    return image
@app.get("/")
def read_root():
    return {"Project": "Building Classification Project."}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    """
    Endpoint to handle building type prediction from an uploaded image.
    """
    # Read the uploaded image bytes
    image_bytes = await file.read()
    
     # Display the uploaded image
    
    # Preprocess the image
    image_tensor = preprocess_image(image_bytes)

    # Disable gradient calculation for efficiency
    with torch.no_grad():
        output = model(image_tensor)

    # Get the predicted class index
    prediction_index = output.argmax(dim=1).item()

    # Map index to corresponding label
    prediction_label = LABELS[prediction_index]

    # Return the prediction as a JSON response
    return {
        "prediction_label": prediction_label
    }
