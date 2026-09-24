from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
import io
import torch.nn.functional as F
from model import SiameseNetwork

app = FastAPI(title="High-Resolution Image Verification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
model.eval()

try:
    model.load_state_dict(torch.load("siamese_model.pth", map_location=device))
    print("Loaded pre-trained weights.")
except FileNotFoundError:
    print("No pre-trained weights found, using uninitialized model.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_image(file_bytes):
    """
    Helper function to load and preprocess an uploaded image.
    """
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0) # Add batch dimension
    return tensor.to(device)

@app.post("/verify")
async def verify_images(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    """
    Endpoint that accepts two images and computes their similarity score based on the Siamese Network.
    """
    img1_bytes = await image1.read()
    img2_bytes = await image2.read()
    
    tensor1 = process_image(img1_bytes)
    tensor2 = process_image(img2_bytes)
    
    with torch.no_grad():
        output1, output2 = model(tensor1, tensor2)
        
    distance = F.pairwise_distance(output1, output2).item()
    
    similarity = torch.exp(torch.tensor(-distance)).item()
    
    return {
        "euclidean_distance": distance,
        "similarity_score": similarity,
        "match": similarity > 0.25
    }
