from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import json
from model import create_model
import io
import uvicorn

app = FastAPI(
    title="Fashion Product Classifier API",
    description="API for classifying fashion product images into article type, base color, season, and gender",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model and label mappings
try:
    with open('label_mappings.json', 'r') as f:
        label_mappings = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(label_mappings)
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

def preprocess_image(image):
    """Preprocess the uploaded image"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def get_prediction(image_tensor):
    """Get model predictions for the image"""
    try:
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
        
        predictions = {}
        probabilities = {}
        
        for attribute, output in outputs.items():
            # Get predicted class
            probs = torch.softmax(output, dim=1)[0]
            predicted_idx = torch.argmax(probs).item()
            
            # Get class name
            idx_to_label = {v: k for k, v in label_mappings[attribute]['label_to_idx'].items()}
            predicted_class = idx_to_label[predicted_idx]
            
            # Get probability
            probability = probs[predicted_idx].item()
            
            # Store predictions
            predictions[attribute] = predicted_class
            probabilities[attribute] = probability
        
        return predictions, probabilities
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting predictions: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Fashion Product Classifier API",
        "endpoints": {
            "/predict": "POST endpoint for image classification",
            "/docs": "API documentation"
        }
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Endpoint for making predictions on uploaded images"""
    try:
        # Check if the file is an image
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess and get predictions
        image_tensor = preprocess_image(image)
        predictions, probabilities = get_prediction(image_tensor)
        
        # Format the response
        response = {
            "predictions": predictions,
            "confidence": {k: round(v * 100, 2) for k, v in probabilities.items()}
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 