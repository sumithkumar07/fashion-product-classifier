import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import json
from model import create_model
import io
import numpy as np

# Set page config
st.set_page_config(
    page_title="Fashion Product Classifier",
    page_icon="👔",
    layout="wide"
)

# Custom CSS to improve the appearance
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stImage {
        max-width: 400px;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        background-color: #f0f2f6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box h3 {
        color: #000000;
        margin-bottom: 0.5rem;
        font-weight: bold;
        text-transform: uppercase;
        font-size: 1.2rem;
    }
    .prediction-box p {
        color: #000000;
        margin: 0.5rem 0;
        font-size: 1.1rem;
    }
    .prediction-box .confidence {
        color: #000000;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and label mappings"""
    try:
        # Load label mappings
        with open('label_mappings.json', 'r') as f:
            label_mappings = json.load(f)
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create and load model
        model = create_model(label_mappings)
        checkpoint = torch.load('best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        return model, label_mappings, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

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

def get_prediction(model, image_tensor, device, label_mappings):
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
        st.error(f"Error getting predictions: {str(e)}")
        return None, None

def format_attribute_name(attribute):
    """Format attribute name for display"""
    if attribute == "articleType":
        return "ARTICLE TYPE"
    elif attribute == "baseColour":
        return "BASE COLOR"
    elif attribute == "season":
        return "SEASON"
    elif attribute == "gender":
        return "GENDER"
    return attribute.upper()

def main():
    st.title("Fashion Product Classifier 👔")
    st.write("""
    Upload an image of a fashion product to get predictions for:
    - Article Type
    - Base Color
    - Season
    - Gender
    """)
    
    # Load model
    model, label_mappings, device = load_model()
    if model is None:
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Create columns for layout
            col1, col2 = st.columns(2)
            
            # Display uploaded image
            with col1:
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, use_column_width=True)
            
            # Preprocess image and get predictions
            image_tensor = preprocess_image(image)
            predictions, probabilities = get_prediction(model, image_tensor, device, label_mappings)
            
            if predictions is not None and probabilities is not None:
                # Display predictions
                with col2:
                    st.subheader("Predictions")
                    for attribute, prediction in predictions.items():
                        confidence = probabilities[attribute] * 100
                        formatted_attribute = format_attribute_name(attribute)
                        
                        # Create a prediction box with custom styling
                        st.markdown(f"""
                            <div class="prediction-box">
                                <h3>{formatted_attribute}</h3>
                                <p><b>Prediction:</b> {prediction.title()}</p>
                                <p class="confidence"><b>Confidence:</b> {confidence:.2f}%</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Add a progress bar for confidence
                        st.progress(confidence / 100)
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.write("Please try uploading a different image.")

if __name__ == "__main__":
    main() 