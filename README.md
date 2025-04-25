# Fashion Product Classifier

A deep learning-based application that classifies fashion product images into multiple attributes:
- Article Type
- Base Color
- Season
- Gender

## Features

- Multi-task classification using ResNet50
- Streamlit web interface for easy interaction
- FastAPI backend for API access
- Support for both CPU and GPU inference

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fashion-product-classifier.git
cd fashion-product-classifier
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Streamlit Web Interface

Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```
Access the interface at: http://localhost:8501

### FastAPI Server

Run the API server:
```bash
python api.py
```
Access the API documentation at: http://localhost:8000/docs

### API Endpoints

- `GET /`: Root endpoint with API information
- `POST /predict`: Upload an image to get predictions

Example API request:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/image.jpg"
```

## Project Structure

```
fashion-product-classifier/
├── api.py                 # FastAPI server
├── streamlit_app.py       # Streamlit web interface
├── model.py              # Model architecture
├── data_loader.py        # Data loading utilities
├── train.py              # Training script
├── label_mappings.json   # Label mappings
├── requirements.txt      # Dependencies
└── README.md            # Documentation
```

## Model Architecture

The model uses a ResNet50 backbone with custom output layers for each attribute:
- Article Type classification
- Base Color classification
- Season classification
- Gender classification

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: [Fashion Product Images Dataset](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset)
- PyTorch and torchvision for the deep learning framework
- FastAPI and Streamlit for the web interfaces 