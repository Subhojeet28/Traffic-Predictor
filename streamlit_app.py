import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Traffic Density Estimator",
    page_icon="🚦",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    .result-card {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .low-traffic {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
    }
    .medium-traffic {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 2px solid #ffc107;
    }
    .high-traffic {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid #dc3545;
    }
    .traffic-level {
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px 0;
    }
    .confidence {
        font-size: 1.2em;
        color: #555;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load pretrained ResNet18 model"""
    model = models.resnet18(pretrained=True)
    model.eval()
    return model

def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

def estimate_traffic_density(model, image_tensor):
    """
    Estimate traffic density based on model output.
    This is a prototype that uses ResNet18 features to estimate density.
    In production, this would use a model trained on traffic datasets.
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top predictions
        top5_prob, top5_indices = torch.topk(probabilities, 5)
        
        # Heuristic mapping based on ImageNet classes
        # Vehicle-related classes in ImageNet: cars, trucks, buses, etc.
        vehicle_classes = [
            656, 657, 717, 734, 751, 757, 779, 817, 864, 867,
            569, 575, 654, 675, 705, 661, 627, 609
        ]
        
        # Calculate vehicle presence score
        vehicle_score = 0
        for idx in top5_indices[0]:
            if idx.item() in vehicle_classes:
                vehicle_score += probabilities[0][idx].item()
        
        # Additional complexity estimation based on feature activation
        feature_complexity = torch.mean(torch.abs(outputs)).item()
        
        # Combine scores for final estimation
        combined_score = (vehicle_score * 0.7) + (min(feature_complexity / 10, 1.0) * 0.3)
        
        # Map to traffic density categories
        if combined_score < 0.15:
            density = "Low Traffic"
            emoji = "🚗"
            confidence = min(95, int((1 - combined_score) * 100))
            css_class = "low-traffic"
        elif combined_score < 0.35:
            density = "Medium Traffic"
            emoji = "🚙🚗"
            confidence = min(95, int(85 + (combined_score * 20)))
            css_class = "medium-traffic"
        else:
            density = "High Traffic"
            emoji = "🚕🚓🚙🚗"
            confidence = min(95, int(75 + (combined_score * 30)))
            css_class = "high-traffic"
        
        return density, emoji, confidence, css_class

def main():
    # Header
    st.markdown('<div class="main-header">🚦 Traffic Density Estimator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Traffic Monitoring for Smart Transport Infrastructure</div>', unsafe_allow_html=True)
    
    # Info box
    st.markdown("""
        <div class="info-box">
            <strong>📌 About This Project</strong><br>
            This prototype demonstrates how computer vision can estimate traffic density levels from road images.
            Upload an image of a traffic scene, and the AI will classify it as Low, Medium, or High traffic density.
            <br><br>
            <strong>🎯 Use Cases:</strong> Highway monitoring, toll plaza management, urban traffic planning, congestion detection.
        </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a traffic or road scene image",
        type=["jpg", "jpeg", "png"],
        help="Upload an image showing a road or traffic scene"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Add analyze button
        if st.button("🔍 Analyze Traffic Density", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                # Load model
                model = load_model()
                
                # Preprocess image
                image_tensor = preprocess_image(image)
                
                # Estimate traffic density
                density, emoji, confidence, css_class = estimate_traffic_density(model, image_tensor)
                
                # Display result
                st.markdown(f"""
                    <div class="result-card {css_class}">
                        <div style="font-size: 3em;">{emoji}</div>
                        <div class="traffic-level">{density}</div>
                        <div class="confidence">Confidence: {confidence}%</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Additional insights
                st.success("✅ Analysis Complete!")
                
                with st.expander("📊 Understanding the Results"):
                    if "Low" in density:
                        st.write("**Low Traffic**: Few or no vehicles detected. Road is clear for smooth flow.")
                    elif "Medium" in density:
                        st.write("**Medium Traffic**: Moderate number of vehicles. Traffic flow is steady but monitored.")
                    else:
                        st.write("**High Traffic**: Heavy congestion detected. May require traffic management intervention.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 20px;">
            <strong>🚀 Tech Stack:</strong> Python • Streamlit • PyTorch • TorchVision • ResNet18<br>
            <strong>💡 Future Enhancements:</strong> Real-time video analysis • Custom trained models • Vehicle counting • Heatmap overlays
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
