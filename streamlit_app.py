import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

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
    .vehicle-count {
        font-size: 1.5em;
        color: #333;
        margin: 10px 0;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 20px 0;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 3px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_yolo_model():
    """Load YOLOv8 model for vehicle detection"""
    try:
        # Load YOLOv8 nano model (fastest, good for demo)
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def detect_vehicles(model, image):
    """
    Detect vehicles in the image using YOLOv8
    Returns: vehicle count, annotated image, and vehicle types breakdown
    """
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Vehicle class IDs in COCO dataset
    vehicle_classes = {
        2: 'car',
        3: 'motorcycle', 
        5: 'bus',
        7: 'truck'
    }
    
    # Run detection
    results = model(img_cv, conf=0.25)  # 25% confidence threshold
    
    # Count vehicles
    vehicle_count = 0
    vehicle_breakdown = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
    
    # Process detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls in vehicle_classes:
                vehicle_count += 1
                vehicle_breakdown[vehicle_classes[cls]] += 1
    
    # Get annotated image
    annotated_img = results[0].plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    return vehicle_count, annotated_img, vehicle_breakdown

def classify_traffic_density(vehicle_count):
    """
    Classify traffic density based on vehicle count
    Thresholds can be adjusted based on image size and use case
    """
    if vehicle_count <= 5:
        density = "Low Traffic"
        emoji = "🚗"
        css_class = "low-traffic"
        description = "Clear road with minimal vehicles. Smooth traffic flow."
    elif vehicle_count <= 15:
        density = "Medium Traffic"
        emoji = "🚙🚗"
        css_class = "medium-traffic"
        description = "Moderate traffic with steady flow. Some congestion possible."
    else:
        density = "High Traffic"
        emoji = "🚕🚓🚙🚗"
        css_class = "high-traffic"
        description = "Heavy congestion detected. Significant traffic density."
    
    return density, emoji, css_class, description

def main():
    # Header
    st.markdown('<div class="main-header">🚦 Traffic Density Estimator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Vehicle Detection & Traffic Analysis</div>', unsafe_allow_html=True)
    
    # Info box
    st.markdown("""
        <div class="info-box">
            <strong>📌 About This Project</strong><br>
            This application uses <strong>YOLOv8 AI model</strong> to detect and count vehicles in traffic images.
            It provides accurate traffic density classification based on actual vehicle counts.
            <br><br>
            <strong>🎯 Detection:</strong> Cars, Motorcycles, Buses, Trucks<br>
            <strong>📊 Classification:</strong> Low (≤5 vehicles) | Medium (6-15) | High (16+)
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_yolo_model()
    
    if model is None:
        st.error("Failed to load YOLOv8 model. Please check your installation.")
        st.info("Run: pip install ultralytics")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a traffic or road scene image",
        type=["jpg", "jpeg", "png"],
        help="Upload an image showing a road or traffic scene"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        
        # Add analyze button
        if st.button("🔍 Analyze Traffic Density", type="primary", use_container_width=True):
            with st.spinner("Detecting vehicles..."):
                try:
                    # Detect vehicles
                    vehicle_count, annotated_img, vehicle_breakdown = detect_vehicles(model, image)
                    
                    # Classify traffic density
                    density, emoji, css_class, description = classify_traffic_density(vehicle_count)
                    
                    # Display annotated image
                    with col2:
                        st.image(annotated_img, caption="Detected Vehicles", use_container_width=True)
                    
                    # Display result card
                    st.markdown(f"""
                        <div class="result-card {css_class}">
                            <div style="font-size: 3em;">{emoji}</div>
                            <div class="traffic-level">{density}</div>
                            <div class="vehicle-count">🚗 {vehicle_count} Vehicles Detected</div>
                            <div style="color: #555; margin-top: 10px;">{description}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Vehicle breakdown
                    st.markdown("### 📊 Vehicle Breakdown")
                    
                    cols = st.columns(4)
                    breakdown_emojis = {
                        'car': '🚗',
                        'motorcycle': '🏍️',
                        'bus': '🚌',
                        'truck': '🚚'
                    }
                    
                    for idx, (vehicle_type, count) in enumerate(vehicle_breakdown.items()):
                        with cols[idx]:
                            st.metric(
                                label=f"{breakdown_emojis[vehicle_type]} {vehicle_type.title()}",
                                value=count
                            )
                    
                    # Additional insights
                    st.success("✅ Analysis Complete!")
                    
                    with st.expander("📈 Detailed Analysis"):
                        st.markdown(f"""
                        <div class="metric-box">
                            <strong>Total Vehicles:</strong> {vehicle_count}<br>
                            <strong>Traffic Level:</strong> {density}<br>
                            <strong>Detection Method:</strong> YOLOv8 Object Detection<br>
                            <strong>Confidence Threshold:</strong> 25%
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if vehicle_count <= 5:
                            st.info("💡 **Recommendation**: Traffic is flowing smoothly. No intervention needed.")
                        elif vehicle_count <= 15:
                            st.warning("⚠️ **Recommendation**: Monitor traffic flow. Consider signal optimization if congestion increases.")
                        else:
                            st.error("🚨 **Recommendation**: High congestion detected. Traffic management intervention may be required.")
                    
                    # Adjustable thresholds
                    with st.expander("⚙️ Adjust Classification Thresholds"):
                        st.write("Current thresholds:")
                        st.write("- **Low**: 0-5 vehicles")
                        st.write("- **Medium**: 6-15 vehicles") 
                        st.write("- **High**: 16+ vehicles")
                        st.info("These thresholds can be customized based on road type, image resolution, and specific use cases.")
                
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
                    st.info("Please try with a different image or check if the model is properly loaded.")
    
    else:
        # Example section
        st.markdown("### 💡 How to Use")
        st.markdown("""
        1. **Upload** an image of a traffic scene or road
        2. Click **Analyze Traffic Density** 
        3. View the results with:
           - Vehicle count and detection
           - Traffic density classification
           - Vehicle type breakdown
           - Annotated image showing detected vehicles
        """)
        
        st.info("📸 **Tip**: For best results, use clear images with good lighting and visible vehicles.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 20px;">
            <strong>🚀 Tech Stack:</strong> Python • Streamlit • YOLOv8 • OpenCV • Ultralytics<br>
            <strong>🎯 Model:</strong> YOLOv8n (Nano) - Trained on COCO dataset with 80 object classes<br>
            <strong>✨ Accuracy:</strong> Real-time vehicle detection with ~25% confidence threshold
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
