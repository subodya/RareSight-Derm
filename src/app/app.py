import streamlit as st
import torch
import sys
import os
import json
import numpy as np
from PIL import Image

# Setup Paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.raresight_net import RareSight

# Page Config
st.set_page_config(page_title="RareSight Demo", layout="wide", page_icon="🔬")

# Load Clinical Descriptions
with open('src/app/class_descriptions.json', 'r') as f:
    DESCRIPTIONS = json.load(f)

CLASSES = {
    0: 'Actinic keratoses', 1: 'Basal cell carcinoma', 2: 'Benign keratosis',
    3: 'Dermatofibroma', 4: 'Melanoma', 5: 'Melanocytic nevi', 6: 'Vascular lesions'
}

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RareSight(device=device)
    
    # Load Trained Weights
    weight_path = 'checkpoints/raresight_best.pth'
    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)
        print("✅ Loaded trained RareSight weights")
    else:
        st.error("⚠️ Model weights not found! Run training script first.")
        
    model.eval()
    return model, device

# Initialize
try:
    model, device = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- UI LAYOUT ---
st.title("RareSight: Pediatric Rare Disease Diagnosis")
st.markdown("### Vision-Language Meta-Learning Framework")

# Sidebar for Config
st.sidebar.header("Episode Configuration")
n_way = st.sidebar.slider("N-Way (Classes)", 2, 5, 3)
k_shot = st.sidebar.slider("K-Shot (Images/Class)", 1, 5, 1)

st.sidebar.markdown("---")
st.sidebar.info("Upload support examples to dynamically teach the model new or rare classes.")

# Main Area
col1, col2 = st.columns([1, 1])

# Left Column: Support Set Upload
with col1:
    st.subheader("1. Define Support Set (The Knowledge)")
    
    selected_classes = st.multiselect(
        "Select Classes for this Diagnosis Episode", 
        list(CLASSES.values()), 
        default=[CLASSES[3], CLASSES[4], CLASSES[5]] # Default: Dermatofibroma vs Melanoma vs Nevi
    )
    
    if len(selected_classes) != n_way:
        st.warning(f"⚠️ Please select exactly {n_way} classes (currently {len(selected_classes)})")
    
    support_data_ready = True
    support_imgs = []
    support_txts = []
    
    # Dynamic Uploader Loop
    tabs = st.tabs(selected_classes) if selected_classes else []
    
    for i, cls_name in enumerate(selected_classes):
        with tabs[i]:
            # Get ID and Description
            cls_id = [k for k, v in CLASSES.items() if v == cls_name][0]
            desc = DESCRIPTIONS[str(cls_id)]
            
            st.caption(f"**Clinical Context:** {desc}")
            
            uploaded_files = st.file_uploader(
                f"Upload {k_shot} images for {cls_name}", 
                accept_multiple_files=True,
                key=f"uploader_{cls_id}"
            )
            
            if uploaded_files and len(uploaded_files) >= k_shot:
                # Process images
                for j in range(k_shot):
                    image = Image.open(uploaded_files[j]).convert('RGB')
                    st.image(image, width=100)
                    
                    # Preprocess for model
                    tensor = model.preprocess(image).unsqueeze(0).to(device)
                    support_imgs.append(tensor)
                    support_txts.append(desc)
            else:
                st.warning(f"Need {k_shot} images.")
                support_data_ready = False

# Right Column: Query and Prediction
with col2:
    st.subheader("2. Patient Diagnosis (The Query)")
    
    query_file = st.file_uploader("Upload Patient Scan", type=['jpg', 'png', 'jpeg'])
    
    if query_file:
        q_image = Image.open(query_file).convert('RGB')
        st.image(q_image, caption="Patient Scan", width=300)
        
        q_tensor = model.preprocess(q_image).unsqueeze(0).to(device)
        
        run_btn = st.button("Run RareSight Analysis", type="primary")
        
        if run_btn and support_data_ready and len(selected_classes) == n_way:
            with st.spinner("Fusion Network Processing..."):
                # Prepare Tensors
                s_tensor = torch.cat(support_imgs)
                
                # Inference
                # Note: We pass just 1 query image
                with torch.no_grad():
                    logits = model(s_tensor, support_txts, q_tensor, n_way, k_shot)
                    probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
                
                # Results
                best_idx = np.argmax(probs)
                pred_cls = selected_classes[best_idx]
                confidence = probs[best_idx] * 100
                
                st.success(f"**Diagnosis: {pred_cls}**")
                st.metric("Confidence Score", f"{confidence:.2f}%")
                
                # Viz
                st.subheader("Prototype Similarity")
                chart_data = {cls: float(p) for cls, p in zip(selected_classes, probs)}
                st.bar_chart(chart_data)
                
                # Explainability
                cls_id_pred = [k for k, v in CLASSES.items() if v == pred_cls][0]
                st.info(f"**Logic:** The patient scan features align closest to the fused prototype of {pred_cls}, which combines visual texture with the clinical definition: *'{DESCRIPTIONS[str(cls_id_pred)]}'*")
                
        elif run_btn and not support_data_ready:
            st.error("Please complete the Support Set uploads first.")