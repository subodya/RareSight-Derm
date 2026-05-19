import streamlit as st
import torch
import sys
import os
import json
import numpy as np
from PIL import Image
import cv2

# Setup Paths
APP_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '../../'))
sys.path.append(PROJECT_ROOT)

from src.models.raresight_net import RareSight

# ============================================================================
# ATTENTION ROLLOUT FOR VISION TRANSFORMERS
# ============================================================================

class AttentionRollout:
    """
    Proper explainability for Vision Transformers (BiomedCLIP)
    Tracks attention flow from [CLS] token to image patches
    """
    
    def __init__(self, model, head_fusion='mean', discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.grid_size = 14  # ViT-B/16 with 224x224 input = 14x14 patches
        
    def get_attention_maps(self, image_tensor):
        """Extract attention weights from all transformer blocks"""
        attention_maps = []
        
        with torch.no_grad():
            # Get the vision encoder
            visual = self.model.backbone.visual
            
            # Patch embedding + positional encoding
            x = visual.trunk.patch_embed(image_tensor)
            x = visual.trunk._pos_embed(x)
            
            # Extract attention from each transformer block
            for block in visual.trunk.blocks:
                B, N, C = x.shape
                
                # Compute Q, K, V
                qkv = block.attn.qkv(x)
                qkv = qkv.reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
                q, k, v = qkv[0], qkv[1], qkv[2]
                
                # Attention scores
                attn = (q @ k.transpose(-2, -1)) * block.attn.scale
                attn = attn.softmax(dim=-1)  # (B, heads, N, N)
                
                attention_maps.append(attn[0].cpu())  # Store first batch
                
                # Continue forward pass
                x = block(x)
        
        return attention_maps
    
    def rollout(self, attention_maps):
        """Compute attention rollout across layers"""
        # Fuse multi-head attention
        fused_attentions = []
        for attn in attention_maps:
            if self.head_fusion == 'mean':
                fused = attn.mean(dim=0)
            elif self.head_fusion == 'max':
                fused = attn.max(dim=0)[0]
            else:
                fused = attn.mean(dim=0)
            fused_attentions.append(fused)
        
        # Add identity (residual connections)
        num_tokens = fused_attentions[0].shape[0]
        identity = torch.eye(num_tokens)
        rollout_matrix = identity
        
        for attn in fused_attentions:
            # Add residual
            attn_residual = attn + identity
            # Normalize
            attn_residual = attn_residual / attn_residual.sum(dim=-1, keepdim=True)
            # Accumulate
            rollout_matrix = torch.matmul(attn_residual, rollout_matrix)
        
        # Get [CLS] attention to patches (exclude [CLS] itself)
        cls_attention = rollout_matrix[0, 1:]
        
        return cls_attention
    
    def generate_heatmap(self, image_tensor):
        """Generate attention heatmap"""
        # Get attention maps
        attention_maps = self.get_attention_maps(image_tensor)
        
        # Compute rollout
        cls_attention = self.rollout(attention_maps)
        
        # Reshape to spatial grid
        heatmap = cls_attention.reshape(self.grid_size, self.grid_size).numpy()
        
        # Discard low attention
        if self.discard_ratio > 0:
            flat = heatmap.flatten()
            threshold = np.percentile(flat, self.discard_ratio * 100)
            heatmap[heatmap < threshold] = 0
        
        # Normalize
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap
    
    def overlay_heatmap(self, original_image, heatmap, alpha=0.4):
        """Overlay heatmap on image"""
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        
        h, w = original_image.shape[:2]
        
        # Resize heatmap
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized),
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend
        overlay = cv2.addWeighted(
            original_image.astype(np.uint8),
            1 - alpha,
            heatmap_colored,
            alpha,
            0
        )
        
        return overlay


# ============================================================================
# STREAMLIT APP
# ============================================================================

# Page Config
st.set_page_config(page_title="RareSight Clinical Support", layout="wide", page_icon="🏥")

# --- RESOURCE LOADING ---
@st.cache_resource
def load_resources():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model
    model = RareSight(device=device)
    weight_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'raresight_finetuned.pth')

    if os.path.exists(weight_path):
        try:
            model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
            print(f"✓ Loaded weights from {weight_path}")
        except Exception as e:
            st.error(f"⚠️ Failed to load weights: {e}")
            st.stop()
    else:
        st.error("⚠️ Model weights not found! Run training script first.")
        st.stop()
    
    model.eval()

    # 2. Load Precomputed Assets
    pt_path = os.path.join(APP_DIR, 'assets', 'disease_prototypes.pt')
    json_path = os.path.join(APP_DIR, 'assets', 'disease_metadata.json')
    
    if os.path.exists(pt_path) and os.path.exists(json_path):
        precomputed_prototypes = torch.load(pt_path, map_location=device)
        with open(json_path, 'r') as f:
            metadata = json.load(f)
    else:
        st.error("⚠️ Precomputed assets missing! Run `python precompute.py` first.")
        st.stop()
        
    return model, device, precomputed_prototypes, metadata

try:
    model, device, BASE_PROTOTYPES, METADATA = load_resources()
    CLASSES = {int(k): v['name'] for k, v in METADATA.items()}
    
    # Initialize Attention Rollout
    attention_rollout = AttentionRollout(model, head_fusion='mean', discard_ratio=0.9)
    
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()


# --- UI LAYOUT ---
st.title("🏥 RareSight: Clinical Decision Support")
st.markdown("Dermatology Assistant for General Practitioners | *Vision-Language AI*")
st.divider()

col1, col2 = st.columns([1.2, 1])

# ==========================================
# ZONE A: PATIENT INTAKE (Left Column)
# ==========================================
with col1:
    st.subheader("1. Patient Intake")
    
    # Standard GP Input
    patient_file = st.file_uploader("Upload Patient Skin Scan", type=['jpg', 'png', 'jpeg'])
    clinical_notes = st.text_area(
        "Clinical Notes (Optional)", 
        placeholder="e.g. 45yo male, itchy red patch for 3 weeks, slight bleeding..."
    )
    
    if patient_file:
        patient_img = Image.open(patient_file).convert('RGB')
        st.image(patient_img, caption="Current Patient", width=250)

    # --- MODE 2: REFINEMENT (Hidden Power Feature) ---
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🛠️ Advanced: Refine Diagnosis with Reference Cases (Few-Shot Mode)"):
        st.info("Not satisfied with default knowledge? Upload 1-3 textbook examples to dynamically teach the system.")
        
        refine_classes = st.multiselect("Select condition(s) to refine:", list(CLASSES.values()))
        
        user_prototypes = {}  # Dynamic overrides
        
        for cls_name in refine_classes:
            cls_id = [k for k, v in CLASSES.items() if v == cls_name][0]
            st.markdown(f"**Upload reference images for {cls_name}**")
            
            upl_files = st.file_uploader(
                f"References for {cls_name}", 
                accept_multiple_files=True, 
                key=f"ref_{cls_id}"
            )
            
            if upl_files:
                # Limit to 5 images
                if len(upl_files) > 5:
                    st.warning(f"⚠️ Maximum 5 images per class. Using first 5.")
                    upl_files = upl_files[:5]
                
                img_tensors = []
                for f in upl_files:
                    try:
                        img = Image.open(f).convert('RGB')
                        # Validate size
                        if img.size[0] < 28 or img.size[1] < 28:
                            st.error(f"Image too small: {f.name}")
                            continue
                        img_tensors.append(model.preprocess(img).unsqueeze(0))
                    except Exception as e:
                        st.error(f"Failed to load {f.name}: {e}")
                
                if img_tensors:
                    # Calculate dynamic prototype
                    s_tensor = torch.cat(img_tensors).to(device)
                    s_texts = [METADATA[str(cls_id)]['description']] * len(img_tensors)
                    
                    with st.spinner(f'Learning new prototype for {cls_name}...'):
                        with torch.no_grad():
                            fused_emb = model.encode_multimodal(s_tensor, s_texts)
                            new_proto = fused_emb.mean(dim=0)
                            new_proto = new_proto / new_proto.norm(dim=-1, keepdim=True)
                            user_prototypes[cls_id] = new_proto
                    
                    st.success(f"✓ Learned {cls_name} from {len(img_tensors)} examples")

# ==========================================
# ZONE B: DIAGNOSIS & EXPLAINABILITY (Right)
# ==========================================
with col2:
    st.subheader("2. AI Analysis & Results")
    
    if not patient_file:
        st.info("👈 Upload a patient scan to begin analysis.")
    else:
        run_btn = st.button("Run Clinical Analysis", type="primary", use_container_width=True)
        
        if run_btn:
            with st.spinner("Analyzing patient scan against knowledge base..."):
                try:
                    # 1. Prepare Query
                    q_tensor = model.preprocess(patient_img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        q_emb = model.backbone.encode_image(q_tensor)
                        q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)
                    
                    # 2. Prepare Prototypes (Merge Default + User Overrides)
                    final_prototypes = []
                    class_order = []
                    
                    for cls_id in sorted(CLASSES.keys()):
                        class_order.append(cls_id)
                        if cls_id in user_prototypes:
                            final_prototypes.append(user_prototypes[cls_id])  # Dynamic
                        else:
                            final_prototypes.append(BASE_PROTOTYPES[cls_id].to(device))  # Precomputed
                    
                    proto_tensor = torch.stack(final_prototypes)  # [7, 512]
                    
                    # 3. Calculate Distances and Probabilities
                    dists = torch.cdist(q_emb, proto_tensor)
                    probs = torch.nn.functional.softmax(-dists, dim=1).cpu().numpy()[0]
                    
                    # 4. Get Top 3
                    top3_indices = np.argsort(probs)[::-1][:3]
                    
                    st.divider()
                    st.markdown("### Top Predicted Conditions")
                    
                    # --- CLINICAL ALERTS & UNCERTAINTY HANDLING ---
                    top_prob = probs[top3_indices[0]]
                    top_class_id = class_order[top3_indices[0]]
                    top_class_name = CLASSES[top_class_id]
                    
                    if top_prob < 0.40:
                        st.warning("⚠️ **Low Confidence Match:** This case does not strongly match known conditions. Clinical correlation required.")
                    
                    if top_class_name == "Melanoma" and top_prob > 0.30:
                        st.error("🚨 **HIGH RISK:** Features suggest Melanoma. Immediate dermatologist referral recommended.")
                    
                    # Display Progress Bars for Top 3
                    for idx in top3_indices:
                        c_id = class_order[idx]
                        c_name = CLASSES[c_id]
                        c_prob = probs[idx]
                        
                        st.markdown(f"**{c_name}** - {c_prob*100:.1f}%")
                        st.progress(float(c_prob))
                    
                    # --- ATTENTION ROLLOUT EXPLAINABILITY ---
                    st.divider()
                    st.markdown("### AI Attention Rollout  (XAI)")
                    st.caption(f"Visualizing the image regions that contributed to the diagnosis of **{top_class_name}**.")
                    
                    with st.spinner("Generating attention heatmap..."):
                        try:
                            # Generate heatmap using Attention Rollout
                            heatmap = attention_rollout.generate_heatmap(q_tensor)
                            
                            # Create overlay
                            overlay = attention_rollout.overlay_heatmap(patient_img, heatmap, alpha=0.4)
                            
                            # Display side-by-side
                            hm_col1, hm_col2 = st.columns(2)
                            with hm_col1:
                                st.image(patient_img, caption="Original Patient Scan", width=300)
                            with hm_col2:
                                st.image(overlay, caption="Attention Heatmap (Red = High Importance)", width=300)
                            
                            st.success("✓ Heatmap shows where the Vision Transformer attended most")
                            
                        except Exception as e:
                            st.error(f"Failed to generate attention map: {e}")
                            st.info("The model can still provide diagnosis without the visualization.")
                    
                    # --- EXPLAINABILITY: SIMILAR REFERENCE CASES ---
                    st.divider()
                    st.markdown(f"### Reference Cases: {top_class_name}")
                    st.caption(f"The patient's scan shares a {top_prob*100:.1f}% semantic match with these textbook prototypes.")
                    
                    ref_images = METADATA[str(top_class_id)]['reference_images'][:3]
                    
                    img_cols = st.columns(3)
                    for i, img_path in enumerate(ref_images):
                        with img_cols[i]:
                            try:
                                full_img_path = os.path.normpath(os.path.join(APP_DIR, img_path))
                                if os.path.exists(full_img_path):
                                    ref_img = Image.open(full_img_path)
                                    st.image(ref_img, caption=f"Reference {i+1}", width=300)
                                else:
                                    st.info(f"Reference {i+1} not available")
                            except Exception as e:
                                st.warning(f"Could not load reference: {e}")
                
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.info("Please check your model weights and precomputed prototypes.")

st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")
st.sidebar.success(f"✓ Model: RareSight (BiomedCLIP)")
st.sidebar.success(f"✓ Device: {device.upper()}")
st.sidebar.success(f"✓ Prototypes: {len(CLASSES)} classes loaded")

if user_prototypes:
    st.sidebar.info(f"🛠️ Refinement active: {len(user_prototypes)} custom prototypes")

st.sidebar.markdown("---")
st.sidebar.caption("⚠️ **Medical Disclaimer:** This is a Clinical Decision Support tool. Final diagnosis must be made by a certified dermatologist.")