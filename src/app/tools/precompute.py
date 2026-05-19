import sys
import os
import json
import torch
import numpy as np
from PIL import Image
import medmnist
from medmnist import INFO

# Setup Paths to project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(project_root)

from src.models.raresight_net import RareSight

# --- CONFIGURATION ---
K_SHOT = 5  # 5 reference images per class
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = os.path.join(project_root, 'checkpoints', 'raresight_best.pth')
DESC_PATH = os.path.join(project_root, 'src','app', 'class_descriptions.json')
ASSETS_DIR = os.path.join(project_root, 'src', 'app', 'assets')
REF_IMG_DIR = os.path.join(ASSETS_DIR, 'reference_images')

CLASSES = {
    0: 'Actinic keratoses', 1: 'Basal cell carcinoma', 2: 'Benign keratosis',
    3: 'Dermatofibroma', 4: 'Melanoma', 5: 'Melanocytic nevi', 6: 'Vascular lesions'
}

def setup_directories():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    os.makedirs(REF_IMG_DIR, exist_ok=True)

def main():
    print("Starting RareSight Precomputation for GP MVP...")
    setup_directories()

    # 1. Load Descriptions
    with open(DESC_PATH, 'r') as f:
        descriptions = json.load(f)

    # 2. Load Trained Model
    model = RareSight(device=DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print("Loaded trained RareSight weights.")
    else:
        print("ERROR: weights not found. Run training first.")
        sys.exit(1)
    model.eval()

    # 3. Load DermaMNIST Dataset
    info = INFO['dermamnist']
    DataClass = getattr(medmnist, info['python_class'])
    dataset = DataClass(split='train', download=True)
    
    prototypes_dict = {}
    metadata_dict = {}

    print("Generating Multi-Modal Prototypes...")
    
    with torch.no_grad():
        for cls_id, cls_name in CLASSES.items():
            print(f"Processing Class {cls_id}: {cls_name}")
            
            # Find indices for this class
            indices = np.where(dataset.labels == cls_id)[0]
            
            # Select the first K_SHOT (5) images to be our "Gold Standard"
            selected_indices = indices[:K_SHOT]
            
            img_tensors = []
            img_paths = []
            desc_text = descriptions[str(cls_id)]
            
            for i, idx in enumerate(selected_indices):
                img_pil, _ = dataset[idx]
                
                # Save physical image for the UI Explainability section
                save_path = os.path.join(REF_IMG_DIR, f"cls_{cls_id}_ref_{i}.jpg")
                img_pil.save(save_path)
                
                # Store the relative path for the web app to use later
                img_paths.append(f"assets/reference_images/cls_{cls_id}_ref_{i}.jpg")
                
                # Preprocess for model
                img_tensors.append(model.preprocess(img_pil).unsqueeze(0))
                
            # Stack and move to device
            s_images = torch.cat(img_tensors).to(DEVICE)
            s_texts = [desc_text] * K_SHOT # Repeat description for all 5 images
            
            # Calculate Fused Embeddings using your novel architecture!
            fused_embeddings = model.encode_multimodal(s_images, s_texts)
            
            # Create Prototype (Mean of the 5 embeddings)
            prototype = fused_embeddings.mean(dim=0)
            
            # Re-normalize (Crucial for distance calculations)
            prototype = prototype / prototype.norm(dim=-1, keepdim=True)
            
            # Store in dictionaries
            prototypes_dict[cls_id] = prototype.cpu() # Move to CPU for saving
            
            metadata_dict[str(cls_id)] = {
                "name": cls_name,
                "description": desc_text,
                "reference_images": img_paths
            }

    # 4. Save to Disk
    pt_path = os.path.join(ASSETS_DIR, 'disease_prototypes.pt')
    json_path = os.path.join(ASSETS_DIR, 'disease_metadata.json')
    
    torch.save(prototypes_dict, pt_path)
    with open(json_path, 'w') as f:
        json.dump(metadata_dict, f, indent=4)

    print("\n✅ Precomputation Complete!")
    print(f"💾 Prototypes saved to: {pt_path}")
    print(f"💾 Metadata saved to: {json_path}")

if __name__ == "__main__":
    main()