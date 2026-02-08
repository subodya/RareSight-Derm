import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import medmnist
from medmnist import INFO
from tqdm import tqdm
import numpy as np

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.raresight_net import RareSight

# --- CONFIG ---
N_WAY = 5
K_SHOT = 5
N_QUERY = 10 
EPISODES = 1000  # 1000 is enough for valid results
LR = 1e-4

# Load Descriptions
desc_path = 'src/app/descriptions.json'
if not os.path.exists(desc_path):
    # Fallback if file not found (prevents crash)
    print("Warning: descriptions.json not found, using placeholders.")
    DESCRIPTIONS = {str(i): "Skin lesion" for i in range(7)}
else:
    with open(desc_path, 'r') as f:
        DESCRIPTIONS = json.load(f)

def get_data():
    """
    Loads MedMNIST. 
    Note: Standard MedMNIST images are 28x28. 
    The model.preprocess will automatically resize them to 224x224.
    """
    info = INFO['dermamnist']
    DataClass = getattr(medmnist, info['python_class'])
    # Removed size=224 as per your request
    return DataClass(split='train', download=True)

def get_episode(dataset, model_preprocess, device):
    classes = np.unique(dataset.labels)
    selected = np.random.choice(classes, N_WAY, replace=False)
    
    s_imgs, s_txts, q_imgs, q_lbls = [], [], [], []
    
    for i, cls in enumerate(selected):
        indices = np.where(dataset.labels == cls)[0]
        
        # Ensure we don't crash if a class has few samples
        needed = K_SHOT + N_QUERY
        replace = len(indices) < needed
        chosen = np.random.choice(indices, needed, replace=replace)
        
        # Get Description
        desc = DESCRIPTIONS.get(str(cls), "Dermatology image")
        
        # Support Set
        for idx in chosen[:K_SHOT]:
            img, _ = dataset[idx] # Returns PIL Image (likely 28x28)
            # preprocess resizes to 224x224
            s_imgs.append(model_preprocess(img).unsqueeze(0))
            s_txts.append(desc)
            
        # Query Set
        for idx in chosen[K_SHOT:]:
            img, _ = dataset[idx]
            q_imgs.append(model_preprocess(img).unsqueeze(0))
            q_lbls.append(i)

    return (torch.cat(s_imgs).to(device), s_txts, 
            torch.cat(q_imgs).to(device), 
            torch.tensor(q_lbls).long().to(device))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting RareSight Training on {device}...")
    
    # 1. Init Model
    model = RareSight(device=device)
    model.train()
    
    # 2. Optimizer
    # We optimize the fusion network and the alpha gate
    optimizer = optim.Adam(list(model.fusion_net.parameters()) + [model.alpha], lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Data
    dataset = get_data()
    
    # 4. Loop
    os.makedirs("checkpoints", exist_ok=True)
    best_loss = float('inf')
    
    pbar = tqdm(range(EPISODES))
    for ep in pbar:
        optimizer.zero_grad()
        
        # Get Batch
        s_img, s_txt, q_img, q_lbl = get_episode(dataset, model.preprocess, device)
        
        # Forward
        logits = model(s_img, s_txt, q_img, N_WAY, K_SHOT)
        loss = criterion(logits, q_lbl)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Logs
        pbar.set_description(f"Loss: {loss.item():.4f} | Alpha: {model.alpha.item():.3f}")
        
        # Save Best
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), "checkpoints/raresight_best.pth")

    print("\n✅ Training Complete.")
    print(f"Best Loss: {best_loss:.4f}")
    print("Model saved to checkpoints/raresight_best.pth")

if __name__ == "__main__":
    main()