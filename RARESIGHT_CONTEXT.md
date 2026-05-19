# 🔬 RareSight Project Context Document

**For: AI Assistants (Claude, ChatGPT, etc.)**  
**Purpose: Complete understanding of codebase, research objectives, and current status**  
**Last Updated: 2025**

---

## 📋 **Project Overview**

### **Official Title**
**RareSight: Vision-Language Few-Shot Meta-Learning for Diagnostic Classification of Rare Pediatric Diseases in Low-Resource Settings**

### **Student Information**
- **Name**: Subodhya Alahakoon (CB012855)
- **Institution**: University of Staffordshire / APIIT
- **Supervisor**: Dr. Rasika Rajapaksha
- **Project Type**: Final Year Project (FYP)
- **Submission Date**: TBD (Interim completed, Final submission pending)

### **Current Status** (As of conversation)
- ✅ **Proposal**: Submitted (5% weight)
- ✅ **Interim Report**: Submitted (15% weight)
- 🔄 **Implementation**: In progress (~70% complete)
- 🔄 **Model Training**: Retraining in progress (accuracy currently low)
- ⏳ **Final Submission**: Pending (40% weight)
- ⏳ **Demo/Viva**: Not yet scheduled

---

## 🎯 **Research Problem & Motivation**

### **The Clinical Problem**
- **300 million people** globally affected by rare diseases
- **50% are pediatric cases** (children)
- **Average diagnostic delay**: 5-7 years from symptoms to diagnosis
- **40% misdiagnosis rate** during diagnostic odyssey
- **1:30,000** patient-to-specialist ratio (vs 1:400 for general medicine)
- **80% of rare diseases** have fewer than 50 documented clinical images

### **The AI Challenge**
Standard deep learning requires thousands of labeled images. Rare diseases have:
- **Ultra-low data availability**: <10 images per disease class
- **Extreme class imbalance**: Common diseases = 10K images, rare = <50
- **Domain shift**: Medical images differ from ImageNet pre-training
- **Clinical heterogeneity**: Similar symptoms across different diseases

### **Why Existing Solutions Fail**
1. **Transfer Learning (ResNet-50)**: Requires hundreds of images for fine-tuning
2. **Zero-Shot CLIP**: Generic, not medical domain-specific
3. **Few-Shot Meta-Learning**: Uses image-only, ignores clinical text
4. **Medical VLMs**: Not integrated with episodic meta-learning

---

## 🧠 **RareSight's Core Innovation**

### **Research Gap Addressed**
RareSight is the **first system** to integrate:
1. **Medical Vision-Language Models** (BiomedCLIP) 
2. **Episodic Meta-Learning** (Prototypical Networks)
3. **Multi-modal Fusion** (Image + Clinical Text)

### **Technical Contribution**
```
Standard Approach: Image → CNN → Classification (needs 1000+ images)
RareSight Approach: (Image + Text) → BiomedCLIP + Fusion → Prototypes → Few-shot Classification (needs 3-10 images)
```

### **Expected Performance Targets**
- **75-80% accuracy** on 5-shot rare disease classification
- Within **5-10%** of fully-supervised models (trained on full dataset)
- **ECE < 0.10** (calibration error for clinical confidence)
- **Grad-CAM IoU > 70%** (explainability alignment with lesion regions)

---

## 🏗️ **System Architecture**

### **High-Level Components**

```
┌─────────────────────────────────────────────────────────────┐
│                    RareSight System                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐      ┌──────────────────┐                │
│  │ Query Image │──┐   │ Support Set      │                │
│  │   (28x28)   │  │   │ (N×K images)     │                │
│  └─────────────┘  │   └──────────────────┘                │
│                   │                                         │
│                   ▼                                         │
│         ┌────────────────────┐                             │
│         │   BiomedCLIP       │ (Frozen, 196M params)       │
│         │   ViT-B/16 Vision  │                             │
│         │   + PubMedBERT     │                             │
│         └────────────────────┘                             │
│                   │                                         │
│                   ▼                                         │
│         ┌────────────────────┐                             │
│         │  Fusion Network    │ (Trainable, 2.9M params)    │
│         │  - Multi-scale     │                             │
│         │  - Cross-attention │                             │
│         │  - Alpha gating    │                             │
│         └────────────────────┘                             │
│                   │                                         │
│                   ▼                                         │
│         ┌────────────────────┐                             │
│         │ Prototypical       │                             │
│         │ Meta-Learning      │                             │
│         │ - Compute prototypes│                            │
│         │ - Distance metric  │                             │
│         │ - Softmax classify │                             │
│         └────────────────────┘                             │
│                   │                                         │
│                   ▼                                         │
│         ┌────────────────────┐                             │
│         │   Explainability   │                             │
│         │   - Attention maps │                             │
│         │   - Grad-CAM/Rollout│                            │
│         │   - Uncertainty    │                             │
│         └────────────────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

### **Detailed Architecture (From Proposal)**

**Component 1: Multi-Scale Vision Encoder**
- Extract features from BiomedCLIP ViT layers 6 (textures) and 12 (semantics)
- Fuse via learned projection: `v_multi = [v₆; v₁₂]` → `v ∈ ℝ⁵¹²`
- **Trainable params**: 524K

**Component 2: Adaptive Text Encoder**
- Prepend 4 learnable prompt embeddings to disease descriptions
- Conditions text features on visual domain (dermatology vs radiology)
- **Trainable params**: 2K prompts + 1.3M fusion = 1.302M

**Component 3: Cross-Modal Attention Fusion**
- Align visual queries with textual keys/values
- Task-adaptive multi-modal integration
- **Trainable params**: 786K

**Component 4: Weighted Prototype Network**
- Compute importance scores for each support example
- Downweight noisy/outlier images
- **Trainable params**: 263K

**Component 5: Temperature-Calibrated Classification**
- Cosine similarity with learnable temperature scaling
- Improves confidence calibration
- **Trainable params**: 1 temperature parameter

**Total Trainable**: ~2.9M parameters  
**Total Frozen**: 196M (BiomedCLIP backbone)

---

## 📊 **Dataset & Evaluation**

### **Primary Dataset: DermaMNIST**
- **Source**: MedMNIST (Standardized medical imaging benchmark)
- **Size**: 10,015 dermatology images (28×28 pixels)
- **Classes**: 7 skin conditions
  - 0: Actinic keratoses (precancerous)
  - 1: Basal cell carcinoma (malignant)
  - 2: Benign keratosis
  - 3: Dermatofibroma (rare)
  - 4: Melanoma (malignant)
  - 5: Melanocytic nevi (common moles)
  - 6: Vascular lesions (rare)
- **Split**: Train/Val/Test
- **Challenge**: Classes 3 and 6 are genuinely rare (low sample count)

### **Clinical Text Descriptions** (From `class_descriptions.json`)
Each class has detailed clinical text:
```json
{
  "0": "Actinic keratoses: Rough, scaly patches on sun-exposed skin...",
  "4": "Melanoma: ABCDE criteria - Asymmetry, Border irregularity, Color variation..."
}
```

### **Evaluation Protocol**
- **Episodic Testing**: Sample N-way K-shot episodes
  - N-way: Number of classes per episode (e.g., 5)
  - K-shot: Number of support images per class (e.g., 5)
  - N-query: Number of test images per episode (e.g., 10)
- **Metrics**:
  - Accuracy, Precision, Recall, F1-score
  - Per-class performance (critical for rare classes)
  - Calibration: ECE (Expected Calibration Error)
  - Explainability: Grad-CAM IoU with ground truth lesions

### **Baseline Comparisons** (From Proposal)
1. **Transfer Learning**: ResNet-50 fine-tuned on DermaMNIST
2. **Standard ProtoNet**: Prototypical Networks with ResNet-50 backbone
3. **Zero-Shot BiomedCLIP**: CLIP with text prompts, no training
4. **LP+text**: Linear probe on BiomedCLIP features
5. **DTL+ResizeMix**: Data augmentation approach
6. **Supervised Upper Bound**: Fully-supervised on entire dataset

---

## 💻 **Current Implementation Status**

### **What Works ✅**
1. **Model Architecture**: `src/models/raresight_net.py`
   - BiomedCLIP integration
   - Fusion network (multi-modal encoding)
   - Prototypical meta-learning forward pass
   - Alpha-gated residual connections

2. **Training Pipeline**: `src/training/train_raresight.py` (basic), `train_advanced.py` (improved)
   - Episodic sampling (N-way K-shot)
   - Data augmentation (flip, rotate, color jitter)
   - Cosine annealing LR scheduler
   - Checkpoint saving

3. **Web Application**: `src/app/app.py`
   - Streamlit-based clinical interface
   - **Mode 1**: Pre-computed prototypes (instant diagnosis)
   - **Mode 2**: Refinement mode (few-shot dynamic learning)
   - **Explainability**: Attention Rollout (not Grad-CAM)
   - Top-3 predictions with confidence scores
   - Reference case visualization
   - Clinical notes input

4. **Prototype Pre-computation**: `src/tools/precompute.py`
   - Extracts 5-shot prototypes from DermaMNIST train set
   - Saves to `src/app/assets/disease_prototypes.pt`
   - Metadata with reference image paths

5. **Explainability**: Attention Rollout implemented
   - Tracks attention flow from [CLS] token to patches
   - Proper method for Vision Transformers (not Grad-CAM hacks)
   - Generates heatmaps overlaid on patient scans

### **What Needs Work 🔄**

1. **Model Accuracy**: Currently LOW (exact number unknown from context)
   - Retraining with `train_advanced.py` (unfreezing last ViT block)
   - Need to evaluate if this helps or hurts

2. **Missing Baselines**: No comparison results yet
   - Need to run ResNet-50 transfer learning baseline
   - Need to run standard ProtoNet baseline
   - Need zero-shot BiomedCLIP baseline

3. **Evaluation Pipeline**: No systematic evaluation script
   - No test set results
   - No confusion matrix
   - No per-class F1 scores
   - No calibration metrics (ECE)

4. **Clinical Features** (Tier 1 roadmap):
   - ❌ Unknown disease detection (reject low-confidence predictions)
   - ❌ Confidence calibration (temperature scaling, confidence labels)
   - ❌ Prototype comparison UI (similarity visualization)
   - ❌ Structured patient input (age, gender, symptoms)
   - ❌ Clinical report generation (PDF export)

5. **Documentation**:
   - Proposal ✅ (submitted)
   - Interim report ✅ (submitted)
   - Final thesis ⏳ (pending)
   - Code documentation ⚠️ (minimal)

---

## 🔧 **Key Technical Decisions**

### **Why BiomedCLIP (not standard CLIP)?**
- Pre-trained on **15M biomedical image-text pairs** (not generic web images)
- Covers dermatology, pathology, radiology, ophthalmology
- **State-of-the-art** on medical zero-shot benchmarks
- ViT-B/16 vision encoder + PubMedBERT text encoder

### **Why Prototypical Networks (not MAML)?**
- **Simpler**: No second-order gradients
- **Faster**: O(N) complexity vs O(N²) for relation networks
- **Interpretable**: Prototypes = class centroids in embedding space
- **Proven**: Works well for medical few-shot learning

### **Why Freeze BiomedCLIP?**
- **196M parameters**: Too large to fine-tune with small data
- **Risk of overfitting**: DermaMNIST train set is tiny
- **Transfer learning philosophy**: Use pre-trained features, train fusion layer

### **Why Multi-Modal (Image + Text)?**
- Clinical diagnosis uses **both** visual appearance and textual descriptions
- Text descriptions provide **semantic constraints** (e.g., "rough scaly patches")
- Proposal claims **+8-10% accuracy** over image-only

### **Current Training Approach** (`train_advanced.py`)
```python
# Episodic Training Loop
for episode in range(4000):
    # Sample N-way K-shot episode
    classes = random.sample(all_classes, N_WAY)
    support_images, support_texts = sample_k_shot(classes, K_SHOT)
    query_images, query_labels = sample_queries(classes, N_QUERY)
    
    # Forward pass
    logits = model(support_images, support_texts, query_images, N_WAY, K_SHOT)
    
    # Cross-entropy loss
    loss = criterion(logits, query_labels)
    
    # Backprop (only fusion network + alpha parameter)
    loss.backward()
    optimizer.step()
```

**Recent Change**: Unfreezing last ViT transformer block
- **Rationale**: Adapt BiomedCLIP to 28×28 DermaMNIST domain (originally trained on high-res)
- **Risk**: Might overfit or destabilize training
- **Status**: Currently retraining to see if accuracy improves

---


```
RareSight-Derm/
├── configs/
│   └── config.yaml                    # Training hyperparameters
│
├── data/
│   └── text_descriptions/
│       └── clinical_descriptions.yaml # Disease text descriptions
│
├── derma_samples/                     # Reference images (extracted)
│   ├── 0_actinic keratoses/
│   ├── 1_basal cell carcinoma/
│   ├── 2_benign keratosis/
│   ├── 3_dermatofibroma/
│   ├── 4_melanoma/
│   ├── 5_melanocytic nevi/
│   └── 6_vascular lesions/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Dataset analysis
│   ├── 02_baseline1_analysis.ipynb   # Baseline experiments
│   └── evaluation.ipynb               # Model evaluation
│
├── src/
│   ├── app/
│   │   ├── app.py                     # 🔥 Streamlit clinical interface
│   │   ├── class_descriptions.json    # Disease metadata
│   │   ├── assets/
│   │   │   ├── disease_prototypes.pt  # Pre-computed prototypes
│   │   │   ├── disease_metadata.json  # Prototype metadata
│   │   │   └── reference_images/       # Support set images
│   │   └── backend_database/          # Reference images (old structure)
│   │
│   ├── models/
│   │   ├── raresight_net.py           # 🔥 Main RareSight model
│   │   ├── baseline_biomedclip.py     # Zero-shot CLIP baseline
│   │   ├── baseline_protonet.py       # Standard ProtoNet baseline
│   │   └── baseline_transfer.py       # Transfer learning baseline
│   │
│   ├── data/
│   │   ├── dataset.py                 # Episodic sampler
│   │   ├── preprocessing.py           # Image transforms
│   │   └── standard_dataset.py        # Standard supervised dataset
│   │
│   ├── training/
│   │   ├── train_raresight.py         # Basic training script
│   │   ├── train_advanced.py          # 🔥 Advanced training (current)
│   │   ├── train_baseline_protonet.py
│   │   ├── train_baseline_transfer.py
│   │   └── eval_baseline_biomedclip.py
│   │
│   ├── utils/
│   │   └── metrics.py                 # Evaluation metrics
│   │
│   └── tools/
│       └── precompute.py              # Pre-compute prototypes
│
├── checkpoints/
│   └── raresight_best.pth             # Trained model weights
│
├── requirements.txt                    # Python dependencies
├── extract_derma_samples.py           # Extract reference images
└── test_*.py                          # Test scripts
```

---

## 🎯 **Research Questions (From Proposal)**

### **RQ1: Few-Shot Accuracy**
> To what extent does integrating BiomedCLIP with Prototypical Networks improve few-shot rare disease accuracy vs. CNN-based meta-learning and zero-shot VLM baselines across 1/5/10-shot regimes?

**What to prove**: RareSight > Standard ProtoNet (ResNet) and > Zero-shot BiomedCLIP

### **RQ2: Multi-Modal Fusion**
> How do multi-modal prototypes (image-text pairs) compare to image-only prototypes, and which fusion mechanisms optimize K≤5 performance?

**What to prove**: Image+Text > Image-only by ~8-10%

### **RQ3: Clinical Interpretability**
> Does few-shot meta-learning achieve clinical interpretability thresholds (Grad-CAM IoU >70%, entropy-error correlation r<-0.50, ECE <0.10)?

**What to prove**: System is explainable and well-calibrated for clinical use

---


### **Weights**
- **Proposal**: 5% ✅ (Completed)
- **Interim**: 15% ✅ (Completed)
- **Final Submission**: 40% ⏳ (Pending)
- **Viva/Demo**: ~40% ⏳ (Pending)

### **Final Submission Requirements**
**Chapter 5 - Implementation**:
- ✅ Partial implementation (web app works)
- ⏳ Full implementation with architecture details
- ⏳ Technical documentation


---

## 🚧 **Current Challenges**

### **1. Low Model Accuracy** 🔴
- **Issue**: Model performance below target (exact number unknown)
- **Hypothesis**: 28×28 resolution too low for BiomedCLIP (trained on 224×224)
- **Current Fix**: Unfreezing last ViT block in `train_advanced.py`
- **Risk**: Might overfit or not help

### **2. No Baseline Comparisons** 🟡
- **Issue**: Can't claim "RareSight is better" without baselines
- **Need**: Run ResNet-50, Standard ProtoNet, Zero-shot CLIP
- **Time**: ~1 day per baseline

### **3. Evaluation Pipeline Missing** 🟡
- **Issue**: No systematic test set evaluation
- **Need**: Script that runs 600 episodes, computes metrics, saves results
- **Metrics**: Accuracy, F1 (per-class), Confusion Matrix, ECE, Calibration plot

### **4. Grad-CAM → Attention Rollout** 🟢
- **Issue**: Original plan used Grad-CAM (wrong for ViT)
- **Fix**: Implemented Attention Rollout (correct method)
- **Status**: Working in app.py

### **5. Clinical Features Incomplete** 🟡
- **Issue**: App is functional but missing Tier 1 features
- **Needed**: Unknown detection, confidence calibration, structured input, reports
- **Impact**: Affects "practical application" narrative



---

## 🎓 **Defense Strategy**

### **Narrative Options**

**Option A: "Practical Clinical Tool"**
- Focus: GP decision support for rare disease triage
- Demo: End-to-end workflow (upload → diagnosis → report)
- Strength: Real-world impact, usability
- Weakness: No clinical validation, low accuracy undermines practicality

**Option B: "Research Prototype"**
- Focus: Novel integration of VLM + meta-learning
- Demo: Ablation studies showing multi-modal gains
- Strength: Academic rigor, methodological novelty
- Weakness: Low accuracy, incomplete baselines

**Option C: "Hybrid - Methodological + Practical"** ⭐ (Recommended)
- Focus: Few-shot learning enables practical deployment with limited data
- Demo: Both instant diagnosis (precomputed) + dynamic refinement
- Strength: Balances research novelty with practical application
- Weakness: Requires both strong results AND usable app

### **Key Talking Points**

1. **Why Few-Shot Learning Matters**:
   - "80% of rare diseases have <50 images. Standard deep learning requires 1000+. Few-shot meta-learning bridges this gap."

2. **Why Multi-Modal Matters**:
   - "Clinical diagnosis uses both visual appearance and textual descriptions. Image-only models ignore half the diagnostic information."

3. **Why BiomedCLIP Matters**:
   - "Generic CLIP was trained on web images. BiomedCLIP was trained on 15M medical images, giving it domain-specific knowledge."

4. **Why Attention Rollout > Grad-CAM**:
   - "Grad-CAM was designed for CNNs. BiomedCLIP uses Vision Transformers. Attention Rollout is the correct explainability method for transformers, tracking attention flow across layers."

5. **If Accuracy is Low**:
   - "This project demonstrates the challenges of few-shot medical AI. The 28×28 resolution of DermaMNIST limits performance, but the methodology is sound and would scale to higher-resolution clinical data."

---

## 📊 **Expected Questions & Answers**

### **Q: Why is your accuracy low?**
**A**: "DermaMNIST images are only 28×28 pixels, severely limiting visual detail. BiomedCLIP was pre-trained on 224×224 images. Despite this domain mismatch, our multi-modal approach still outperforms image-only baselines by X%. With higher-resolution clinical data, we expect accuracy to reach the 75-80% target."

### **Q: How is this different from standard ProtoNet?**
**A**: "Three key differences: (1) We use BiomedCLIP instead of ResNet-50, leveraging medical domain knowledge from 15M pre-trained images. (2) We integrate clinical text descriptions via multi-modal fusion, not just images. (3) We add weighted prototypes to downweight noisy support examples."

### **Q: Why not just use zero-shot CLIP?**
**A**: "Zero-shot CLIP requires carefully crafted text prompts and has no task-specific learning. Our episodic meta-learning adapts to the specific diseases in the support set, learning optimal fusion weights and prototype representations."

### **Q: How would a doctor use this?**
**A**: "Scenario: A rural GP encounters a suspicious skin lesion. They upload a photo to RareSight. The system instantly compares against pre-computed prototypes of 7 conditions, returning a ranked differential diagnosis with confidence scores and attention heatmaps showing which image regions influenced the prediction. If unsure, the GP can upload 2-3 reference images of a suspected condition, and RareSight re-computes prototypes dynamically. This flags high-risk cases (melanoma) for specialist referral."

### **Q: What's your main contribution?**
**A**: "RareSight is the first system to integrate medical vision-language models with episodic meta-learning. This enables accurate rare disease diagnosis from 3-10 images instead of 1000+, making AI-assisted diagnosis feasible in low-resource settings where data is scarce."

### **Q: Why didn't you validate with real doctors?**
**A**: "Clinical validation requires IRB approval and access to medical professionals, which is beyond the scope of an undergraduate FYP. However, the system is designed with clinical usability in mind: explainability (attention heatmaps), uncertainty quantification (confidence scores), and differential diagnosis (top-3 predictions). Future work includes prospective clinical trials."



## 📚 **Key References (From Proposal)**

1. **Snell et al. (2017)**: Prototypical Networks for Few-shot Learning (Original ProtoNet paper)
2. **Radford et al. (2021)**: CLIP - Learning Transferable Visual Models (Original CLIP)
3. **Zhang et al. (2025)**: BiomedCLIP (Medical vision-language model used in RareSight)
4. **Zhao et al. (2025)**: Survey of CLIP in medical imaging (Literature review source)
5. **Schaefer et al. (2020)**: Machine learning in rare diseases (Problem domain)

---

## 🔗 **Some Important File Paths**

### **Model**
- `src/models/raresight_net.py` - Main model architecture
- `checkpoints/raresight_best.pth` - Trained weights

### **Training**
- `src/training/train_advanced.py` - Current training script (unfreezes last ViT block)
- `src/training/train_raresight.py` - Basic training script

### **App**
- `src/app/app.py` - Streamlit clinical interface
- `src/app/assets/disease_prototypes.pt` - Pre-computed prototypes
- `src/app/class_descriptions.json` - Disease metadata

### **Data**
- DermaMNIST downloaded via `medmnist` package
- `data/text_descriptions/clinical_descriptions.yaml` - Clinical text

