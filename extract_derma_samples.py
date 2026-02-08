import os
import medmnist
from medmnist import INFO

# Config
dataset_name = "dermamnist"
num_samples_per_class = 5
output_root = "derma_samples"

os.makedirs(output_root, exist_ok=True)

info = INFO[dataset_name]
DataClass = getattr(medmnist, info['python_class'])
train_dataset = DataClass(split='train', download=True)

print(f"Loaded DermaMNIST — total train samples: {len(train_dataset)}")

num_classes = len(info['label'])
class_counts = {i: 0 for i in range(num_classes)}

for img, label in train_dataset:
    label = int(label)

    if class_counts[label] >= num_samples_per_class:
        continue

    class_name = info['label'][str(label)]
    class_folder = os.path.join(output_root, f"{label}_{class_name}")
    os.makedirs(class_folder, exist_ok=True)

    # img is already a PIL Image
    img_pil = img.convert("RGB")
    save_path = os.path.join(
        class_folder, f"{label}_{class_counts[label]}.png"
    )
    img_pil.save(save_path)

    class_counts[label] += 1
    print(f"Saved: {save_path}")

    if all(v >= num_samples_per_class for v in class_counts.values()):
        break

print("\nExtraction complete.")
