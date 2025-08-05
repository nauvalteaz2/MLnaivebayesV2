# balancing.py
import os
import random
import shutil

random.seed(42)

INPUT_PATH = '../datasetCrop/train'
OUTPUT_PATH = 'datasetCrop_balanced/train'

# Hitung jumlah gambar per class
class_counts = {}
for class_id in os.listdir(INPUT_PATH):
    class_folder = os.path.join(INPUT_PATH, class_id)
    class_counts[class_id] = len(os.listdir(class_folder))

print("Distribusi awal:")
for k, v in sorted(class_counts.items()):
    print(f"Class {k}: {v} gambar")

# Target balancing (undersample ke kelas terkecil)
target_per_class = min(class_counts.values())
print(f"\n🔄 Menyeimbangkan semua kelas ke {target_per_class} gambar per kelas.")

# Buat ulang folder output
if os.path.exists(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)
os.makedirs(OUTPUT_PATH, exist_ok=True)

for class_id in os.listdir(INPUT_PATH):
    src_folder = os.path.join(INPUT_PATH, class_id)
    dst_folder = os.path.join(OUTPUT_PATH, class_id)
    os.makedirs(dst_folder, exist_ok=True)

    all_images = os.listdir(src_folder)
    if len(all_images) > target_per_class:
        selected = random.sample(all_images, target_per_class)
    else:
        selected = all_images.copy()
        while len(selected) < target_per_class:
            selected += random.sample(all_images, min(target_per_class - len(selected), len(all_images)))

    for fname in selected:
        shutil.copy(os.path.join(src_folder, fname), os.path.join(dst_folder, fname))

print(f"\n✅ Balancing selesai. Dataset tersimpan di: {OUTPUT_PATH}")
