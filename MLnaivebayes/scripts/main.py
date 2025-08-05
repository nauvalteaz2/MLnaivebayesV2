import os
import cv2
import shutil
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from PIL import Image
from collections import Counter
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import joblib
import random

# === PATH ===
DATASET_PATH = '../homeObject'
OUTPUT_PATH = '../datasetCrop'
FILTERED_PATH = '../datasetFiltered/train'

# ============================================================== #
# === STEP 1: EKSTRAKSI OBJEK (YOLO Label ke Crop) ============ #
# ============================================================== #
def extract_objects():
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    def process_split(split):
        image_dir = os.path.join(DATASET_PATH, split, 'images')
        label_dir = os.path.join(DATASET_PATH, split, 'labels')
        for fname in tqdm(os.listdir(image_dir)):
            if not fname.endswith('.jpg'):
                continue
            image_path = os.path.join(image_dir, fname)
            label_path = os.path.join(label_dir, fname.replace('.jpg', '.txt'))
            img = cv2.imread(image_path)
            if img is None:
                continue
            h, w, _ = img.shape
            if not os.path.exists(label_path):
                continue

            with open(label_path, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id, x, y, bw, bh = map(float, parts)
                    class_id = int(class_id)
                    x1 = int((x - bw/2) * w)
                    y1 = int((y - bh/2) * h)
                    x2 = int((x + bw/2) * w)
                    y2 = int((y + bh/2) * h)
                    crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    class_folder = os.path.join(OUTPUT_PATH, split, str(class_id))
                    os.makedirs(class_folder, exist_ok=True)
                    crop_filename = f"{fname.replace('.jpg', '')}_{i}.jpg"
                    cv2.imwrite(os.path.join(class_folder, crop_filename), crop)

    for split in ['train', 'valid']:
        process_split(split)
    print("✅ Ekstraksi objek selesai.")

# ============================================================== #
# === STEP 2A: EDA ============================================ #
# ============================================================== #
def run_eda(base_path, title="EDA"):
    print(f"📊 Menjalankan EDA ({title})...")
    class_counts = {cls: len(os.listdir(os.path.join(base_path, cls))) for cls in os.listdir(base_path)}

    # Distribusi jumlah gambar per kelas
    plt.figure(figsize=(10, 5))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title(f"Distribusi Citra per Class ({title})")
    plt.xlabel("Class ID")
    plt.ylabel("Jumlah Citra")
    plt.show()

    # Contoh gambar dari 10 kelas pertama
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    axs = axs.flatten()
    for idx, class_id in enumerate(sorted(os.listdir(base_path))[:10]):
        img_path = glob(f"{base_path}/{class_id}/*.jpg")[0]
        img = Image.open(img_path)
        axs[idx].imshow(img)
        axs[idx].axis("off")
        axs[idx].set_title(f"Class {class_id}")
    plt.show()

def check_image_sizes(base_path):
    sizes = []
    for class_id in os.listdir(base_path):
        for img_name in os.listdir(os.path.join(base_path, class_id)):
            img = cv2.imread(os.path.join(base_path, class_id, img_name))
            if img is None:
                continue
            h, w = img.shape[:2]
            sizes.append((w, h))
    widths, heights = zip(*sizes)
    plt.figure(figsize=(8, 5))
    sns.histplot(widths, color='blue', label='Width')
    sns.histplot(heights, color='red', label='Height')
    plt.legend()
    plt.title(f"Distribusi Ukuran Gambar ({os.path.basename(base_path)})")
    plt.show()

def detect_blank_images(base_path, threshold=5):
    blank_count = 0
    for class_id in os.listdir(base_path):
        for img_name in os.listdir(os.path.join(base_path, class_id)):
            img = cv2.imread(os.path.join(base_path, class_id, img_name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.mean() < threshold:
                blank_count += 1
    print(f"⚠️ Gambar terlalu gelap/blank di {base_path}: {blank_count}")

# ============================================================== #
# === PCA ===================================================== #
# ============================================================== #
def extract_features_for_pca(folder):
    X, y = [], []
    for class_id in os.listdir(folder):
        for img_name in os.listdir(os.path.join(folder, class_id))[:50]:
            img = cv2.imread(os.path.join(folder, class_id, img_name))
            if img is None:
                continue
            img = cv2.resize(img, (64, 64))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            feat = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
            X.append(feat)
            y.append(int(class_id))
    return np.array(X), np.array(y)

def visualize_pca_features(folder):
    X, y = extract_features_for_pca(folder)
    if len(X) == 0:
        print(f"⚠️ Tidak ada data untuk PCA di {folder}")
        return
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Class")
    plt.title(f"PCA Visualisasi Fitur HOG ({os.path.basename(folder)})")
    plt.show()

# ============================================================== #
# === STEP 2B: FILTER KELAS MAYORITAS ========================= #
# ============================================================== #
def filter_majority_classes(input_train_path=os.path.join(OUTPUT_PATH, 'train'),
                            output_filtered_path=FILTERED_PATH,
                            min_samples=1500):
    print(f"🔍 Memfilter kelas dengan jumlah sampel > {min_samples}...")
    if os.path.exists(output_filtered_path):
        shutil.rmtree(output_filtered_path)
    os.makedirs(output_filtered_path, exist_ok=True)

    for cls in os.listdir(input_train_path):
        src = os.path.join(input_train_path, cls)
        if not os.path.isdir(src):
            continue
        imgs = [f for f in os.listdir(src) if f.endswith(('.jpg', '.png'))]

        if len(imgs) > min_samples:
            dst = os.path.join(output_filtered_path, cls)
            os.makedirs(dst, exist_ok=True)
            for img_name in imgs:
                shutil.copy(os.path.join(src, img_name), os.path.join(dst, img_name))
        else:
            print(f"⚠️ Class {cls} ({len(imgs)} sampel) dihapus karena < {min_samples}")

    print("✅ Dataset difilter. Hanya kelas mayoritas yang digunakan.")

# ============================================================== #
# === STEP 3: TRAINING MODEL ================================== #
# ============================================================== #
def extract_features(folder):
    X, y = [], []
    for class_id in os.listdir(folder):
        for fname in os.listdir(os.path.join(folder, class_id)):
            img = cv2.imread(os.path.join(folder, class_id, fname))
            if img is None:
                continue
            img = cv2.resize(img, (64, 64))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            feature = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
            X.append(feature)
            y.append(int(class_id))
    return np.array(X), np.array(y)

def filter_validation_classes(valid_path, allowed_classes):
    for cls in os.listdir(valid_path):
        if cls not in allowed_classes:
            shutil.rmtree(os.path.join(valid_path, cls), ignore_errors=True)
    print("✅ Validasi difilter hanya untuk kelas mayoritas.")

def train_model():
    print("🧠 Training Naive Bayes...")

    allowed_classes = os.listdir(FILTERED_PATH)

    # Filter validasi agar hanya mengandung kelas mayoritas
    valid_path_filtered = os.path.join(OUTPUT_PATH, 'valid_filtered')
    if os.path.exists(valid_path_filtered):
        shutil.rmtree(valid_path_filtered)
    shutil.copytree(os.path.join(OUTPUT_PATH, 'valid'), valid_path_filtered)
    filter_validation_classes(valid_path_filtered, allowed_classes)

    # PCA untuk train dan validasi filtered
    print("📊 PCA untuk Dataset Filtered (Train):")
    visualize_pca_features(FILTERED_PATH)

    print("📊 PCA untuk Validasi Filtered:")
    visualize_pca_features(valid_path_filtered)

    # Ekstraksi fitur dan training
    X_train, y_train = extract_features(FILTERED_PATH)
    X_val, y_val = extract_features(valid_path_filtered)

    model = GaussianNB().fit(X_train, y_train)
    y_pred = model.predict(X_val)

    print("📋 Classification Report (Filtered Classes):")
    print(classification_report(y_val, y_pred))
    joblib.dump(model, "naivebayes_majority.pkl")
    print("✅ Model disimpan (naivebayes_majority.pkl).")

# ============================================================== #
# === MAIN PIPELINE =========================================== #
# ============================================================== #
if __name__ == "__main__":
    extract_objects()
    run_eda(os.path.join(OUTPUT_PATH, 'train'), title="SEBELUM FILTER")
    check_image_sizes(os.path.join(OUTPUT_PATH, 'train'))
    detect_blank_images(os.path.join(OUTPUT_PATH, 'train'))
    visualize_pca_features(os.path.join(OUTPUT_PATH, 'train'))

    filter_majority_classes()

    # 🔥 EDA setelah filtering
    run_eda(FILTERED_PATH, title="SETELAH FILTER MAYORITAS")
    check_image_sizes(FILTERED_PATH)
    detect_blank_images(FILTERED_PATH)
    visualize_pca_features(FILTERED_PATH)

    train_model()
