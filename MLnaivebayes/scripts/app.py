import streamlit as st
import joblib
import numpy as np
import cv2
from PIL import Image
from skimage.feature import hog

# === Load model dan class_map ===
model = joblib.load("naivebayes_majority.pkl")
class_map = joblib.load("class_map.pkl")  # dict: {0: 'bed', ...}

# === Judul App ===
st.set_page_config(page_title="Klasifikasi Objek Rumah", layout="centered")
st.title("🔍 Klasifikasi Objek Rumah Tangga")
st.write("Upload gambar crop objek (misalnya hasil potongan dari ruang tamu) untuk diklasifikasikan.")

# === Upload Gambar ===
uploaded_file = st.file_uploader("Pilih Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_resized = image.resize((64, 64))
    img_np = np.array(img_resized)

    # Konversi ke grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Ekstrak fitur HOG
    feature = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)

    # Prediksi
    prediction = model.predict([feature])[0]
    label = class_map.get(prediction, "Tidak diketahui")

    # Tampilkan
    st.image(image, caption="Gambar yang Diupload", use_container_width=True)
    st.success(f"Hasil Klasifikasi: **{label.upper()}**")
