import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

st.set_page_config(layout="wide", page_title="Hybrid Image Retrieval", page_icon="🍎")

@st.cache_resource
def load_cv_model():
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    return model

base_model = load_cv_model()
database_path = 'data/'

def segment_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img, mask

def extract_color_features(img, mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, [180, 256], [0, 180, 0, 256])
    return hist.flatten()

def extract_resnet_features(img_path, model):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(img_array_expanded)
    features = model.predict(preprocessed_img, verbose=0)
    return features.flatten()

@st.cache_data
def build_database_index(db_path, _model):
    color_feats_list = []
    resnet_feats_list = []
    image_paths_list = []
    
    if not os.path.exists(db_path):
        return None, None, []

    for filename in os.listdir(db_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(db_path, filename)
            try:
                img, mask = segment_image(img_path)
                color_vec = extract_color_features(img, mask)
                color_norm = normalize(color_vec.reshape(1, -1), norm='l2').flatten()
                
                resnet_vec = extract_resnet_features(img_path, _model)
                resnet_norm = normalize(resnet_vec.reshape(1, -1), norm='l2').flatten()
                
                color_feats_list.append(color_norm)
                resnet_feats_list.append(resnet_norm)
                image_paths_list.append(img_path)
            except Exception as e:
                print(f"Gagal indexing {filename}: {e}")
                
    return np.array(color_feats_list), np.array(resnet_feats_list), image_paths_list

with st.spinner("Sedang memuat database citra... Mohon tunggu..."):
    db_color, db_resnet, image_paths = build_database_index(database_path, base_model)

    st.title("Sistem Temu Kembali Citra Buah", anchor="false")
    st.write("Proyek UAS Computer Vision - Kombinasi Ekstraksi Fitur Tradisional (HSV Histogram) & Deep Learning (ResNet50)")

    st.sidebar.header("Kontrol Bobot Fitur (Feature Fusion)", anchor=False)
    color_weight = st.sidebar.slider("Fokus Fitur Warna (Tradisional)", 0.0, 1.0, 0.5, 0.1)
    resnet_weight = st.sidebar.slider("Fokus Fitur Bentuk/Tekstur (AI ResNet)", 0.0, 1.0, 0.5, 0.1)
    top_n = st.sidebar.selectbox("Jumlah Hasil Tampilan (Top-N)", [3, 5, 10], index=1)

    st.sidebar.markdown("---")
    st.sidebar.info("Tips: Coba ubah slider ke 1.0 Warna saja atau 1.0 AI saja untuk melihat perbedaan akurasi pencarian!")

    uploaded_file = st.file_uploader("Unggah Gambar Query Kamu dari Device...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        temp_query_path = "temp_query.jpg"
        with open(temp_query_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            st.subheader("Gambar Query Asli", anchor=False)
            st.image(uploaded_file, width=512)
            
        with col_q2:
            st.subheader("Hasil Segmentasi Objek (Mask)", anchor=False)
            _, query_mask = segment_image(temp_query_path)
            st.image(query_mask, caption="Mask Hitam-Putih Otsu Thresholding", width=512)

        # --- TOMBOL AKSI PENCARIAN ---
        if st.button("Jalankan Pencarian Gambar", type="primary"):
            if len(image_paths) == 0:
                st.error(f"Folder '{database_path}' kosong atau tidak ditemukan!")
            else:
                with st.spinner("Sistem sedang menghitung Cosine Similarity..."):
                    q_img, q_mask = segment_image(temp_query_path)
                    q_color = extract_color_features(q_img, q_mask)
                    q_resnet = extract_resnet_features(temp_query_path, base_model)
                    
                    q_color_norm = normalize(q_color.reshape(1, -1), norm='l2').flatten()
                    q_resnet_norm = normalize(q_resnet.reshape(1, -1), norm='l2').flatten()
                    
                    q_hybrid = np.concatenate([q_color_norm * color_weight, q_resnet_norm * resnet_weight]).reshape(1, -1)
                    db_hybrid = np.concatenate([db_color * color_weight, db_resnet * resnet_weight], axis=1)
                    
                    similarities = cosine_similarity(q_hybrid, db_hybrid)[0]
                    sorted_indices = np.argsort(similarities)[::-1][:top_n]
                    
                    st.markdown("---")
                    st.subheader(f"Hasil Gambar Terkemiripan Tertinggi (Top {top_n})", anchor=False)
                    
                    grid_cols = st.columns(top_n)
                    for rank_idx, db_idx in enumerate(sorted_indices):
                        with grid_cols[rank_idx]:
                            sim_score = similarities[db_idx]
                            sim_percentage = min(sim_score * 100, 100.0) 
                            img_file_path = image_paths[db_idx]
                            
                            st.metric(label=f"Peringkat ke-{rank_idx + 1}", value=f"{sim_percentage:.2f} %")
                            st.image(img_file_path, caption=os.path.basename(img_file_path), use_container_width=True)