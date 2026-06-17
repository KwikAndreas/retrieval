import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

st.set_page_config(layout="wide", page_title="Image Retrieval", page_icon="🍎")

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
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_resnet_features(img_path, model):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(img_array_expanded)
    features = model.predict(preprocessed_img, verbose=0)
    return features.flatten()

@st.cache_data
def build_database_index(db_path, _model, version=2):
    color_feats_list = []
    resnet_feats_list = []
    image_paths_list = []
    
    if not os.path.exists(db_path):
        return None, None, []

    for root, dirs, files in os.walk(db_path):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, filename)
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

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stImage > img {
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.title("Sistem Temu Kembali Citra")
st.markdown("Content-Based Image Retrieval menggunakan ekstraksi fitur hibrida (HSV Histogram & ResNet50).")

with st.spinner("Memuat indeks database..."):
    db_color, db_resnet, image_paths = build_database_index(database_path, base_model, version=2)

st.divider()

col_main, col_sidebar = st.columns([3, 1], gap="large")

with col_sidebar:
    st.subheader("Konfigurasi")
    
    with st.expander("Pembobotan Fitur", expanded=True):
        color_weight = st.slider("Bobot Warna (HSV)", 0.0, 1.0, 0.5, 0.1)
        resnet_weight = st.slider("Bobot Tekstur (ResNet50)", 0.0, 1.0, 0.5, 0.1)
        
    top_n = st.selectbox("Jumlah Hasil (Top-N)", [3, 5, 10, 15, 20], index=1)
    
    if st.button("Segarkan Database", use_container_width=True, help="Klik untuk memuat ulang database jika ada perubahan gambar atau kode."):
        st.cache_data.clear()
        
    st.caption("Format gambar yang didukung: JPG, JPEG, PNG.")

with col_main:
    st.subheader("Query Pencarian")
    uploaded_file = st.file_uploader("Unggah citra referensi", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        temp_query_path = "temp_query.jpg"
        with open(temp_query_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        q_col1, q_col2, _ = st.columns([1, 1, 2])
        with q_col1:
            st.image(uploaded_file, caption="Citra Asli", use_container_width=True)
        with q_col2:
            _, query_mask = segment_image(temp_query_path)
            st.image(query_mask, caption="Mask Segmentasi (Otsu)", use_container_width=True)

        if st.button("Jalankan Pencarian", type="primary"):
            if len(image_paths) == 0:
                st.error("Database kosong atau direktori tidak ditemukan.")
            else:
                with st.spinner("Menghitung similaritas..."):
                    q_img, q_mask = segment_image(temp_query_path)
                    q_color = extract_color_features(q_img, q_mask)
                    q_resnet = extract_resnet_features(temp_query_path, base_model)
                    
                    q_color_norm = normalize(q_color.reshape(1, -1), norm='l2').flatten()
                    q_resnet_norm = normalize(q_resnet.reshape(1, -1), norm='l2').flatten()
                    
                    q_hybrid = np.concatenate([q_color_norm * color_weight, q_resnet_norm * resnet_weight]).reshape(1, -1)
                    db_hybrid = np.concatenate([db_color * color_weight, db_resnet * resnet_weight], axis=1)
                    
                    similarities = cosine_similarity(q_hybrid, db_hybrid)[0]
                    sorted_indices = np.argsort(similarities)[::-1][:top_n]
                    
                    st.divider()
                    st.subheader("Hasil Pencarian")
                    
                    num_cols = min(top_n, 5)
                    for i in range(0, top_n, num_cols):
                        cols = st.columns(num_cols)
                        for j in range(num_cols):
                            rank_idx = i + j
                            if rank_idx < top_n:
                                db_idx = sorted_indices[rank_idx]
                                with cols[j]:
                                    sim_score = similarities[db_idx]
                                    sim_percentage = min(sim_score * 100, 100.0) 
                                    img_path = image_paths[db_idx]
                                    
                                    category = os.path.basename(os.path.dirname(img_path))
                                    
                                    st.image(img_path, use_container_width=True)
                                    st.caption(f"**{category.upper()}** • {sim_percentage:.1f}%")