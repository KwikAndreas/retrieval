# 🍎 Hybrid Fruit Image Retrieval System

> Proyek UAS Mata Kuliah Computer Vision

Sistem ini merupakan implementasi **Hybrid Content-Based Image Retrieval (CBIR)** untuk pencarian citra buah berdasarkan kemiripan visual.

Metode yang digunakan menggabungkan:

- HSV Color Histogram
- ResNet50 Feature Extraction
- Feature Fusion
- Cosine Similarity

---

## ✨ Fitur

- Upload gambar query
- Segmentasi objek menggunakan Otsu Thresholding
- Ekstraksi fitur warna HSV
- Ekstraksi fitur bentuk dan tekstur menggunakan ResNet50
- Pengaturan bobot fitur secara interaktif
- Menampilkan hasil pencarian Top-N
- Dashboard berbasis Streamlit

---

## 🏗️ Alur Sistem

```text
Input Image
      │
      ▼
Otsu Thresholding
      │
      ▼
Feature Extraction
 ├── HSV Histogram
 └── ResNet50
      │
      ▼
L2 Normalization
      │
      ▼
Feature Fusion
      │
      ▼
Cosine Similarity
      │
      ▼
Top-N Retrieval
```

---

## 🛠️ Teknologi

- Python
- OpenCV
- TensorFlow / Keras
- ResNet50
- Scikit-Learn
- NumPy
- Streamlit

---

## 📂 Struktur Proyek

```text
project/
├── app.py
├── data/
├── requirements.txt
```

---

## 🚀 Menjalankan Program

Clone repository:

```bash
git clone https://github.com/kwikandreas/retrieval.git
cd retrieval
```

Install dependency:

```bash
pip install -r requirements.txt
```

Jalankan aplikasi:

```bash
streamlit run app.py
```

---

## 📸 Hasil Program

### Input Query

![Query](screenshots/query.png)

### Hasil Segmentasi

![Segmentation](screenshots/segmentation.png)

### Hasil Retrieval

![Result](screenshots/result.png)

---

## 👥 Tim Pengembang

Proyek UAS Computer Vision

- Fricilia Angelica 32230
- Kwik Andreas Jonathan 32230111
- Alvian Rivaldo 32230
- Dennis Christan 32230123