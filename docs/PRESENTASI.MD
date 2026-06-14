# 🎤 CATATAN PRESENTASI UAS KOMPUTER VISION

# 🍎 Sistem Temu Kembali Citra Buah (Hybrid CBIR)

---

# SLIDE 1 — PENDAHULUAN

## Apa yang kami buat?

Kami membuat aplikasi **Image Retrieval** yang dapat mencari gambar buah yang paling mirip berdasarkan isi visual gambar.

Sistem tidak mencari berdasarkan nama file, tetapi berdasarkan:

* Warna buah
* Bentuk buah
* Tekstur buah

---

## Mengapa memilih topik Image Retrieval?

Karena pencarian gambar menggunakan nama file atau metadata sering tidak efektif.

Sebagai contoh:

Jika terdapat ratusan gambar buah, pengguna mungkin tidak mengetahui nama file yang tepat.

Dengan Image Retrieval, pengguna cukup mengunggah gambar buah dan sistem akan mencari gambar yang paling mirip secara otomatis.

---

## Tujuan Proyek

Membangun sistem pencarian citra buah menggunakan kombinasi:

* Computer Vision tradisional
* Deep Learning

agar hasil pencarian menjadi lebih akurat.

---

# SLIDE 2 — METODOLOGI

## Metode yang Digunakan

1. Otsu Thresholding
2. HSV Histogram
3. ResNet50
4. Feature Fusion
5. Cosine Similarity

---

## Alur Sistem

Input Gambar

↓

Segmentasi Objek

↓

Ekstraksi Fitur Warna (HSV)

↓

Ekstraksi Fitur Bentuk dan Tekstur (ResNet50)

↓

Normalisasi Fitur

↓

Feature Fusion

↓

Cosine Similarity

↓

Menampilkan Top-N Hasil Terbaik

---

## Penjelasan Singkat Tiap Metode

### Otsu Thresholding

Digunakan untuk memisahkan objek buah dari background.

### HSV Histogram

Digunakan untuk mengenali karakteristik warna buah.

### ResNet50

Digunakan untuk mengenali bentuk dan tekstur buah.

### Feature Fusion

Menggabungkan fitur warna dan fitur deep learning.

### Cosine Similarity

Mengukur tingkat kemiripan antara gambar query dan gambar database.

---

# SLIDE 3 — HASIL DAN PEMBAHASAN

## Struktur Program

Program dibangun menggunakan:

* Streamlit
* OpenCV
* TensorFlow/Keras
* Scikit-Learn

---

## Bagian Kode 1 — Load Model ResNet50

```python
model = ResNet50(
    weights='imagenet',
    include_top=False,
    pooling='avg'
)
```

### Yang Dijelaskan Saat Presentasi

* ResNet50 digunakan sebagai feature extractor.
* Menggunakan model yang sudah dilatih pada ImageNet.
* Tidak melakukan training ulang.
* Menghasilkan vektor fitur 2048 dimensi.

---

## Bagian Kode 2 — Segmentasi Objek

```python
cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)
```

### Yang Dijelaskan

* Mengubah gambar menjadi grayscale.
* Menggunakan threshold otomatis.
* Menghasilkan mask objek buah.

---

## Bagian Kode 3 — Ekstraksi Warna

```python
cv2.calcHist(...)
```

### Yang Dijelaskan

* Mengambil distribusi warna objek.
* Menggunakan ruang warna HSV.
* Lebih stabil dibanding RGB.

---

## Bagian Kode 4 — Ekstraksi Fitur Deep Learning

```python
features = model.predict(...)
```

### Yang Dijelaskan

* Mengambil fitur bentuk dan tekstur.
* Menghasilkan representasi visual objek.

---

## Bagian Kode 5 — Feature Fusion

```python
np.concatenate(...)
```

### Yang Dijelaskan

* Menggabungkan fitur HSV dan ResNet50.
* Bobot dapat diatur melalui slider.

---

## Bagian Kode 6 — Cosine Similarity

```python
cosine_similarity(...)
```

### Yang Dijelaskan

* Mengukur kemiripan antar vektor fitur.
* Nilai lebih tinggi berarti gambar lebih mirip.

---

# SLIDE 4 — SCREENSHOT HASIL PROGRAM

## Screenshot 1

Tampilan Awal Program

Yang disampaikan:

"Aplikasi dibuat menggunakan Streamlit sehingga pengguna dapat berinteraksi melalui browser."

---

## Screenshot 2

Hasil Segmentasi

Yang disampaikan:

"Sistem berhasil memisahkan objek buah dari background menggunakan Otsu Thresholding."

---

## Screenshot 3

Hasil Retrieval

Yang disampaikan:

"Sistem menampilkan gambar dengan tingkat kemiripan tertinggi berdasarkan perhitungan Cosine Similarity."

---

## Pembahasan Hasil

* Sistem mampu menemukan gambar buah yang mirip.
* Kombinasi HSV dan ResNet50 memberikan hasil yang lebih baik dibanding menggunakan satu fitur saja.
* Pengguna dapat mengubah fokus pencarian melalui slider.

---

# SLIDE 5 — KESIMPULAN

1. Sistem Image Retrieval berhasil dibuat menggunakan metode Hybrid CBIR.

2. HSV Histogram berhasil merepresentasikan informasi warna.

3. ResNet50 berhasil merepresentasikan bentuk dan tekstur buah.

4. Feature Fusion membuat informasi visual menjadi lebih lengkap.

5. Cosine Similarity berhasil digunakan untuk menentukan tingkat kemiripan antar gambar.

---

# SLIDE 6 — REFERENSI

* OpenCV Documentation
* TensorFlow Documentation
* Keras ResNet50 Documentation
* Scikit-Learn Documentation
* Streamlit Documentation

---

# PERTANYAAN DOSEN YANG PALING MUNGKIN MUNCUL

## Kenapa memilih HSV?

Karena HSV memisahkan warna dan pencahayaan sehingga lebih stabil dibanding RGB.

---

## Kenapa menggunakan ResNet50?

Karena ResNet50 sudah terbukti baik untuk ekstraksi fitur visual dan tidak perlu dilatih ulang.

---

## Kenapa menggunakan Cosine Similarity?

Karena output fitur berupa vektor sehingga Cosine Similarity cocok untuk mengukur kemiripan antar vektor.

---

## Apa kelebihan sistem ini?

Menggabungkan fitur warna, bentuk, dan tekstur sekaligus.

---

## Apa kekurangan sistem ini?

* Sensitif terhadap background yang kompleks.
* Dataset masih terbatas.
* Belum menggunakan segmentasi berbasis AI modern.
