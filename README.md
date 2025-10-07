# 🖼️ Image Processing App – PyQt5 Edition

Aplikasi pengolahan citra digital berbasis **Python + PyQt5 + PIL + OpenCV**.  
Didesain untuk kebutuhan **praktikum, penelitian, dan eksplorasi teknik pengolahan citra** dengan antarmuka interaktif dan mudah digunakan.

---

## ✨ Fitur Utama

### 🎨 Transformasi & Manipulasi Citra
- Flip Horizontal & Vertical  
- Rotate (dengan slider interaktif)  
- Translate (geser posisi gambar)  
- Zoom In / Out (tanpa mengubah kualitas)  
- Crop Area (drag pada gambar)  
- Remove Background otomatis  

### 🧮 Operasi Aritmatika
- Penjumlahan, Pengurangan, Perkalian, dan Pembagian antar citra  
- Operasi Bitwise (AND / OR)  
- Blending Transparansi & Horizontal Fade  

### 🧩 Pengolahan Morfologi
- Erosi, Dilasi, Opening, Closing  
- Pilihan kernel: Square / Cross / Custom  

### 🌈 Pewarnaan & Grayscale
- Konversi warna (Red, Cyan, Orange, Brown, Gray, Purple, dll)  
- Metode grayscale: Lightness, Luminance, Average  

### 📊 Histogram & Fuzzy Equalization
- Histogram Input / Output  
- Equalisasi Histogram  
- Fuzzy Histogram Equalization (Grayscale & RGB)  

### 🔍 Segmentasi Citra
- Global Thresholding  
- Adaptive Thresholding  
- K-Means Clustering  
- Watershed Segmentation  
- Region Growing  

---

## 🧠 Teknologi yang Digunakan

| Komponen | Deskripsi |
|-----------|------------|
| **Python 3.13+** | Bahasa utama |
| **PyQt5** | GUI interaktif |
| **Pillow (PIL)** | Manipulasi citra dasar |
| **OpenCV** | Operasi lanjutan dan segmentasi |
| **Matplotlib** | Visualisasi histogram dan hasil segmentasi |
| **NumPy** | Pengolahan data numerik |

---

## 🧰 Cara Menjalankan Aplikasi

### 1️⃣ Clone Repository
```bash
git clone https://github.com/riandafatur28/imageProcessing.git
cd imageProcessing
