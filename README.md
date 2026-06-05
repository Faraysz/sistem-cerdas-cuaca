# 🌤️ Sistem Cerdas Penentu Cuaca

Sistem prediksi cuaca sederhana berbasis **Machine Learning** menggunakan algoritma Decision Tree. Program menerima input suhu dan kelembapan dari pengguna, lalu memprediksi apakah cuaca akan **Hujan** atau **Cerah**.

---

## 📋 Deskripsi

Program ini merupakan implementasi dasar dari sistem kecerdasan buatan (AI) untuk klasifikasi cuaca. Model dilatih menggunakan data sampel suhu dan kelembapan, kemudian digunakan untuk memprediksi kondisi cuaca berdasarkan input pengguna secara real-time melalui terminal.

---

## 🛠️ Tech Stack

| Teknologi | Kegunaan |
|-----------|----------|
| Python 3.x | Bahasa pemrograman utama |
| pandas | Manipulasi dan pengolahan data |
| scikit-learn | Machine learning (DecisionTreeClassifier) |

---

## 📂 Struktur Project

```
sistem-cerdas-cuaca/
└── main.py        # File utama program
```

---

## ⚙️ Cara Instalasi & Menjalankan

### 1. Clone repository

```bash
git clone https://github.com/Faraysz/sistem-cerdas-cuaca.git
cd sistem-cerdas-cuaca
```

### 2. Install dependencies

```bash
pip install pandas scikit-learn
```

### 3. Jalankan program

```bash
python main.py
```

### 4. Masukkan input

```
Masukkan suhu (contoh 30): 30
Masukkan kelembapan (contoh 80): 80
🌧️ Prediksi Cuaca: HUJAN
```

---

## 🧠 Cara Kerja

```
Data Latih (suhu + kelembapan)
        ↓
Decision Tree Classifier (training)
        ↓
Input User (suhu & kelembapan)
        ↓
Prediksi → HUJAN 🌧️ atau CERAH ☀️
```

### Data Latih

| Suhu (°C) | Kelembapan (%) | Cuaca  |
|-----------|----------------|--------|
| 30        | 80             | Hujan  |
| 32        | 70             | Hujan  |
| 28        | 90             | Hujan  |
| 25        | 85             | Hujan  |
| 20        | 60             | Cerah  |
| 18        | 50             | Cerah  |
| 35        | 40             | Cerah  |
| 33        | 30             | Cerah  |

> **Pola:** Suhu tinggi + kelembapan tinggi → Hujan. Suhu tinggi + kelembapan rendah → Cerah.

---

## 📌 Catatan

- Data latih bersifat hardcoded (8 sampel) — cocok untuk keperluan pembelajaran
- Model tidak disimpan (di-retrain setiap kali program dijalankan)
- Untuk akurasi lebih tinggi, dataset perlu diperluas

---

## 📄 Lisensi

MIT License — bebas digunakan untuk keperluan belajar dan pengembangan.

---

> Dibuat sebagai proyek pembelajaran **Sistem Cerdas / AI** 🤖
