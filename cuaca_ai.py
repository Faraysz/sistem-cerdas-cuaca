# ==============================
# SISTEM CERDAS PENENTU CUACA
# ==============================

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# 1️⃣ DATA LATIH (contoh pengalaman)
data = {
    "suhu": [30, 32, 28, 25, 20, 18, 35, 33],
    "kelembapan": [80, 70, 90, 85, 60, 50, 40, 30],
    "cuaca": ["Hujan", "Hujan", "Hujan", "Hujan", "Cerah", "Cerah", "Cerah", "Cerah"]
}

df = pd.DataFrame(data)

# 2️⃣ Ubah teks jadi angka
df["cuaca"] = df["cuaca"].map({"Hujan": 1, "Cerah": 0})

# 3️⃣ Pisahkan input & output
X = df[["suhu", "kelembapan"]]
y = df["cuaca"]

# 4️⃣ Buat & latih model AI
model = DecisionTreeClassifier()
model.fit(X, y)

# 5️⃣ INPUT DARI USER
suhu = int(input("Masukkan suhu (contoh 30): "))
kelembapan = int(input("Masukkan kelembapan (contoh 80): "))

# 6️⃣ PREDIKSI
hasil = model.predict([[suhu, kelembapan]])

# 7️⃣ OUTPUT
if hasil[0] == 1:
    print("🌧️ Prediksi Cuaca: HUJAN")
else:
    print("☀️ Prediksi Cuaca: CERAH")
