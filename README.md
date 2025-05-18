markdown
# 🎓 Project Kuliah Semester 6

Ini adalah project tugas kuliah semester 6 yang terdiri dari dua bagian utama: **Backend** dan **Frontend**, serta penyiapan database dari folder `db_projek_export`.

---

## 📁 Struktur Folder

```

project-kuliah/
│
├── backend/                # Kode backend (API)
├── frontend/               # Kode frontend (User Interface)
├── db\_projek\_export/       # Dataset untuk MongoDB
│   ├── ds\_ulasan.7z        # Arsip berisi file CSV
│
└── README.md               # Dokumentasi project

````

---

## 🚀 Instalasi Project

### 🔧 Backend

Masuk ke folder `backend` dan jalankan perintah berikut untuk menginstal dependensi:

```bash
cd backend
npm install
```

### 🎨 Frontend

Masuk ke folder `frontend` dan jalankan perintah berikut untuk menginstal dependensi:

```bash
cd frontend
npm install
```

---

## 🍃 Setup Database MongoDB

### 1. Ekstrak File Dataset

Masuk ke folder `db_projek_export`, lalu ekstrak file:

```
ds_ulasan.7z
```

Setelah diekstrak, kamu akan mendapatkan file:

* `ds_produk.csv`
* `ds_ulasan.csv`
* `ds_sentimen.csv`

### 2. Buat Database

Buat database baru di MongoDB dengan nama:

```
db_projek
```

### 3. Import CSV ke MongoDB

Gunakan skrip Python berikut untuk mengimpor file CSV ke MongoDB:

#### 📜 import\_data.py

```python
import pandas as pd
from pymongo import MongoClient

# Koneksi ke MongoDB lokal
client = MongoClient("mongodb://localhost:27017/")
db = client["db_projek"]

# Fungsi import CSV ke MongoDB
def import_csv_to_mongo(file_path, collection_name):
    df = pd.read_csv(file_path)
    records = df.to_dict(orient='records')
    db[collection_name].insert_many(records)
    print(f"✅ Berhasil import {collection_name} ({len(records)} data)")

# Jalankan import untuk ketiga file
import_csv_to_mongo("ds_produk.csv", "ds_produk")
import_csv_to_mongo("ds_ulasan.csv", "ds_ulasan")
import_csv_to_mongo("ds_sentimen.csv", "ds_sentimen")
```

### 💡 Catatan

* Pastikan MongoDB lokal sudah berjalan di `mongodb://localhost:27017`.
* Pastikan file `ds_produk.csv`, `ds_ulasan.csv`, dan `ds_sentimen.csv` berada di **satu folder** dengan file `import_data.py`.
* Install dependensi Python sebelum menjalankan skrip:

```bash
pip install pandas pymongo
```

---

## ✅ Ringkasan

| Komponen | Teknologi                      |
| -------- | ------------------------------ |
| Backend  | Node.js, Express               |
| Frontend | React / Next.js / dll\*        |
| Database | MongoDB                        |
| Dataset  | CSV (produk, ulasan, sentimen) |

> \*Catatan: Sesuaikan dengan teknologi frontend yang kamu gunakan.

---

## ✨ Kontribusi & Pengembangan

Silakan kembangkan project ini dengan menambahkan fitur-fitur lanjutan seperti:

* Chatbot berbasis AI
* Rekomendasi produk pintar
* Statistik penjualan & ulasan
* Dashboard admin

---
