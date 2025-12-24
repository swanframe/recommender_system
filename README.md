# recommender_system

Recommender System sederhana berbasis:
- **Popularity**: item terpopuler berdasarkan total `watch_seconds`
- **Item-based Collaborative Filtering**: cosine similarity antar item (berdasarkan matriks user-item dari `watch_seconds`)

Tech stack: **Python, Pandas, Scikit-Learn, FastAPI**

---

## Struktur Proyek

```txt
recommender_system/
├─ src/
│  └─ recommender_system/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ recommender.py
│     ├─ main.py
│     └─ data/
│        ├─ __init__.py
│        └─ data_loader.py
├─ data/
│  ├─ raw/
│  │  ├─ users.csv
│  │  ├─ items.csv
│  │  └─ events.csv
│  └─ processed/
├─ tests/
└─ requirements.txt
````

---

## Dataset

CSV yang dibutuhkan (letakkan di `data/raw/`):

* `users.csv`: `(user_id, name, age, gender, region)`
* `items.csv`: `(item_id, title, content_type, genre)`
* `events.csv`: `(user_id, item_id, event_type, watch_seconds, timestamp)`

---

## Setup

Disarankan pakai virtual environment.

### 1) Buat venv & install dependencies

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Pastikan data tersedia

```txt
data/raw/users.csv
data/raw/items.csv
data/raw/events.csv
```

---

## Menjalankan API

Dari root project:

```bash
uvicorn recommender_system.main:app --app-dir src --reload --port 8000
```

API akan tersedia di:

* Health: `http://127.0.0.1:8000/health`
* Docs Swagger: `http://127.0.0.1:8000/docs`

> Optional: override lokasi data dengan env var:

```bash
export DATA_RAW_DIR="/absolute/path/to/recommender_system/data/raw"
```

---

## Endpoint

### 1) GET /health

Cek status service.

Response contoh:

```json
{"status":"ok"}
```

### 2) GET /popular?k=

Mengembalikan item populer berdasarkan total `watch_seconds`.

Response contoh:

```json
{
  "k": 5,
  "items": [
    {"item_id":"i3","title":"Hospital Playlist","score":12345.0}
  ]
}
```

### 3) GET /recommendations?user_id=&k=

Mengembalikan rekomendasi untuk user.
Response selalu berisi `fallback_used`.

* Jika user baru / tidak punya interaksi ⇒ `fallback_used=true` dan sistem memakai popular.
* Jika tidak ⇒ item-based cosine similarity (`fallback_used=false`)

Response contoh:

```json
{
  "user_id": "u207",
  "k": 5,
  "fallback_used": false,
  "items": [
    {"item_id":"i5","title":"Extraordinary Attorney Woo","score":0.123}
  ]
}
```

---

## Algoritma Singkat

### Popularity

1. Hitung `sum(watch_seconds)` per `item_id`
2. Urutkan menurun → ambil top-k

### Item-based Cosine Similarity

1. Bangun matriks `user-item` dari `sum(watch_seconds)` per `(user_id, item_id)`
2. Hitung cosine similarity antar vektor item (item-item similarity)
3. Untuk user tertentu:

   * Skor item = `Σ(sim(item, item_yang_ditonton) * watch_seconds_user_pada_item_yang_ditonton)`
4. Filter:

   * Jangan rekomendasikan item yang sudah ditonton total **> 600 detik** oleh user tersebut
5. Jika rekomendasi kurang dari k, sistem menambahkan dari popular sebagai “top up”.

---

## Testing Cepat (curl)

### Health

```bash
curl "http://127.0.0.1:8000/health"
```

### Popular

```bash
curl "http://127.0.0.1:8000/popular?k=5"
```

### Recommendations

```bash
curl "http://127.0.0.1:8000/recommendations?user_id=u207&k=5"
```

---

## Final Check (Checklist)

* [ ] File CSV ada di `data/raw/`
* [ ] `pip install -r requirements.txt` sukses
* [ ] `uvicorn recommender_system.main:app --app-dir src --reload` jalan
* [ ] `/docs` bisa dibuka
* [ ] `/popular` dan `/recommendations` mengembalikan response valid

````