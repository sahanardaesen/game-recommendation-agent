# Game Recommendation Assistant

Doğal dilde soru sorduğunuz, RAG tabanlı bir oyun tavsiye asistanı.

## Yaptıkları

Kullanıcıdan gelen sorguyu (örn. "atmosferik korku odaklı oyun öner") anlamlı bir vektöre çevirip oyun veritabanındaki en alakalı sonuçları bulur.

## Kullanılan Teknolojiler

- **Python**
- **Hugging Face Transformers** (`sentence-transformers/all-MiniLM-L6-v2` — embedding modeli)
- **ChromaDB** (vektör veritabanı)
- **LangChain / LangGraph** (planlanıyor — RAG pipeline)
- **Açık kaynak LLM** (planlanıyor — tavsiye üretimi)
- **PEFT** (isteğe bağlı — ince ayar)

## Proje Yapısı

```
├── data/
│   └── games.json          # Oyun verileri (tür, etiket, açıklama)
├── src/
│   └── indexer.py          # Embedding + ChromaDB indeksleme ve arama
├── requirements.txt
└── README.md
```

## Kurulum ve Çalıştırma

```powershell
# 1) Sanal ortam oluştur
python -m venv .venv

# 2) Bağımlılıkları kur
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3) Oyunları vektör veritabanına indeksle
.venv\Scripts\python.exe src\indexer.py

# 4) Arama testi
.venv\Scripts\python.exe src\indexer.py search "rahatlatıcı çiftçilik oyunu"
```

## Gelişim Durumu

1. ✅ Proje klasör yapısı ve bağımlılıklar
2. ✅ Oyun veritabanı hazırlama
3. ✅ Vektör embedding'ler ve vektör veritabanı
4. ⏳ RAG pipeline (LangChain + LLM)
5. ⏳ Açık kaynak LLM entegrasyonu
6. ⏳ Terminal arayüzü (İnteraktif CLI)
7. ⏳ Test ve geliştirme