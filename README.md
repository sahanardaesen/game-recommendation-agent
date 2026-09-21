# Game Recommendation Assistant

Doğal dilde soru sorduğunuz, RAG tabanlı bir oyun tavsiye asistanı.

## Yaptıkları

Kullanıcıdan gelen sorguyu (örn. "atmosferik korku odaklı oyun öner") anlamlı bir vektöre çevirip oyun veritabanındaki en alakalı sonuçları bulur, ardından açık kaynak bir LLM ile Türkçe tavsiye cümleleri üretir.

```
soru → embedding → ChromaDB (en alakalı oyunlar) → Qwen2.5 (Ollama) → Türkçe tavsiye
```

## Kullanılan Teknolojiler

- **Python**
- **Hugging Face Transformers** (`sentence-transformers/all-MiniLM-L6-v2` — embedding modeli)
- **ChromaDB** (vektör veritabanı)
- **LangChain** (RAG pipeline — prompt kurma ve LLM çağrısı)
- **Ollama + Qwen2.5 3B** (açık kaynak LLM — tavsiye üretimi)
- **PEFT** (planlanıyor — isteğe bağlı ince ayar)
- **LangGraph** (planlanıyor — pipeline orkestrasyonu)

## Proje Yapısı

```
├── data/
│   └── games.json          # Oyun verileri (tür, etiket, açıklama)
├── src/
│   ├── indexer.py          # Embedding + ChromaDB indeksleme ve arama
│   └── rag.py              # RAG pipeline (arama + LLM tavsiye)
├── requirements.txt
└── README.md
```

## Kurulum ve Çalıştırma

Ön koşul: [Ollama](https://ollama.com/download) kurulu ve `qwen2.5:3b` modeli indirilmiş olmalı.

```powershell
# 1) Sanal ortam oluştur
python -m venv .venv

# 2) Bağımlılıkları kur
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3) Oyunları vektör veritabanına indeksle
.venv\Scripts\python.exe src\indexer.py

# 4) RAG ile oyun tavsiyesi al
.venv\Scripts\python.exe src\rag.py "korku temalı oyun öner"

# (Opsiyonel) Ham vektör araması testi
.venv\Scripts\python.exe src\indexer.py search "rahatlatıcı çiftçilik oyunu"
```

## Gelişim Durumu

1. ✅ Proje klasör yapısı ve bağımlılıklar
2. ✅ Oyun veritabanı hazırlama
3. ✅ Vektör embedding'ler ve vektör veritabanı
4. ✅ RAG pipeline (ChromaDB + LangChain + Qwen2.5)
5. ⏳ LLM entegrasyonu geliştirme (prompt/cevap kalitesi)
6. ⏳ LangGraph orkestrasyonu + İnteraktif CLI
7. ⏳ Test ve geliştirme