# Remote GPU + PEFT (QLoRA) — Kurulum ve Kullanım

Bu klasör, paragraftaki **"uzaktan erişimle yüksek performanslı GPU sistemlerinde çalışma"** ve **"PEFT / fine-tuning"** maddelerini kanıtlamak için hazırlanmıştır.

## Mantık: kim nereye bağlanır?

```
SEN (GitHub)        SENİN PC          UZAK GPU SUNUCUSU (Colab T4 / RunPod / Vast)
   │  push/pull        │                          │
   └───── git ─────────┘                          │
                       └───── yalnızca SEN bağlanır ─► (git clone + python çalıştır)
```

- **Her şey uzakta (Seçenek 1):** Sunucuda repo `git clone` edilir, Ollama kurulur, `agent.py` çalışır. Senin bilgisayarın sadece "ekran".
- **Uzaktaki Ollama'yı API olarak kullanma (Seçenek 2):** Yerel kod, uzaktaki Ollama'ya HTTP isteği yapar. Bu opsiyonda sunucu güvenliği için SSH tüneli kullanılır, **Ollama asla internete açık bırakılmaz.**

## Güvenlik kuralları (kesin)

1. **Hiçbir anahtar/kimlik bilgisi** repoya commit edilmez.
2. `.env` dosyası `.gitignore`'da; repoya sadece boş `.env.example` girer.
3. Ollama'ya bağlantı: `http://localhost:11434` (yerel) veya **SSH tüneli** — `0.0.0.0`'a açanlar başkalarının kullanımına açılır, yapma.
4. Colab not defterindeki token'lar **senin** Google hesabına aittir; başka kimseye paylaşılmaz.
5. Çalıştırmadan önce betiklerin içeriğini okuyabilirsin — hepsi `remote/` klasöründe.

## GPU seçenekleri

| Seçenek | VRAM | Maliyet | Ne için? |
|---|---|---|---|
| **Evdeki RTX 4060** | 8 GB | 0 (zaten var) | **Gerçek QLoRA eğitimi** (süre sınırı yok) |
| **Google Colab T4** | 16 GB | 0 (oturum sınırlı) | Hızlı RAG demo + küçük eğitim denemesi |
| RunPod / Vast.ai / Lambda | 16-24 GB+ | ücretli/saatlik | Büyük modeller, uzun eğitim |

## A) Evdeki RTX 4060'ta QLoRA eğitimi (önerilen)

RTX 4060 (8 GB) Qwen2.5-3B'yi 4-bit QLoRA ile ince ayar yapar. ~15-30 dk sürer.

### 1) CUDA'lı PyTorch kur (bir kez)

Mevcut `.venv` içinde CUDA destekli PyTorch olduğundan emin ol:

```powershell
# Yeni temiz venv aç ve repo'yu klonla (veya mevcut repoya geç)
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\python.exe -m pip install "bitsandbytes>=0.43" "peft>=0.13" "accelerate>=1.0"
.venv\Scripts\python.exe -m pip install transformers sentence-transformers chromadb langchain langchain-ollama langgraph datasets
```

Not: Konsolda `nvidia-smi` ile CUDA çalışır olmalı. Bitsandbytes 0.41+ Windows'u destekler.

### 2) Veri setini üret (CPU, çok hızlı)

```powershell
.venv\Scripts\python.exe remote\make_dataset.py   # remote\data\train.jsonl (87 örnek)
```

### 3) Kuru koş — kod doğru mu? (GPU gerektirmez)

```powershell
.venv\Scripts\python.exe remote\finetune.py --dryrun
```

### 4) Eğitimi çalıştır (RTX 4060)

```powershell
.venv\Scripts\python.exe remote\finetune.py --max-steps 100
```

İyileştirme çevre değişkenleri:

- VRAM yetmezse: `--max-length 256 --batch-size 1`
- Süre yetmezse: `--max-steps 30`
- Çıktı: `remote\qlora_out\` (LoRA adaptörü — `adapter_config.json` + `adapter_model.safetensors`)

### 5) (İleri) İnce ayarlı modeli dene

Adaptörü Ollama'ya export etmek (GGUF) için: `llama.cpp` ile `convert_hf_to_gguf.py` + `llama-quantize` kullanılır. Demo için gerekli değil — `PeftModel.from_pretrained` ile PyTorch'ta yükleyip test edebilirsin.

## B) Google Colab'da hızlı RAG demo (ücretsiz T4)

`remote/colab_setup.ipynb` dosyasını aç: https://colab.research.google.com → **File > Upload notebook** → önce **Runtime > Change runtime type > T4 GPU** seç → hücreleri sırayla çalıştır.

Not defteri: repoyu klonlar, Ollama+qwen2.5 kurar, veriyi indexler, bir soru sorar. Oturum süresi sınırlı olduğundan **tam eğitim için değil, uzak GPU/Colab çalışma kanıtı** içindir.

## C) Kalıcı sunucu (RunPod / Vast.ai) — opsiyonel

- Kendi Linux VM'i kur, `ssh` ile bağlan
- `git clone https://github.com/sahanardaesen/game-recommendation-agent`
- Aşama A'daki kurulumların Linux karşılığını uygula
- İstersen Ollama'yı `OLLAMA_HOST=127.0.0.1:11434` ile tut ve **SSH tüneli** aç:
  ```powershell
  ssh -L 11434:localhost:11434 user@sunucu-ip
  ```
  Bu sayede `http://localhost:11434` senin makinenle sunucu arasında şifreli geçer.

## Dosyalar

```
remote/
├── README.md              # bu dosya (güvenlik + kurulum)
├── make_dataset.py        # sentetik eğitim verisi üretici
├── finetune.py            # QLoRA eğitim betiği (--dryrun / gerçek eğitim)
├── colab_setup.ipynb      # T4'te hızlı RAG demosu
└── data/
    └── train.jsonl        # üretilmiş 87 örnek (deterministik, seed=42)
```