# Whisper Tiny Fine-Tuning per la Lingua Italiana 🇮🇹
Questo repository contiene lo script e le istruzioni necessarie per eseguire il fine-tuning di **Whisper Tiny** (`openai/whisper-tiny.en`) ottimizzandolo per il riconoscimento vocale (STT) in lingua italiana, utilizzando il dataset **Mozilla Common Voice 11.0**.
Il progetto è ottimizzato per essere eseguito su schede video NVIDIA (specificamente testato su **RTX 3060 12GB**) sfruttando i Tensor Core tramite FP16 e il congelamento dell'encoder per massimizzare la velocità di addestramento.
---
## 🛠️ Fase 1: Configurazione sul PC di Sviluppo
Se stai preparando il progetto sul tuo PC attuale prima di spostarti sulla macchina da addestramento, assicurati di posizionarti nella cartella del progetto, attivare l'ambiente virtuale e generare i file necessari.
### 1. Genera il file delle dipendenze
Esegui questo comando nel terminale (es. Fish/Bash) per creare un `requirements.txt` che punti direttamente ai pacchetti compatibili con **CUDA 12.1**:
```bash
printf -- "--index-url https://download.pytorch.org/whl/cu121\ntorch\ntorchvision\ntorchaudio\nbitsandbytes\ntransformers\ndatasets\naccelerate\nsoundfile\nlibrosa\nevaluate\njiwer\ntensorboard\n" > requirements.txt
``` 
### 2. Crea lo script `train_whisper.py`
Crea il file `train_whisper.py` e incolla il codice di addestramento ottimizzato.
### 3. Sincronizza su GitHub
Salva le modifiche ed esegui il push sulla tua repository:
```bash
git add requirements.txt train_whisper.py
git commit -m "Setup completato: codice e dataset automatico"
git push origin main
```
---
## 🏃‍♂️ Fase 2: Installazione e Avvio sul PC da Training (RTX 3060)
Spostati sulla macchina dotata della GPU dedicata e segui questi passaggi in ordine.
### 1. Clonazione del Progetto e Installazione di `uv`
Clona la repository e installa il gestore di pacchetti ultra-veloce `uv`:
```bash
# Clona il progetto
git clone https://github.com/Alegott/train_STT_whisper.git
cd train_STT_whisper
# Installa uv
curl -LsSf https://astral.sh/uv/install.sh | sh
exec fish  # Riavvia la shell (o usa 'exec bash' a seconda del tuo sistema)
```
### 2. Installazione di FFmpeg (Obbligatorio)
Whisper richiede l'eseguibile di sistema `ffmpeg` per la manipolazione dei file audio.
- Su Arch Linux: `sudo pacman -S ffmpeg`
- Su Ubuntu/Debian: `sudo apt update && sudo apt install ffmpeg`
### 3. Creazione Ambiente Virtuale e Dipendenze
Configura l'ambiente con Python 3.12 e installa i pacchetti tramite `uv pip` in pochissimi secondi:
```bash
uv venv .venv --python 3.12
source .venv/bin/activate
# Installa PyTorch CUDA 12.1 e le librerie di training
uv pip install -r requirements.txt
```
### 4. Test di Verifica della GPU
Verifica che PyTorch veda correttamente la tua RTX 3060:
```bash
python -c "import torch; print('GPU Rilevata:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'ERRORE')"
```
> **Nota:** Se l'output restituisce `NVIDIA GeForce RTX 3060`, il sistema è pronto per l'addestramento.
---
## 🚀 Avvio dell'Addestramento
Per far partire il processo di fine-tuning, esegui semplicemente:
```bash
python train_whisper.py
```
