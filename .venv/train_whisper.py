import torch
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union

def main():
    # Scegliamo il modello base inglese (molto performante per la sua stazza)
    model_id = "openai/whisper-tiny.en"  

    # 1. CARICAMENTO MODELLO E CONFIGURAZIONE LINGUA
    # Forziamo il processore e il tokenizer a gestire la lingua italiana
    processor = WhisperProcessor.from_pretrained(model_id, language="italian", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_id)

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="italian", task="transcribe")
    model.config.suppress_tokens = []

    # TRUCCO DI PERFORMANCE: Congeliamo l'encoder (la parte che ascolta)
    # In questo modo la RTX 3060 allenerà solo il decoder a scrivere in italiano, dimezzando i tempi
    model.freeze_encoder()

    # 2. CARICAMENTO DATASET AUTOMATICO (Mozilla Common Voice Italia)
    print("⏳ Scaricamento automatico del dataset Common Voice Italiano...")
    # Scarica lo split di allenamento (train) e validazione
    dataset = load_dataset("mozilla-foundation/common_voice_11_0", "it", split="train", trust_remote_code=True)
    
    # Adattiamo i nomi delle colonne per lo script
    dataset = dataset.rename_column("sentence", "transcription")
    
    # TRUCCO: Selezioniamo le prime 5000 frasi per un addestramento rapido ed efficiente
    dataset = dataset.select(range(5000)) 
    
    # Forza la frequenza audio a 16.000 Hz (lo standard obbligatorio per Whisper)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # 3. PRE-PROCESSAMENTO DEI DATI (Audio -> Spettrogramma, Testo -> Token)
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
        return batch

    print("⏳ Conversione dei file audio in corso...")
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names, num_proc=1)
    
    # Dividiamo i dati: 90% per l'addestramento vero e proprio, 10% per i test di verifica
    dataset = dataset.train_test_split(test_size=0.1)

    # 4. DATA COLLATOR (Gestisce il riempimento/padding dei testi e audio corti)
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 5. METRICA DI ACCURATEZZA (WER - Word Error Rate)
    wer_metric = evaluate.load("wer")
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # 6. PARAMETRI DI TRAINING OTTIMIZZATI PER LA RTX 3060
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-tiny-italian-custom",
        per_device_train_batch_size=8,   # Dimensione ideale per gli 12GB della 3060
        gradient_accumulation_steps=2,   # Simula un batch totale di 16 per maggiore stabilità
        learning_rate=1e-5,              # Passo di apprendimento lento per non distruggere i pesi base
        warmup_steps=100,
        max_steps=2000,                 # Numero totale di cicli di addestramento
        gradient_checkpointing=True,
        fp16=True,                      # ATTIVATO: Sfrutta i Tensor Core FP16 della tua RTX 3060 (Velocità 3x)
        evaluation_strategy="steps",
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        eval_steps=500,
        logging_steps=50,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    # 7. INIZIALIZZAZIONE E AVVIO DEL TRAINER
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.tokenizer,
    )

    print("🚀 Modello configurato. Avvio addestramento sulla RTX 3060...")
    trainer.train()

    # SALVATAGGIO DEL MODELLO FINALE
    trainer.save_model("./whisper-tiny-ita-final")
    processor.save_pretrained("./whisper-tiny-ita-final")
    print("✅ Fine-tuning completato con successo! Modello salvato in ./whisper-tiny-ita-final")

if __name__ == "__main__":
    main()
