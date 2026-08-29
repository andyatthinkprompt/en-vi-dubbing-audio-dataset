<div align="center">

# EN-VI Same-Speaker Cross-Lingual Speech Dataset (1-Hour Edition)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets)
[![Speakers](https://img.shields.io/badge/Speakers-25%20Human%20Voices-blue.svg)](https://github.com)
[![Duration](https://img.shields.io/badge/Total%20Audio-~1.0%20Hour-brightgreen.svg)](https://github.com)
[![Audio Format](https://img.shields.io/badge/Audio-24kHz%20WAV%20%7C%2044.1kHz%20MP3-orange.svg)](https://github.com)

**A Parallel Speech Corpus for Cross-Lingual Voice-Preserving Zero-Shot Speech Synthesis and Automated Dubbing.**

[Dataset Summary](#-dataset-summary) •
[Dataset Scale & Metrics](#-dataset-scale--metrics) •
[Directory Structure](#-directory-structure) •
[Record Schema](#-record-schema) •
[Speaker Splits](#-speaker-splits) •
[How to Load](#-how-to-load) •
[Quality Control](#-quality-control) •
[Citation](#-citation)

</div>

---

## 📌 Dataset Summary

The **EN-VI Same-Speaker Dataset (1-Hour Edition)** is a paired bilingual speech corpus constructed for cross-lingual zero-shot voice cloning and speech-to-speech translation research:

$$
(A_{vi}^s, T_{vi}) \longleftrightarrow (A_{en}^s, T_{en}) \quad \text{such that} \quad \text{Speaker}(A_{vi}^s) \approx \text{Speaker}(A_{en}^s)
$$

- **Vietnamese Source Audio ($A_{vi}, T_{vi}$)**: Real human speech recordings extracted from `pnnbao-ump/VieNeu-TTS-140h` (24 kHz Lossless Mono WAV).
- **English Cloned Audio ($A_{en}, T_{en}$)**: Cloned synthetic speech generated via ElevenLabs Instant Voice Cloning (`eleven_multilingual_v2`) using 60–100 seconds of clean reference speech per speaker.
- **Strict Speaker Splits**: 18 Train (72%), 4 Validation (16%), and 3 Test (12%) partitions isolated strictly by Speaker ID to evaluate generalization on **unseen speakers**.

---

## 📊 Dataset Scale & Metrics

| Metric | Value |
|:---|:---|
| **Total Audio Duration** | **`56.44 minutes (~1.0 Hour)`** |
| **Total Unique Speakers** | **`25 real human speakers`** |
| **Total Bilingual Pairs** | **`111 paired utterances`** |
| **Reference Voice Clips** | **`266 clean human reference clips`** (~34.3 minutes) |
| **Target Vietnamese Duration** | **`8.67 minutes`** (520.1 seconds) |
| **Target English Duration** | **`13.47 minutes`** (808.5 seconds) |
| **Mean Speaker Similarity ($\text{SIM}_{spk}$)** | **`0.9732`** (Cosine similarity on speaker embeddings) |
| **Disk Size** | **`240.47 MB`** |

---

## 📁 Directory Structure

```text
en-vi-same-speaker-dataset/
├── README.md                          # Dataset documentation and loading instructions
├── DATASET_CARD.md                    # Hugging Face dataset card specification
├── LICENSE                            # Apache 2.0 License
├── metadata.parquet                   # Master dataset index (Apache Parquet format)
├── metadata.jsonl                     # Master dataset index (JSON Lines format)
├── splits/                            # Speaker partition definitions
│   ├── train_speakers.txt             # 18 Train speakers
│   ├── val_speakers.txt               # 4 Validation speakers
│   └── test_speakers.txt              # 3 Test speakers (Unseen evaluation)
├── qc/                                # Quality Control audit reports
│   ├── speaker_similarity.parquet     # ECAPA-TDNN speaker similarity scores
│   ├── asr_scores.parquet             # Whisper ASR transcription & WER/CER metrics
│   └── rejected_samples.parquet       # Filtered/rejected sample logs
└── speakers/                          # Multi-speaker audio recordings (spk_001 to spk_025)
    ├── spk_001/
    │   ├── reference/                 # 60-100s clean reference utterances
    │   │   ├── spk_001_ref_01.wav
    │   │   └── spk_001_ref_combined.wav
    │   ├── vi/                        # Original Vietnamese recordings (24 kHz WAV)
    │   └── en/                        # Cloned English speech (MP3/WAV)
    └── ...
```

---

## 📋 Record Schema

```json
{
  "pair_id": "spk_001_000001",
  "speaker_id": "spk_001",
  "gender": "male",
  "original_source_speaker": "jellyfish1010_0046",
  "vi_audio": "speakers/spk_001/vi/000001.wav",
  "vi_text": "Đồng thời, tăng huyết áp lâu ngày có thể gây ra tổn thương cho lớp nội mạc của thành động mạch, làm cho các tế bào trong thành mạch bị hư hại.",
  "vi_duration": 7.45,
  "en_audio": "speakers/spk_001/en/000001.mp3",
  "en_text": "At the same time, prolonged hypertension can cause damage to the inner lining of the arterial wall, damaging cells within the blood vessels.",
  "en_duration": 8.41,
  "reference_audio_ids": [
    "spk_001_ref_combined",
    "spk_001_ref_01",
    "spk_001_ref_02"
  ],
  "translation_model": "NLLB-200 / Helsinki-NLP",
  "dubbing_provider": "ElevenLabs",
  "dubbing_model": "eleven_multilingual_v2",
  "speaker_similarity_vi_en": 0.9813,
  "split": "train"
}
```

---

## 👥 Speaker Splits

- **Train Split (72% - 18 Speakers)**: `spk_001`, `spk_002`, `spk_003`, `spk_005`, `spk_006`, `spk_007`, `spk_008`, `spk_009`, `spk_010`, `spk_012`, `spk_013`, `spk_014`, `spk_015`, `spk_016`, `spk_018`, `spk_019`, `spk_020`, `spk_021`
- **Validation Split (16% - 4 Speakers)**: `spk_022`, `spk_023`, `spk_024`, `spk_025`
- **Test Split (12% - 3 Speakers - Unseen)**: `spk_004`, `spk_011`, `spk_017`

---

## 💻 How to Load the Dataset

```python
import pandas as pd

# Load master metadata
df = pd.read_parquet("metadata.parquet")
print(f"Total pairs: {len(df)}")
print(f"Total unique speakers: {df['speaker_id'].nunique()}")
print(f"Mean speaker similarity: {df['speaker_similarity_vi_en'].mean():.4f}")
```

---

## 🔍 Quality Control (QC)

- **Duration bounds**: $3.0\text{s} \le \text{duration} \le 15.0\text{s}$.
- **Clipping ratio**: $< 1.0\%$.
- **Silence ratio**: $< 25.0\%$.
- **Speaker Embedding Cosine Similarity**: Mean **`0.9732`**.

---

## 📖 Citation

```bibtex
@dataset{en_vi_same_speaker_2026,
  title={EN-VI Same-Speaker Cross-Lingual Dataset for Voice-Preserving Speech Synthesis and Dubbing},
  author={ThinkPrompt Research Team},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/thinkprompt/en-vi-same-speaker-dataset}}
}
```

---

## 📄 License

This dataset is released under the [Apache 2.0 License](LICENSE).
