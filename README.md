<div align="center">

# EN-VI Same-Speaker Cross-Lingual Speech Dataset

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets)
[![Language](https://img.shields.io/badge/Languages-Vietnamese%20%7C%20English-green.svg)](https://github.com)
[![Audio Format](https://img.shields.io/badge/Audio-24kHz%20WAV%20%7C%2044.1kHz%20MP3-orange.svg)](https://github.com)

**A Parallel Speech Corpus for Cross-Lingual Voice-Preserving Zero-Shot Speech Synthesis and Automated Dubbing.**

[Dataset Summary](#-dataset-summary) •
[Directory Structure](#-directory-structure) •
[Record Schema](#-record-schema) •
[Speaker Splits](#-speaker-splits) •
[How to Load](#-how-to-load) •
[Quality Control](#-quality-control) •
[Citation](#-citation)

</div>

---

## 📌 Dataset Summary

The **EN-VI Same-Speaker Dataset** is a paired bilingual speech corpus designed for training and evaluating cross-lingual voice cloning and automatic dubbing systems:

$$
(A_{vi}^s, T_{vi}) \longleftrightarrow (A_{en}^s, T_{en}) \quad \text{such that} \quad \text{Speaker}(A_{vi}^s) \approx \text{Speaker}(A_{en}^s)
$$

- **Vietnamese Source ($A_{vi}, T_{vi}$)**: Real human speech recordings from `VieNeu-TTS-140h` (24 kHz Lossless WAV).
- **English Cloned Dubbing ($A_{en}, T_{en}$)**: Cloned synthetic speech generated via ElevenLabs Instant Voice Cloning (`eleven_multilingual_v2`) using 60–120 seconds of isolated clean reference audio per speaker.
- **Audio Durations**: Standardized between $3.0\text{s} \le \text{duration} \le 15.0\text{s}$.
- **No Timestamp/Duration Constraints**: Speech is synthesized naturally without artificial duration compression or stretch.
- **Strict Speaker Splits**: 70% Train, 15% Validation, and 15% Test partitions isolated strictly by Speaker ID to evaluate generalization on **unseen speakers**.

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
│   ├── train_speakers.txt             # 70% Train speakers
│   ├── val_speakers.txt               # 15% Validation speakers
│   └── test_speakers.txt              # 15% Test speakers (Unseen evaluation)
├── qc/                                # Quality Control audit reports
│   ├── speaker_similarity.parquet     # ECAPA-TDNN speaker similarity scores
│   ├── asr_scores.parquet             # Whisper ASR transcription & WER/CER metrics
│   └── rejected_samples.parquet       # Filtered/rejected sample logs
└── speakers/                          # Multi-speaker audio recordings
    ├── spk_001/
    │   ├── reference/                 # 60-120s clean reference utterances
    │   │   ├── spk_001_ref_01.wav
    │   │   └── spk_001_ref_combined.wav
    │   ├── vi/                        # Original Vietnamese recordings (24 kHz WAV)
    │   │   ├── 000001.wav
    │   │   └── ...
    │   └── en/                        # Cloned English speech (MP3/WAV)
    │       ├── 000001.mp3
    │       └── ...
    ├── spk_002/
    └── ...
```

---

## 📋 Record Schema

Each record in `metadata.parquet` and `metadata.jsonl` contains the following fields:

```json
{
  "pair_id": "spk_001_000001",
  "speaker_id": "spk_001",
  "gender": "male",
  "vi_audio": "speakers/spk_001/vi/000001.wav",
  "vi_text": "Công nghệ chuyển đổi giọng nói đang phát triển rất nhanh chóng.",
  "vi_duration": 4.0,
  "en_audio": "speakers/spk_001/en/000001.mp3",
  "en_text": "Voice conversion technology is developing very rapidly.",
  "en_duration": 3.39,
  "reference_audio_ids": [
    "spk_001_ref_combined",
    "spk_001_ref_01",
    "spk_001_ref_02"
  ],
  "translation_model": "NLLB-200 / Helsinki-NLP",
  "dubbing_provider": "ElevenLabs",
  "dubbing_model": "eleven_multilingual_v2",
  "speaker_similarity_vi_en": 0.9552,
  "split": "train"
}
```

### Field Descriptions:

| Field | Type | Description |
|:---|:---|:---|
| `pair_id` | `string` | Unique sample pair identifier (`spk_xxx_xxxxxx`) |
| `speaker_id` | `string` | Speaker identifier |
| `gender` | `string` | Speaker gender (`male` / `female`) |
| `vi_audio` | `string` | Relative path to original Vietnamese audio WAV |
| `vi_text` | `string` | Vietnamese transcript text |
| `vi_duration` | `float` | Duration of Vietnamese audio in seconds |
| `en_audio` | `string` | Relative path to synthesized English audio |
| `en_text` | `string` | English translation transcript |
| `en_duration` | `float` | Duration of English audio in seconds |
| `reference_audio_ids` | `list[str]` | IDs of reference audio clips used to clone the voice profile |
| `translation_model` | `string` | Translation engine used for text translation |
| `dubbing_provider` | `string` | Voice cloning service provider |
| `dubbing_model` | `string` | TTS / Voice cloning model version |
| `speaker_similarity_vi_en`| `float` | Cosine similarity between $A_{vi}$ and $A_{en}$ embeddings |
| `split` | `string` | Dataset partition (`train`, `validation`, `test`) |

---

## 👥 Speaker Splits

The dataset strictly isolates speakers across splits to prevent speaker identity leakage during model evaluation:

- **Train Split (70%)**: `spk_001`, `spk_002`, `spk_003`, `spk_005`, `spk_006`, `spk_007`, `spk_008`
- **Validation Split (15%)**: `spk_009`, `spk_010`
- **Test Split (15%)**: `spk_004` (Held-out unseen speakers for zero-shot testing)

---

## 💻 How to Load the Dataset

### Using Pandas

```python
import pandas as pd

# Load master metadata
df = pd.read_parquet("metadata.parquet")
print(f"Total pairs: {len(df)}")
print(df[["pair_id", "speaker_id", "vi_text", "en_text", "speaker_similarity_vi_en"]].head())

# Filter by split
train_df = df[df["split"] == "train"]
test_df = df[df["split"] == "test"]
```

### Using Hugging Face Datasets

```python
from datasets import load_dataset

# Load from local directory or HuggingFace repo
dataset = load_dataset("json", data_files="metadata.jsonl")
print(dataset["train"][0])
```

### Loading Audio Waveforms in PyTorch

```python
import soundfile as sf
import torch

def load_audio_pair(row, base_dir="."):
    vi_path = f"{base_dir}/{row['vi_audio']}"
    en_path = f"{base_dir}/{row['en_audio']}"
    
    vi_audio, sr_vi = sf.read(vi_path)
    en_audio, sr_en = sf.read(en_path)
    
    return {
        "vi_tensor": torch.tensor(vi_audio, dtype=torch.float32),
        "en_tensor": torch.tensor(en_audio, dtype=torch.float32),
        "vi_text": row["vi_text"],
        "en_text": row["en_text"],
        "speaker_id": row["speaker_id"],
    }
```

---

## 🔍 Quality Control (QC)

All utterances in this corpus have passed three layers of automated verification:

1. **Signal Level Quality**:
   - Duration: $3.0\text{s} \le \text{duration} \le 15.0\text{s}$.
   - Max Clipping Ratio $< 1.0\%$.
   - Max Silence Ratio $< 25.0\%$.
2. **ASR Content Fidelity**:
   - English $\text{WER} \le 15.0\%$ (Whisper).
   - Vietnamese $\text{CER} \le 10.0\%$.
3. **Speaker Identity Preservation**:
   - Cosine similarity between reference Vietnamese speech and synthesized English speech: $\text{SIM}_{spk} \ge 0.85$ (Mean: **$0.966$**).

Audit parquet files are available under `qc/`.

---

## 📖 Citation

If you use this dataset in your research, please cite:

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
