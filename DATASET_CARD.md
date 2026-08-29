---
language:
- vi
- en
tags:
- speech-synthesis
- text-to-speech
- voice-cloning
- cross-lingual
- audio-dubbing
- zero-shot-tts
license: apache-2.0
task_categories:
- text-to-speech
- audio-to-audio
pretty_name: "EN-VI Same-Speaker Cross-Lingual Dataset"
size_categories:
- 1K<n<10K
---

# EN-VI Same-Speaker Cross-Lingual Speech Dataset

A specialized bilingual parallel speech corpus designed for **Cross-Lingual Speaker-Preserving Zero-Shot Speech Synthesis and Dubbing**.

$$
(A_{ref}, T_{target}) \rightarrow \hat{A}_{target} \quad \text{such that} \quad \text{Speaker}(\hat{A}_{target}) \approx \text{Speaker}(A_{ref})
$$

## Dataset Summary

- **Source Corpus**: Derived from `VieNeu-TTS-140h` Vietnamese multi-speaker dataset.
- **Voice Cloning & Dubbing Engine**: ElevenLabs Instant Voice Cloning (IVC) (`eleven_multilingual_v2`).
- **Translation Engine**: NLLB-200 / Helsinki-NLP / Human-verified English translations.
- **Target Sampling Rate**: 24 kHz Lossless WAV & 44.1 kHz MP3.
- **Strict Speaker Splitting**: 70% Train, 15% Validation, 15% Test without speaker overlap.

## Dataset Structure

```text
data/en_vi_same_speaker/
├── metadata.parquet
├── metadata.jsonl
├── splits/
│   ├── train_speakers.txt
│   ├── val_speakers.txt
│   └── test_speakers.txt
├── qc/
│   ├── speaker_similarity.parquet
│   ├── asr_scores.parquet
│   └── rejected_samples.parquet
└── speakers/
    ├── spk_001/
    │   ├── reference/        # 60-120s clean reference utterances
    │   ├── vi/               # Original Vietnamese audio
    │   └── en/               # Synthesized English audio
    └── ...
```

## Record Schema

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
  "reference_audio_ids": ["spk_001_ref_combined", "spk_001_ref_01"],
  "translation_model": "NLLB-200 / Helsinki-NLP",
  "dubbing_provider": "ElevenLabs",
  "dubbing_model": "eleven_multilingual_v2",
  "speaker_similarity_vi_en": 0.9552,
  "split": "train"
}
```

## Quality Control (QC) Metrics

1. **Audio SNR & Silence**: All audio samples satisfy $3.0\text{s} \le \text{duration} \le 15.0\text{s}$, clipping $< 1\%$, silence ratio $< 25\%$.
2. **Content Fidelity**: Whisper ASR $\text{WER}_{EN} \le 15\%$, $\text{CER}_{VI} \le 10\%$.
3. **Speaker Similarity**: Cosine distance on frozen speaker embeddings ($\ge 0.65$, mean $> 0.95$).

## Citation

```bibtex
@dataset{en_vi_same_speaker_2026,
  title={EN-VI Same-Speaker Cross-Lingual Dataset for Voice-Preserving Speech Synthesis and Dubbing},
  author={ThinkPrompt Research},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/thinkprompt/conditional-audio-dubbing}}
}
```
