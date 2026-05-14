# moshi-distill

Knowledge distillation of [Moshi](https://github.com/kyutai-labs/moshi) — Kyutai's 7.7B-parameter real-time voice language model — into a 1.7B student using SmolLM2, running entirely on free-tier Kaggle dual-T4 GPUs.

> **Authors:** Taslim Hossain Tamim · Afia Mubassira Ali Raisa · Sami Uddin  
> **Course:** CSE 465 — Machine Learning Systems  

---

## What This Project Does

Moshi is the first fully open real-time duplex voice AI — it listens and speaks simultaneously. At 7.7B parameters and 15.4 GB VRAM just for inference, it's out of reach for most hardware. This project compresses it into a 1.7B student that:

- Fits in **~6 GB VRAM** at inference time (2.57× reduction)
- Has **4.7× fewer** trainable parameters
- Stays fully compatible with Moshi's streaming inference interface
- Can run the official Moshi web UI unmodified

The core idea: replace Moshi's 32-layer Helium Temporal Transformer with SmolLM2-1.7B, keep all other Moshi components intact (Mimi audio codec, Depformer, frozen heads), and train via a four-phase knowledge distillation curriculum.

---

## Architecture

```
Teacher (kyutai/moshiko-pytorch-bf16, 7.7B)
    │
    │  Phase 0: run teacher over 500h of LibriSpeech
    │  cache hidden states + top-256 text logits (220 GB)
    ▼
┌─────────────────────────────────────────────┐
│              Student Model                  │
│                                             │
│  Frozen Moshi embeddings (emb, text_emb)   │
│         │  [B, T, 4096]                     │
│         ▼                                   │
│   in_adapter  (4096 → 2048)                 │
│         │                                   │
│   SmolLM2-1.7B backbone  ← TRAINABLE        │
│   (24 layers, hidden=2048)                  │
│         │                                   │
│   out_adapter (2048 → 4096)                 │
│         │  [B, T, 4096]                     │
│         ▼                                   │
│  Frozen Moshi heads (out_norm, text_linear, │
│  depformer, linears)                        │
└─────────────────────────────────────────────┘
```

**Trainable:** SmolLM2 backbone + adapters = 1.627B params  
**Frozen:** all original Moshi modules = 1.111B params  
**Total student shell:** 2.738B params

---

## Training Pipeline

The project runs in 10 sequential notebook sessions:

| Step | Notebook | What It Does |
|------|----------|--------------|
| 1 | `01_inference-full-moshi.ipynb` | Early exploration / baseline attempt |
| 2 | `02_session-s0-smoke-test.ipynb` | Validates full compute graph on dual T4 (10 fake steps) |
| 3 | `03_session-s1-teacher-caching.ipynb` | Pilot 600-window cache + exports frozen Moshi heads |
| 4 | `04_session-s2-full-cache.ipynb` | Full 60,000-window teacher cache across 3 shards × 4 parts |
| 5 | `05_session-s14-codes-cache.ipynb` | Caches Mimi codes for Phase 4 supervision |
| 6 | `06_phase1-hidden-bootstrap.ipynb` | Phase 1: cosine loss on cached hidden states |
| 7 | `07_phase2-logit-alignment.ipynb` | Phase 2: adds sparse JSD on cached text logits |
| 8 | `08_phase3-attention-polish.ipynb` | Phase 3: adds attention KL with live teacher |
| 9 | `09_phase4-depformer.ipynb` | Phase 4: Depformer co-adaptation (in progress) |
| 10 | `10_student-web-demo.ipynb` | Live browser voice demo via ngrok |

### Loss Curriculum

| Phase | Loss | Gate to next phase |
|-------|------|--------------------|
| P1 | `1.0 × cosine(hidden)` | val CosSim > 0.80 |
| P2 | `1.0 × cosine + 0.1 × JSD(text logits)` | val text KL < 0.20 |
| P3 | `1.0 × cosine + 0.1 × JSD + 0.05 × KL(attention)` | val CosSim > 0.85 |
| P4 | `CE(depformer codebooks)` | CE < ln(2048) ≈ 7.62 |

---

## Results

| Phase | Steps | Val CosSim | Val Text JSD | Status |
|-------|-------|-----------|--------------|--------|
| P1 — Hidden bootstrap | 2,058 | **0.896** | — | ✅ Gate passed |
| P2 — Logit alignment | 611 | ~0.890 | **0.041** | ✅ Gate passed |
| P3 — Attention polish | 1,056 | **0.900** | ~0.040 | ✅ Gate passed |
| P4 — Depformer (diagnostic) | 500 | frozen | — | 🔄 In progress |

**VRAM at inference:** ~6.0 GB (vs 15.4 GB teacher) — **2.57× reduction**  
**Total training cost:** ~80–100 GPU-hours on free Kaggle T4s

---

## Datasets (Kaggle)

All cached datasets are publicly available on Kaggle:

| Dataset | Contents | Size |
|---------|----------|------|
| `mhassann/moshi-cache-s0p0` … `s2p3` (×12) | Teacher hidden states + top-256 logits | ~18 GB each |
| `mhassann/moshi-cache-codes` | Mimi codes for all 60,000 windows | 2.3 GB |
| `mhassann/moshi-p1-ckpt` | Phase 1 student checkpoint (step 2058) | 6.6 GB |
| `mhassann/moshi-p2-ckpt` | Phase 2 student checkpoint (step 611) | 6.6 GB |
| `mhassann/moshi-p3-ckpt` | Phase 3 student checkpoint (step 1056) | 6.6 GB |
| `tasfiatanha/moshi-frozen-heads` | Frozen Moshi head weights | ~2 GB |
| `tasfiatanha/moshi-repo` | Moshi source code (editable install) | — |

---

## Reproducing This Work

### Requirements

- Kaggle account with GPU T4 × 2 runtime (free tier works)
- Internet access enabled in the Kaggle notebook settings
- ~25 GB free disk space per caching session

### Step 1 — Smoke Test

Open `02_session-s0-smoke-test.ipynb` on Kaggle. Attach:
- `tasfiatanha/moshi-repo`

Run all cells. Every cell must pass with no OOM and finite losses before proceeding.

### Step 2 — Generate Teacher Cache

Run `04_session-s2-full-cache.ipynb` 12 times, once for each combination of:
```
SHARD_IDX ∈ {0, 1, 2}
PART_IDX  ∈ {0, 1, 2, 3}
```
Each run produces one Kaggle dataset (`moshi-cache-s{N}p{M}`).  
Then run `05_session-s14-codes-cache.ipynb` to produce `moshi-cache-codes`.

Verify all 12 + 1 datasets exist before proceeding.

### Step 3 — Train Phase 1

Attach all 12 cache datasets + `moshi-frozen-heads` + `moshi-cache-codes` + `moshi-repo`.  
Run `06_phase1-hidden-bootstrap.ipynb` until val CosSim > 0.80 (expect ~2000 steps).  
Checkpoint is saved to `mhassann/moshi-p1-ckpt`.

### Step 4 — Train Phase 2

Attach the Phase 1 checkpoint dataset in addition to the datasets from Step 3.  
Run `07_phase2-logit-alignment.ipynb` until val text KL < 0.20 (expect ~600 steps).  
Checkpoint saved to `mhassann/moshi-p2-ckpt`.

### Step 5 — Train Phase 3

Attach the Phase 2 checkpoint. Run `08_phase3-attention-polish.ipynb`.  
Note: Phase 3 re-loads the teacher onto `cuda:1` — ensure both GPUs have headroom.  
Checkpoint saved to `mhassann/moshi-p3-ckpt`.

### Step 6 — Run the Demo

Attach the Phase 3 checkpoint + `moshi-frozen-heads` + `moshi-repo`.  
Run `10_student-web-demo.ipynb`.  
Open the printed ngrok URL in a browser with microphone access enabled.

> **Note:** Replace `mhassann` and `tasfiatanha` in dataset path strings with your own Kaggle username before running.

---

## Key Engineering Notes

**Why cache-first?**  
The 7.7B teacher alone needs 15.4 GB. The student training graph needs another ~9.5 GB. They can't coexist on 2× T4s simultaneously (except Phase 3 where the teacher fits on `cuda:1` in inference-only mode). Caching decouples the two completely.

**Why are torch.compile and CUDAGraphed disabled?**  
Kaggle T4s (sm_75) lack hardware bfloat16. PyTorch's Inductor occasionally emits bf16 intrinsics anyway, causing crashes. CUDA graph capture also conflicts with gradient checkpointing. Both are patched out at the start of every session.

**Why JSD instead of KL for logit alignment?**  
JSD is symmetric and bounded in [0, ln 2], making it numerically stable when the student's dense softmax and the teacher's sparse top-256 distribution partially disagree. Forward KL would penalize the student heavily for mass placed outside the top-256 support.

**Why is Phase 4 not converging?**  
The Depformer was trained by Kyutai against the original 7.7B TT's exact hidden distribution. Even a 10% cosine gap (cos_sim = 0.90 instead of 1.0) is enough to push Depformer outputs into collapsed repetition (`<0x14>` babble). This is the main open problem — see the paper for proposed fixes.

---

## Repository Structure

```
moshi-distill/
├── notebooks/
│   ├── 01_inference-full-moshi.ipynb
│   ├── 02_session-s0-smoke-test.ipynb
│   ├── 03_session-s1-teacher-caching.ipynb
│   ├── 04_session-s2-full-cache.ipynb
│   ├── 05_session-s14-codes-cache.ipynb
│   ├── 06_phase1-hidden-bootstrap.ipynb
│   ├── 07_phase2-logit-alignment.ipynb
│   ├── 08_phase3-attention-polish.ipynb
│   ├── 09_phase4-depformer.ipynb
│   └── 10_student-web-demo.ipynb
├── moshi/
│   └── models/
│       └── smol_temporal.py        ← SmolTemporalTransformer implementation
├── paper/
│   └── moshi_compression_paper.tex ← IEEE-format research paper (LaTeX)
└── README.md
```

---

## Citation

If you use this work, please cite:

```bibtex
@misc{tamim2025moshidistill,
  title   = {Compressing Moshi: Knowledge Distillation of a 7.7B-Parameter
             Voice Language Model into a 1.7B Student via Staged
             Cache-and-Train on Commodity GPUs},
  author  = {Taslim Hossain Tamim and Afia Mubassira Ali Raisa and Sami Uddin},
  year    = {2025},
  url     = {https://github.com/taslimhossaintamim/moshi-distill}
}
```

---

## Acknowledgements

- [Kyutai](https://kyutai.org) for releasing Moshi and its weights under an open license
- [HuggingFace](https://huggingface.co) for SmolLM2 and model hosting
- [Kaggle](https://kaggle.com) for free GPU compute
- [LibriSpeech](https://www.openslr.org/12) for the open speech corpus

---

## License

Code in this repository is released under the MIT License.  
Moshi model weights are subject to [Kyutai's model license](https://github.com/kyutai-labs/moshi/blob/main/LICENSE).  
LibriSpeech data is released under CC BY 4.0.
