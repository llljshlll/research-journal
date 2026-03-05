# FLUX.1-dev LoRA Fine-tuning for Dental 3DGS Refinement

> FLUX.1-dev에 LoRA를 적용하여 3D Gaussian Splatting(3DGS) 렌더링 결과물의 치아 이미지 품질을 향상시키는 연구입니다.
> FlowEdit 파이프라인과 결합하여 noisy 렌더를 clean 치아 이미지로 변환합니다.

---

## Overview

3DGS로 생성한 치아 렌더링 이미지에는 floating artifact, blurry edge, semi-transparent noise 등의 품질 문제가 존재합니다.
본 연구에서는 FLUX.1-dev 모델에 LoRA를 fine-tuning하여 FlowEdit의 velocity field 예측을 치아 도메인에 특화시킵니다.

| Source (3DGS render) | Target (GT dental photo) | FlowEdit + LoRA result |
|:---:|:---:|:---:|
| ![source](<!-- source image link -->) | ![target](<!-- target image link -->) | ![result](<!-- result image link -->) |

---

## Method

### FlowEdit + LoRA 결합 원리

FlowEdit은 inversion 없이 이미지를 편집하는 ODE 기반 방법입니다:

$$\Delta V_t = V_t^{\text{tar}}(z_t, p_{\text{tar}}) - V_t^{\text{src}}(z_t, p_{\text{src}})$$

FLUX transformer가 두 velocity field를 모두 예측하므로, LoRA로 transformer를 fine-tuning하면 $V_t^{\text{src}}$와 $V_t^{\text{tar}}$ 모두 치아 도메인에 특화됩니다.

### Training Mode: `both`

```
source image + source_text  →  flow matching loss   (Vt_src 개선)
target image + target_text  →  flow matching loss   (Vt_tar 개선)
```

두 방향을 모두 학습함으로써 FlowEdit의 delta 계산 정확도를 높입니다.

### Flow Matching Loss

$$z_t = (1 - \sigma) \cdot z_0 + \sigma \cdot \epsilon, \quad \sigma \sim \text{Logit-Normal}(0, 1)$$

$$v_{\text{target}} = \epsilon - z_0, \quad \mathcal{L} = \|v_{\text{pred}} - v_{\text{target}}\|^2$$

---

## LoRA Architecture

### 적용 위치

FLUX transformer는 **Double Stream Block × 19** + **Single Stream Block × 38** 으로 구성됩니다.

**Double Stream Block**

```
txt stream  ──▶  Linear [V, K, Q]          ← add_q/k/v_proj   ✅ LoRA
img stream  ──▶  Linear [V, K, Q]          ← to_q/k/v         ✅ LoRA
                     │
               RoPE Attention ──▶ split
                     │
            txt output Linear              ← add_out_proj      ✗ (미포함)
            img output Linear              ← to_out.0          ✅ LoRA
                     │
txt FFN:  Linear ──▶ GeLU ──▶ Linear       ← ff_context.net.0.proj / .2  ✅ LoRA
img FFN:  Linear ──▶ GeLU ──▶ Linear       ← ff.net.0.proj / .2          ✅ LoRA
```

**Single Stream Block**

```
x  ──▶  Linear [V, K, Q]  ──▶  RoPE Attention  ──┐
         (to_q/k/v) ✅                            │
                                                  CAT ──▶ Linear (to_out.0) ✅
        Linear ──▶ GeLU  ─────────────────────────┘
   (ff.net.0.proj) ✅  (ff.net.2) ✅
```

### LoRA 설정 (dental_v1 기준)

| 항목 | 값 |
|---|---|
| Target modules | 11개 패턴 (Q/K/V, output proj, FFN) |
| 총 LoRA 레이어 수 | 323개 |
| Rank (r) | 4 |
| Alpha | 4 |
| Trainable params | 10,739,712 (0.09%) |
| Base model params | 11.9B |

---

## Dataset

### 구성

- **Source**: 3DGS 렌더링 이미지 (noisy, floating artifact 포함)
- **Target**: 실제 치아 임상 사진 (clean GT)
- **총 54 pairs** → `both` mode로 **108 samples**

### 촬영 각도별 캡션

| Ring | 각도 | Source 캡션 | Target 캡션 |
|---|---|---|---|
| ring1 | top-down | `3DGS render of a tooth, floating artifacts, ... top-down view` | `Clinical dental photograph, ... top-down view` |
| ring2 | upper angle | `... upper angle view` | `... upper angle view` |
| ring3 | lateral | `... lateral view` | `... lateral view` |
| ring4 | lower angle | `... lower angle view` | `... lower angle view` |
| ring5 | bottom-up | `... bottom-up view` | `... bottom-up view` |

### 데이터셋 구조

```
LoRA/dataset/
├── images/
│   ├── source/   # 00000.png ~ 00053.png  (3DGS renders)
│   └── target/   # 00000.png ~ 00053.png  (GT dental photos)
├── test/
│   ├── noise/    # 테스트용 noisy 입력
│   └── result/   # FlowEdit 출력 결과
└── metadata.jsonl
```

---

## Training

### VRAM 최적화 (RTX 4090 24GB)

| 컴포넌트 | 최적화 | VRAM |
|---|---|---|
| FLUX transformer (12B) | 4-bit NF4 QLoRA | 22GB → **~5.5GB** |
| VAE | CPU offload (encode/decode 시만 GPU) | ~0.5GB |
| CLIP text encoder | CPU offload | ~0.5GB |
| T5-XXL text encoder | CPU offload | ~9GB |
| Optimizer | 8-bit AdamW | state 절반 절약 |
| **총 peak VRAM** | | **~15.5GB** |

### 학습 명령

```bash
# 처음부터 학습
conda run -n skyfall-gs_copy python LoRA/train_flux_lora.py \
    --config LoRA/configs/dental.yaml

# 체크포인트에서 재개 (epoch 30 → 70)
conda run -n skyfall-gs_copy python LoRA/train_flux_lora.py \
    --config LoRA/configs/dental.yaml \
    --resume_from LoRA/output/dental_v1/checkpoint_epoch30 \
    --start_epoch 30 \
    --num_epochs 70
```

### 학습 설정 (`configs/dental.yaml`)

```yaml
lora_rank: 4
lora_alpha: 4
num_epochs: 30
learning_rate: 1.0e-4
batch_size: 1
train_mode: "both"
use_qlora: true
use_8bit_adam: true
```

---

## Inference (FlowEdit + LoRA)

### 실행

```bash
# LoRA 적용
conda run -n skyfall-gs_copy python LoRA/test_flowedit.py \
    --lora_path LoRA/output/dental_v1/checkpoint_epoch20 \
    --input_dir  LoRA/dataset/test/noise \
    --output_dir LoRA/dataset/test/result

# Base FLUX (비교용, LoRA 없이)
conda run -n skyfall-gs_copy python LoRA/test_flowedit.py \
    --input_dir  LoRA/dataset/test/noise \
    --output_dir LoRA/dataset/test/result_base
```

### FlowEdit 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `T_steps` | 28 | 총 denoising step |
| `n_max` | 15 | editing 적용 step 수 (클수록 더 많이 바뀜) |
| `n_min` | 0 | regular denoising 마무리 step |
| `src_guidance` | 1.5 | source prompt guidance scale |
| `tar_guidance` | 5.5 | target prompt guidance scale |

### 결과 비교

| Epoch | Input | Base FLUX | LoRA result |
|:---:|:---:|:---:|:---:|
| 10 | ![](<!-- input -->) | ![](<!-- base -->) | ![](<!-- ep10 -->) |
| 20 | ![](<!-- input -->) | ![](<!-- base -->) | ![](<!-- ep20 -->) |
| 30 | ![](<!-- input -->) | ![](<!-- base -->) | ![](<!-- ep30 -->) |

---

## 파일 구조

```
LoRA/
├── train_flux_lora.py      # 학습 스크립트
├── test_flowedit.py        # FlowEdit + LoRA 추론
├── flowedit_with_lora.py   # 단일 이미지 추론 (legacy)
├── configs/
│   └── dental.yaml         # 학습 설정
├── dataset/
│   ├── images/source/      # 학습 source 이미지
│   ├── images/target/      # 학습 target 이미지
│   ├── test/               # 테스트 이미지
│   ├── metadata.jsonl      # 캡션 및 페어 정보
│   └── prepare_dataset.py  # 데이터셋 전처리
└── output/
    └── dental_v1/
        ├── checkpoint_epoch5/
        ├── checkpoint_epoch10/
        └── ...
```

---

## Notes

- **FlowEdit_utils.py 버그 수정**: diffusers ≥ 0.29에서 `vae_scale_factor=8`로 변경되어 `prepare_latents`의 image_ids 계산이 틀리는 문제 → `_prepare_latent_image_ids`를 latent 실제 크기 기준으로 직접 호출하도록 수정
- **QLoRA + PEFT**: `prepare_model_for_kbit_training`은 LLM 전용(get_input_embeddings 필요)이므로 diffusion transformer에 미적용, `requires_grad_(False)` + `get_peft_model()` 순서로 처리
