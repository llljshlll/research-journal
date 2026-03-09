# FLUX.1-dev LoRA Fine-tuning for Dental 3DGS Refinement

> FLUX.1-dev에 LoRA를 적용하여 3D Gaussian Splatting(3DGS) 렌더링 결과물의 치아 이미지 품질을 향상시키는 연구입니다.
> FlowEdit 파이프라인과 결합하여 noisy 렌더를 clean 치아 이미지로 변환합니다.

---

## Overview

3DGS로 생성한 치아 렌더링 이미지에는 floating artifact, blurry edge, semi-transparent noise 등의 품질 문제가 존재합니다.
본 연구에서는 FLUX.1-dev 모델에 LoRA를 fine-tuning하여 FlowEdit의 velocity field 예측을 치아 도메인에 특화시킵니다.

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

(이미지 첨부 /Users/jangseohyeon/Desktop/연구실/research-journal/docs/assets/projects/skyfall-gs/LoRA/LoRA_double_stream.png)

**Single Stream Block**

(이미지 첨부 /Users/jangseohyeon/Desktop/연구실/research-journal/docs/assets/projects/skyfall-gs/LoRA/LoRA_single_stream.png)


---

## Dataset

### 구성

- **Source**: 3DGS 렌더링 이미지 (noisy, floating artifact 포함)
- **Target**: 실제 치아 임상 사진 (clean GT)
- **총 54 pairs** → `both` mode로 **108 samples**
| Source (3DGS render) | Target (GT dental photo) |
|:---:|:---:|
| ![source](<!-- source image link -->) | ![target](<!-- target image link -->) |


---


### 학습 설정 

```yaml
lora_rank: 4
lora_alpha: 4
num_epochs: 70
learning_rate: 1.0e-4
batch_size: 1
train_mode: "both"
use_qlora: true
use_8bit_adam: true
```

---

### 결과 비교

| Epoch | Input | Base FLUX | LoRA result |
|:---:|:---:|:---:|:---:|
| 10 | ![](<!-- input -->) | ![](<!-- base -->) | ![](<!-- ep10 -->) |
| 20 | ![](<!-- input -->) | ![](<!-- base -->) | ![](<!-- ep20 -->) |
| 30 | ![](<!-- input -->) | ![](<!-- base -->) | ![](<!-- ep30 -->) |

---
