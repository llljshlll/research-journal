# TINKER: Diffusion’s Gift to 3D — Multi-View Consistent Editing from Sparse Inputs

3DGS + diffusion 기반 3D editing 논문  
per-scene optimization 없이  
**1~2장 입력만으로 multi-view consistent 3D editing**을 목표로 함  

기존 3D editing 파이프라인이  
“multi-view image 생성 → scene별 GS/NeRF finetuning”에 의존하던 구조라면,  
**TINKER는 diffusion foundation model의 latent 3D prior를 직접 활용**하여  
multi-view consistency를 image 단계에서 해결하는 접근.

---

## 1. Motivation

### Problem
- **Sparse input 기반 3D editing**
  - 실제 환경에서는 모든 view를 확보하기 어려움
  - 1~2장의 입력으로도 전체 3D 장면을 일관되게 편집하고 싶음

- **Multi-view consistency**
  - diffusion sampling의 stochasticity로 인해
    - 색상, 질감, 구조가 view 간 불일치
  - image-level editing 결과를 그대로 3DGS에 쓰면
    → flickering / geometry distortion 발생

- **Per-scene optimization의 비효율성**
  - 기존 방법 다수는
    - scene마다 finetuning
    - SDS 기반 장시간 최적화 필요
  - 실용성·확장성 낮음

---

### Limitations of Existing Methods
- **Instruct-NeRF2NeRF / EditSplat / GaussCtrl**
  - multi-view image 생성 이후
    → **scene-specific optimization 필수**
- **SDS 기반 3D editing**
  - diffusion prior에 과도하게 의존
  - geometry drift 및 instability
- **FLUX Kontext (기존)**
  - 두 이미지를 concat하면 **pairwise consistency**는 확보
  - reference-based editing 불가
  - image pair를 바꾸면 전역 consistency 붕괴

결과적으로  
**diffusion의 성능은 충분하지만,  
이를 3D-consistent하게 사용하는 파이프라인이 부재**한 상태.

---

## 2. Mechanism

TINKER는 **two-component framework**로 구성됨.

![Overall pipeline of TINKER](../../docs/assets/papers/tinker/fig3_pipeline.png)

*Source: TINKER (Zhao et al., 2025), Fig. 3*

핵심 구성:
1. **Referring Multi-View Consistent Image Editing**
2. **Sparse-to-Dense Scene Completion (Depth-guided Video Diffusion)**

> 핵심 철학  
> **3D representation은 3DGS가 담당**  
> **Diffusion은 image-level supervision과 view propagation만 담당**

---

## 2.1 Referring Multi-View Consistent Image Editing

### 핵심 관찰
- 최신 image editing foundation model(FLUX Kontext)은
  - 두 이미지를 **horizontal concat**하면
    → **local pair consistency**는 매우 강함
- 하지만
  - 서로 다른 pair 간 consistency는 유지되지 않음
  - reference-based editing 능력 부재

![Limitation of vanilla FLUX Kontext](../../docs/assets/papers/tinker/fig2_flux_limitation.png)

*Source: TINKER, Fig. 2*

---

### Dataset Construction (Self-generated)

모델이 **스스로 학습용 dataset을 생성**.

#### Step 1 — Pairwise Editing
- 동일 scene의 두 view \(I_a, I_b\) 선택
- LLM으로 prompt \(P\) 생성
- FLUX Kontext로 editing 수행

```
I'_a, I'_b = E(concat(I_a, I_b), P)
```


#### Step 2 — DINO 기반 Filtering
- **편집 실패 샘플 제거**
  - 원본–결과 feature similarity ↑
- **multi-view 불일치 샘플 제거**
  - 편집된 두 view 간 similarity ↓

→ “잘 편집되었고, view 간 일관된 pair”만 유지

---

### Referring Editing Fine-tuning

최종 학습 입력 구성:
```
Input : concat(I_a, I'_b)
Target : concat(I'_a, I'_b)
```

- I′_b : reference edited view
- I_a  : 아직 편집되지 않은 다른 view

학습 방식:
- **LoRA fine-tuning**
- **Flow Matching (Rectified Flow) loss**

```
Loss = E || E_θ(z_t, t, P) − u(z'_t) ||^2
```


결과:
- reference view를 기준으로
- 다른 view에 **동일한 editing intent를 전파** 가능

![Referring-based multi-view editing](../../docs/assets/papers/tinker/fig4_referring_editing.png)

*Source: TINKER, Fig. 4*

---

## 2.2 Scene Completion — Sparse to Dense View Propagation

Sparse view만으로
view-by-view editing을 반복하는 것은 비현실적.

TINKER는 이를  
**video diffusion 기반 scene completion 문제**로 재정의.

---

### 관점 전환
> **Editing = Reconstruction**

- 원본 scene을 sparse view로부터 복원 가능하다면
- 편집된 sparse view로부터도 scene 복원 가능

---

### Model Design

- Backbone: **WAN2.1 (Video Diffusion Transformer)**
- 조건 입력:
  - **Depth maps** (강한 geometry constraint)
  - **Sparse edited reference views**

![Scene completion architecture](../../docs/assets/papers/tinker/fig5_scene_completion.png)

*Source: TINKER, Fig. 5*

---

### Why Depth Condition?

기존 ray-map / camera-token conditioning:
- geometry 제약 약함
- view inconsistency 발생

Depth의 장점:
- 명시적 geometry constraint
- camera pose 정보 암묵적 포함
- 3D editing에 적합

---

### Training Objective

입력 구성:
```
X_t_input = concat(Z_t, Depth_tokens, Reference_view_tokens)
```

Loss:
```
Loss = E || Φ_θ(X_t_input, t) − u(Z_t) ||^2
```

설계 특징:
- text prompt 제거
- depth를 **reference가 아닌 constraint**로 사용
- reference view와 target frame에 동일 positional embedding 부여

---

## 2.3 One-Shot / Few-Shot 3D Editing Pipeline

![Overall editing workflow](../../docs/assets/papers/tinker/fig3_pipeline.png)

### Few-shot
1. 3DGS에서 sparse views 렌더링
2. Referring editing으로 sparse views 편집
3. Depth + edited views → scene completion
4. 생성된 dense views로 **3DGS 재최적화**

### One-shot
- 단 하나의 edited view로 시작
- 생성된 view를 다시 reference로 사용
- iterative propagation

> **Per-scene finetuning 없이 전체 파이프라인 수행 가능**

---

## 3. 정리

- diffusion은 **scene representation이 아님**
- diffusion은
  - multi-view consistent image editing
  - sparse-to-dense view propagation
  역할만 수행
- 3D structure는
  - 오직 **3DGS optimization**으로 형성

**TINKER의 본질**
> “Diffusion을 3D generator로 쓰지 않고,  
> 3DGS가 따라야 할 image distribution을 안정적으로 제공”

---

## 4. Experimental Results

- One-shot / Few-shot 모두
  - 기존 SOTA 대비
    - 높은 CLIP-dir
    - 높은 DINO consistency
- **per-scene optimization 제거**
- single 24GB GPU에서도 실행 가능

---

## 5. Limitations

- **Diffusion prior 의존**
  - hallucination 완전 제거 불가
- **Large geometry deformation 불가**
  - depth-constrained reconstruction 특성
- **Self-generated dataset 한계**
  - fine detail inconsistency 가능


