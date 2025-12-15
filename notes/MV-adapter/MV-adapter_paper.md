# MV-Adapter: Multi-view Consistent Image Generation Made Easy

---

## 1. Motivation
기존 multi-view diffusion 모델들은 대부분 **입력 이미지로부터 3D geometry를 추론**하고,  
그 결과로 multi-view 이미지를 생성하는 **bottom-up pipeline**을 사용

**MV-Adapter**는 **mesh → image 방향(top-down**) 접근을 취하며,  
**reference image와 명시적인 3D geometry condition (position, normal**)을 함께 사용하여  
multi-view 전반에서 **reference image의 시각적 일관성(visual consistency**)을 유지  

---

## 2. Mechanism

MV-Adapter는 기존 **Stable Diffusion (SD2.1 / SDXL)** 구조를 변경하지 않고,  
**multi-view consistency**를 학습하기 위한 모듈(adapter)만 추가하는 방식으로 동작함 
<img src="images/MV-adapter/insert.png" alt="Decoupled Attention Layers" width=600> 

핵심은 다음 두 가지 구성 요소로 이뤄짐

1. **Condition Guider** – position/geometry condition을 인코딩하여 UNet 내부에 주입  
2. **Decoupled Attention Layers** – 기존 attention 구조를 병렬화하여  
   multi-view, image, text 정보를 동시에 처리
<img src="images/MV-adapter/pipeline.png" alt="Decoupled Attention Layers" width=600> 

---
### 2.1 Condition Guider

Condition Guider는 각 view에서 렌더링된 **Position Map + Normal Map**을  
geometry-conditioned feature로 변환하고,  
**U-Net Down Block 내부의 모든 ResNet Block 출력에 직접 더해주는 (Residual Add)** 모듈  
각 view의 구조적 정보를 강하게 보존하면서 multi-view 이미지 생성을 지원해주는 역할을 함

<img src="images/MV-adapter/UNet.png" alt="Decoupled Attention Layers"> 

#### 입력
- **Position Map (3채널)**  
  - mesh 표면의 world-space 좌표 (X, Y, Z)
- **Normal Map (3채널)**  
  - 표면 방향 벡터 (nx, ny, nz)

→ concat하여 **(6, H, W)**  
→ 6 views에 대해 batch로 구성 → **(6, 6, H, W)**

> 코드에서는 mesh 기반으로 Position + Normal을 직접 렌더링하여 사용함



#### 처리 방식 & 역할
- concat된 geometry condition을  
  **Conv → Norm → Activation**으로 feature 변환  
- Down Block 내부 **모든 ResNet Block** 출력에 직접 Add (Residual Injection)   
```
Feature_out = Feature_out + ConditionFeature
```
- U-Net backbone의 feature 흐름 전체에 geometry 정보를 지속적으로 통합  
- View별 구조(grid alignment, pose consistency) 유지에 핵심적 역할

> **Condition Guider = Geometry-aware residual injection layer**  
> U-Net Downsample 영역 전체에 geometry 정보를 강하게 주입하여  
> 각 view 내부의 구조적 일관성을 보장하는 핵심 모듈

---


### 2.2 Decoupled Attention Layers

MV-Adapter의 핵심은 **기존 self-attention 구조를 duplicate하고 parallelize** 하는 것

#### 구성 요소
<img src="images/MV-adapter/pipeline.png" alt="Decoupled Attention Layers" width=600> 

| Type | Query | Key/Value | 목적 |
|------|--------|------------|------|
| Spatial Self-Attn | 현재 view latent | Text embedding | 현재 view feature |
| Multi-view Attn | 현재 view latent | 다른 view latent | 3D 구조 일관성 |
| Image Cross-Attn | 현재 view latent | Reference image latent | 시각적 스타일 일관성 |
| Text Cross-Attn | (Self + Multi-View + Image Cross)의 합산 feature | CLIP text embedding | 의미 반영 |  

  
Spatial Self-Attention, Text Cross-Attention은 기존 stable diffusion UNet의 기본 블록이며,  
Multi-View Attention, Image Cross-Attention은 Spatial Self-Attention 블록을 복제한 구조의 블록  


#### Parallel Architecture

기존 Stable Diffusion의 attention은 **serial residual connection** 구조로 되어 있지만,  
MV-Adapter는 이를 **parallel residual structure**로 변경  

<img src="images/MV-adapter/sd_layer.png" alt="Decoupled Attention Layers" width=600>


- 기존 Stable Diffusion의 UNet에서 나온 latent feature를 모든 attention 블록(Self, Multi-view, Image Cross, Text Cross)이 동시에 공유하므로,  
  기존 pretrained weight 그대로 활용 가능  
- 학습시, 추가한 attention layer의 **output projection을 0으로 초기화(zero-init)** →  
  controlNet 학습처럼 초기에 기존 모델의 feature space를 전혀 교란하지 않음  
- Multi-view와 Image cross-attention이 각각 Multi-View Attention, Image Cross-Attention에서 학습되면서  
  점진적으로 geometry-aware consistency를 학습함

> 이 병렬 구조는 pretrained prior를 유지하면서  
> **새로운 3D geometric prior**를 효율적으로 흡수할 수 있도록 설계됨


---

### 2.3 Training

학습 시에는 **기존 Stable Diffusion과 동일한 noise prediction loss (ε-MSE**)를 사용하되,  
MV-Adapter의 파라미터만 업데이트함

---

### 2.4 Inference Pipeline

추론 시에는 다음 순서로 작동  
<img src="images/MV-adapter/pipeline.png" alt="Decoupled Attention Layers" width=600>
1. Text prompt, reference image, camera/geometry map 입력  
- image reference
   - vae로 latent 생성
   - time step을 0으로 설정(noise없는 상태)해서 기존 Unet에 넣음
2. Condition Guider가 각 condition을 feature map으로 변환  
3. Decoupled Attention Layers가 multi-view / image / text 정보를 병렬로 융합  
4. 각 시점(view)에 대한 latent representation을 동시에 업데이트  
5. VAE decoder를 통해 multi-view 이미지를 복원

---

## 3. inference
pretraiened model에 대해 치아 도메인 실험

| View (Azimuth) | Normal | Position | Inference |
|---|---|---|---|
| Front (0°) | ![normal-front](images/mv/front/normal.png) | ![position-front](images/mv/front/position.png) | ![infer-front](images/mv/front/infer.png) |
| Right (90°) | ![normal-fr](images/mv/front_right/normal.png) | ![position-fr](images/mv/front_right/position.png) | ![infer-fr](images/mv/front_right/infer.png) |
| Left (270°) | ![normal-right](images/mv/right/normal.png) | ![position-right](images/mv/right/position.png) | ![infer-right](images/mv/right/infer.png) |
| up (180°) | ![normal-back](images/mv/back/normal.png) | ![position-back](images/mv/back/position.png) | ![infer-back](images/mv/back/infer.png) |
| down (270°) | ![normal-left](images/mv/left/normal.png) | ![position-left](images/mv/left/position.png) | ![infer-left](images/mv/left/infer.png) |
| back (315°) | ![normal-fl](images/mv/front_left/normal.png) | ![position-fl](images/mv/front_left/position.png) | ![infer-fl](images/mv/front_left/infer.png) |

open surface라서 상, 하, 후면의 inference가 이상하게 나옴
=> mesh의 뒷면을 채움(./mesh_closed 링크)

