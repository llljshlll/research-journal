# Skyfall-GS: Synthesizing Immersive 3D Urban Scenes from Satellite Imagery

GS 와 diffusion 의 조합 논문  
우리와 같이 diffusion 모델을 쓰면서 GS로 뷰를 재구성하기 위해 논문 읽음  

---

## 1. Motivation

### Problem
- **위성영상 → 3D city reconstruction**
  - building의 측면 구조가 거의 관측되지 않음
  - **seasonal / illumination differences** 로 인해 color와 lighting이 일관되지 않음

### Limitations of Existing Methods
- **Sat-NeRF**
  - geometry가 전반적으로 **blurred**
  - **facades**가 뭉개지거나 왜곡됨
- **CityDreamer / GaussianCity**
  - **semantic map + height map** 기반 접근
  - 구조는 안정적이지만 texture가 **synthetic**
  - geometry 표현이 과도하게 단순화됨
- **NeRF / 3DGS-based methods**
  - **satellite view → ground-level / oblique view** 로의 generalization에 실패

### Key Idea of Skyfall-GS
- **Two-stage framework**
  - **Stage 1 — Initial Reconstruction (3DGS)**
    - satellite images를 입력으로 **coarse 3D Gaussian scene**을 초기화
  - **Stage 2 — Iterative Dataset Update (IDU)**
    - 현재 3DGS로부터 images를 render
    - diffusion model을 사용해 renderings를 **higher-quality images로 refinement**
    - refined images를 supervision으로 사용해 3DGS를 재학습
    - 위 과정을 **iteratively 반복**

diffusion model은 **scene representation이 아니며, image-level supervisor**로만 사용됨

---

## 2. Mechanism

Skyfall-GS는 **satellite imagery 기반 3D city reconstruction**을 위해  
 **two-stage framework**로 구성
  
  
**3D scene representation은 3DGS가 담당하고,  
diffusion model은 image-level supervision만 제공**


핵심은 다음 두 가지 구성 요소로 이뤄짐
### Stage 1 — Initial Reconstruction (3DGS-based)
- satellite images를 입력으로 **initial 3D Gaussian scene**을 생성
- photometric supervision을 통해
  - roof surfaces
  - ground planes
  중심의 **coarse geometry**를 복원
- 이 단계에서 생성된 3DGS는
  - ground-level view에 대해서는 geometry가 불완전하며
  - facades와 occluded regions는 거의 복원되지 않음

### Stage 2 — Synthesis via Iterative Dataset Update (IDU)
- 현재 3DGS를 다양한 **camera angles**에서 rendering
- rendered images를 입력으로
  - **diffusion model**이 image-level refinement 수행
  - artifacts 제거 및 texture realism 향상
- diffusion으로 생성된 **refined images**를
  - 새로운 supervision data로 사용하여
  - 3DGS를 다시 학습
- 위 과정을 **Iterative Dataset Update (IDU)** 방식으로 반복

이 단계에서 diffusion model은
- 3D geometry를 직접 생성하거나 수정하지 않으며
- 오직 **“3DGS가 학습해야 할 더 나은 target images”**를 제공하는 역할만 수행한다.

---

## 2.1 Stage 1 — Initial Reconstruction (3DGS-based)
Stage 1의 목표는  
satellite imagery만을 사용해 **coarse 3D Gaussian scene**을 초기화하는 것이다.  
이 단계에서 생성되는 3DGS는 완성된 geometry가 아니라,  
Stage 2에서 refinement를 진행하기 위한 **초기 구조(anchor)** 역할을 한다.

Satellite imagery의 특성상  
parallax가 매우 작고 appearance variation이 크기 때문에,  
단순한 photometric optimization만으로는 안정적인 geometry를 얻기 어렵다.  
이를 보완하기 위해 Skyfall-GS는 Stage 1에서  
세 가지 핵심 regularization 및 supervision을 함께 사용한다.

---

### 2.1.1 Appearance Modeling

Satellite imagery는 서로 다른 **date, season, time of day**에서 촬영되기 때문에,  
동일한 구조물이라도 image마다 **illumination, color tone, shadow pattern**이 크게 다르다.  
이러한 appearance variation은 3DGS가 geometry를 학습하는 과정에서  
불필요한 혼란을 유발하며, geometry 수렴을 불안정하게 만든다.

Appearance Modeling의 목적은 Satellite imagery만으로 최대한 안정적인 3DGS 초기화

---

#### Core Idea
- **illumination / date difference ≠ geometry**
- appearance variation을 **latent space**로 분리
- 3DGS는 geometry와 material 구조에 집중하도록 유도


#### Implementation

Skyfall-GS는 **WildGaussians 스타일**의 appearance modeling을 사용하여,  
다음 세 가지 정보를 MLP에 입력하고  
**affine color transform parameters**를 예측함


##### (1) Per-image embedding
- 각 satellite image \( j \)마다 하나의 embedding을 학습
- 해당 이미지의 전역적인 appearance 특성을 표현
  - 예:  
    *“오후에 촬영됨”*  
    *“겨울 계절”*  
    *“전체적으로 노란 색감, 그림자가 김”*

→ image-level illumination / seasonal condition을 latent로 인코딩


##### (2) Per-Gaussian embedding
- 각 Gaussian \( i \)마다 하나의 embedding을 학습
- 특정 공간 위치에서 반복적으로 나타나는 **지역적 appearance variation**을 표현
  - 예:  
    *“이 Gaussian은 항상 그림자가 걸리는 위치”*  
    *“나무나 녹지 영역이라 계절에 따라 색이 크게 변함”*

→ local appearance bias를 geometry와 분리하여 저장


##### (3) Base color (SH DC component)
- 각 Gaussian의 **0th-order Spherical Harmonics**를 base color로 사용
- view-dependent effect가 제거된, 가장 기본적인 color representation


#### Affine Color Transform

위 세 가지 입력을 MLP \( f \)에 전달하여  
**affine color transform parameters**를 예측한다.

- **Scale (γ)**  
  - saturation / contrast 성분 조절
- **Bias (β)**  
  - brightness 및 color shift 성분 조절

최종 color는 다음과 같이 계산된다.
'''
c_final = γ · c_view-dependent + β
'''
여기서
- `c_view-dependent`는  
  기존 3DGS가 Spherical Harmonics를 통해 계산한 color
- `(γ, β)`는  
  illumination / date variation을 보정하는 역할만 수행

#### Effect

- illumination 및 seasonal variation이 geometry 학습에 미치는 영향을 제거
- 동일한 구조물이 image마다 다른 색으로 보이더라도
  geometry는 일관되게 유지됨
- Stage 1 optimization의 **stability와 convergence**를 크게 향상
  
> **Appearance Modeling은**  
> geometry와 appearance를 분리하여  
> satellite imagery의 복잡한 촬영 조건에서도  
> 안정적인 초기 3D Gaussian scene을 형성하는 핵심 모듈

---

### 2.1.2 Opacity Regularization

Satellite imagery는 지상에서 수백 km 떨어진 위치에서 촬영되기 때문에,  
camera position이 변해도 **object 간 relative position 변화**가 거의 발생하지 않는다.  
즉, 3D reconstruction에서 핵심적인 **depth cue를 parallax로부터 얻기 어려운 환경**이다.

이러한 조건에서 3DGS를 photometric loss만으로 학습하면,  
모델은 실제 surface 위치를 정확히 추정하지 못한 채  
**“이미지에 보이는 색을 맞추기 위해” 3D 공간 곳곳에 Gaussian을 배치**하게 된다.

---

#### Floating Gaussians (Floaters)

이 과정에서 발생하는 **floaters**는  
실제 surface와 무관한 위치에 존재하는 Gaussian들로,  
photometric loss를 일시적으로 만족시키기 위해 생성되는 **geometry noise**이다.

이러한 Gaussian들의 특징은 다음과 같다.

- 실제 표면이 아님
- 3D 공간에서 공중에 떠다니는 형태
- **낮은 opacity (α)** 를 가짐  
  - 완전히 불투명하지도
  - 완전히 투명하지도 않은 **반투명 상태**

이는 모델이 해당 Gaussian을  
“진짜 표면”이 아니라  
“색을 조금만 기여시키는 임시 요소”로 사용하기 때문이다.


#### Binary Entropy Regularization on Opacity

Skyfall-GS는 이러한 floaters를 제거하기 위해  
Gaussian의 **opacity (α)** 에 대해  
**binary entropy regularization**을 적용한다.

- entropy \( H(\alpha) \) 는  
  - \( \alpha = 0.5 \) 에서 최대
  - \( \alpha \to 0 \) 또는 \( \alpha \to 1 \) 로 갈수록 감소
- Skyfall-GS는  
  **−H(α)** 를 loss로 사용하여 이를 minimize

그 결과,

- \( \alpha \approx 0.2, 0.3, 0.4 \) 와 같은  
  **중간 opacity Gaussian**은 강하게 penalize됨
- Gaussian들은 학습 과정에서 자연스럽게
  - **α ≈ 1** : 실제 surface
  - **α ≈ 0** : 제거 대상  
  둘 중 하나로 정렬됨

#### Effect

Opacity Regularization을 통해

- floating Gaussians가 자동으로 제거되고
- 의미 없는 geometry noise가 감소하며
- building edge와 facade와 같은 구조가 더 선명해진다.

논문에서는 ablation study를 통해  
Opacity Regularization이  
**floating artifacts를 줄이고 geometry clarity를 향상시킨다**는 점을 보고한다.

> **Opacity Regularization은**  
> low-parallax satellite setting에서 발생하는  
> geometry ambiguity를 정리하고,  
> 3D Gaussian scene을 구조적으로 정제하는 핵심 안정화 기법이다.

---

### 2.1.3 Pseudo-camera Depth Supervision

Satellite imagery는 카메라가 지상으로부터 매우 멀리 위치하기 때문에,  
view 변화에 따른 **effective parallax**가 거의 발생하지 않는다.  
이로 인해 pure photometric supervision만으로는  
roof, road, facade와 같은 구조의 **depth ordering**을 안정적으로 고정하기 어렵다.

이를 보완하기 위해 Skyfall-GS는  
**pseudo-camera depth supervision**을 도입하여  
3DGS에 추가적인 **depth shape prior**를 제공한다.

---

#### Core Idea

- 실제 ground-level images는 존재하지 않지만
- 현재 3DGS로부터 **지상 근처에서 본 것처럼 렌더링한 pseudo views**는 생성 가능
- 해당 pseudo views에 대해
  **monocular depth model**이 예측한 depth는
  자연 이미지 분포에 기반한 **합리적인 depth structure prior**로 활용 가능

즉, **3DGS → RGB rendering → monocular depth model → depth prior → GS optimization 의 흐름으로 depth supervision**을 구성

#### (1) Pseudo-camera Rendering

- scene 주변에 **ground-level에 가까운 pseudo cameras**를 랜덤하게 샘플링
- 각 pseudo camera에 대해
  - 현재 3DGS로부터 **RGB image**와 **depth map**을 렌더링
- 이 depth는
  - 현재 GS가 추정한 geometry에 기반한 depth

#### (2) Monocular Depth Prediction (MoGe)

- 렌더링된 RGB image를
  **monocular depth model (MoGe)**에 입력
- MoGe는 단일 RGB image만을 사용하여
  - pixel-wise depth ordering
  - 자연스러운 depth gradient
  를 예측

이 결과는
- absolute scale은 정확하지 않지만
- **relative depth structure**는 신뢰할 수 있는
  **pseudo ground-truth depth**로 사용된다.

#### (3) Depth Correlation Loss

- 3DGS가 렌더링한 depth와
- MoGe가 예측한 depth 간의
  **scale-invariant correlation loss (Pearson correlation)**를 계산

이를 통해
- 절대 depth 값이 아닌
- **depth shape와 ordering**만을 supervision으로 사용

즉, camera scale이나 정확한 거리 정보 없이도  
geometry가 합리적인 3D 형태를 따르도록 유도한다.


#### Effect

Pseudo-camera Depth Supervision을 통해

- roof와 ground plane의 뒤틀림이 감소하고
- facade와 같은 수직 구조의 depth ordering이 안정화되며
- Stage 1에서 생성되는 3DGS가
  Stage 2 refinement를 위한 **geometry anchor**로서 충분한 품질을 갖게 된다.

> **Pseudo-camera Depth Supervision은**  
> satellite imagery로는 직접 관측할 수 없는  
> depth 구조를 외부 monocular prior로 보완하여,  
> 초기 3D Gaussian scene의 기하적 일관성을 강화하는 장치이다.


## 2.2 Stage 2 — Synthesis via Iterative Dataset Update (IDU)

Stage 2의 목표는  
Stage 1에서 생성된 **coarse 3D Gaussian scene**을 기반으로,  
satellite imagery만으로는 복원하기 어려운 **facades와 occluded regions**를  
점진적으로 보완하는 것이다.

이 단계에서는 3DGS를 한 번에 완성하려 하지 않고,  
현재 3DGS로부터 생성한 renderings를  
**diffusion model을 통해 더 높은 품질의 이미지로 refinement**한 뒤,  
이를 다시 3DGS의 학습 데이터로 사용하는 **iterative optimization**을 수행한다.

또한 모든 camera angle을 동시에 다루지 않고,  
**top-down view에서 ground-level view로 점진적으로 이동하는 curriculum**을 적용하여  
geometry ambiguity가 큰 view에서도 안정적인 refinement가 가능하도록 설계한다.


### 2.2.1 Curriculum Learning for Camera Angles

Skyfall-GS는 동일한 3DGS 결과라도  
**camera elevation에 따라 view quality가 크게 달라진다**는 현상을 관찰한다.

- **High elevation views (satellite-like views)**  
  - roof, road와 같은 top-facing surfaces 위주로 관측됨  
  - 입력 satellite imagery와 유사한 시점  
  - 3DGS가 비교적 안정적으로 맞춤
- **Low elevation views (ground-level / oblique views)**  
  - building facade, vertical surfaces, occluded regions가 드러남  
  - satellite view에서 거의 관측되지 않았던 영역  
  - geometry가 틀어지거나 뭉개지는 현상 발생

이러한 특성으로 인해,  
모든 camera angle을 동시에 다루는 방식은  
diffusion hallucination과 geometry inconsistency를 유발하기 쉽다.

---

#### Curriculum Strategy

Skyfall-GS는 이를 해결하기 위해  
**camera elevation을 점진적으로 낮추는 curriculum learning strategy**를 사용한다.

- 초기 episode  
  - **높은 elevation + 큰 radius**  
  - 위에서 내려다보는 **안전한 view** 위주로 렌더링
- 이후 episode로 갈수록  
  - camera elevation을 점차 낮추고  
  - 지상에 가까운 view를 점진적으로 포함

이 과정에서  
이전 episode에서 diffusion refinement로 **이미 안정화된 3DGS**가  
다음 episode의 더 어려운 view를 **구조적으로 제약**하게 된다.


#### Camera Setup in IDU

IDU 단계에서 camera는 다음과 같이 구성된다.

- **N_p look-at points**
  - scene 전체에 균일하게 분포된 target points
  - 각 카메라가 바라보는 중심점 역할
- 각 look-at point 주변에
  - **circular orbit** 형태로 camera를 배치
  - 서로 다른 elevation과 radius에서 scene을 렌더링

이를 통해
- 다양한 위치와 각도에서의 view를 확보하면서도
- curriculum에 따라 **view difficulty를 제어**할 수 있다.


#### Effect

Camera curriculum을 통해

- 쉬운 view에서 먼저 geometry를 안정화하고
- 점진적으로 occluded regions를 드러내며
- ground-level view에서도 일관된 geometry를 유지할 수 있다.

> **Curriculum Learning for Camera Angles는**  
> diffusion-guided refinement가  
> 점진적으로 geometry를 확장하도록 만드는  
> Stage 2의 핵심 안정화 전략이다.

---

### 2.2.2 Diffusion Refinement (FlowEdit + FLUX.1)

Stage 2에서 diffusion model은  
3D geometry를 직접 생성하는 역할이 아니라,  
**현재 3DGS가 만든 renderings를 더 높은 품질의 이미지로 보정(refinement)**하는  
image-level supervisor로 사용된다.

Skyfall-GS는 diffusion model로 **FLUX.1**을 사용하고,  
이미지 편집을 위해 **FlowEdit** 프레임워크를 적용한다.

---

#### Core Idea

3DGS로부터 얻은 rendered image를  
**diffusion denoising 과정의 중간 단계에 해당하는 noisy sample**로 해석하고,  
diffusion model이 **남은 denoising steps**를 수행하도록 한다.

즉,

- 3DGS rendering  
  → 완전한 noise는 아니지만 artifact와 불완전한 texture를 포함한 이미지
- Diffusion refinement  
  → 해당 이미지를 출발점으로 삼아
    자연 이미지 분포에 맞도록 추가 denoising 수행
- 결과  
  → 보다 **sharp하고 realistic한 image**

이렇게 생성된 refined image는  
다시 3DGS의 supervision data로 사용된다.


#### FlowEdit + FLUX.1

- **FLUX.1**
  - text-conditioned diffusion model
  - 고해상도 satellite-like image distribution을 학습한 prior 제공
- **FlowEdit**
  - 기존 diffusion 모델의 구조를 변경하지 않고
  - 입력 이미지를 denoising trajectory 중간에 삽입하여
    image-to-image refinement를 수행하는 방식

이를 통해
- 3DGS renderings의 구조는 유지하면서
- texture, edge, illumination artifact만 선택적으로 보정할 수 있음


#### Self-Bootstrapping Loop

Diffusion refinement는 단일 단계로 끝나지 않고,  
**Iterative Dataset Update (IDU)** 과정 안에서 반복된다.
'''
coarse 3DGS → render → diffusion refinement → refined images → GS retraining → improved 3DGS
'''

즉,
- Stage 1에서 대략적인 geometry 뼈대를 만든 뒤
- diffusion이 이를 **“깨끗한 satellite-like image”로 재작성(rewriting)**하고
- 3DGS가 그 결과를 다시 학습하는
  **self-bootstrapping loop**가 형성된다.


#### Role of Diffusion

이 과정에서 diffusion model은

- **scene representation이 아니며**
- 3D geometry를 직접 수정하지 않고
- 오직 **더 나은 target images를 제공하는 역할**만 수행한다.

> **Diffusion Refinement는**  
> 3DGS가 자연 이미지 분포에 맞는 appearance와 구조를 학습하도록 돕는  
> Stage 2의 핵심 image-level supervision 메커니즘이다.

---

### 2.2.3 Multi-sample Diffusion

Diffusion refinement를 **한 번만 수행할 경우**,  
서로 다른 camera view 간 **visual consistency가 쉽게 깨지는 문제**가 발생한다.  
이는 diffusion의 **stochastic denoising process** 특성상,  
각 view가 서로 다른 denoising trajectory를 따르기 때문이다.

또한 3DGS 자체도  
단일 view supervision에 과도하게 맞춰질 경우  
**single-view overfitting**이 발생하기 쉽다.  
이 경우 Gaussians가 특정 view에 맞게 왜곡되고,  
novel view에서는 artifacts가 나타난다.

---

#### Motivation

이론적으로는  
“모든 view에서 일관된 결과를 만드는”  
**이상적인 denoising trajectory 분포**가 존재할 수 있다.  
그러나 각 view에 대해 diffusion을 한 번씩만 실행하면,  
매우 큰 trajectory space 안에서  
그러한 “좋은 경로”를 **우연히 선택할 확률은 극히 낮다**.

즉,
- single-sample diffusion  
  → view-specific hallucination이 그대로 supervision으로 전달됨
- 결과  
  → 3DGS가 잘못된 구조를 geometry로 고정할 위험 증가


#### Multi-sample Strategy

이를 완화하기 위해 Skyfall-GS는  
각 camera view에 대해 diffusion refinement를  
**multiple times (Ns samples)** 수행한다.

- 각 sample은
  - 동일한 rendered image를 입력으로 사용하지만
  - 서로 다른 random seed로 denoising trajectory를 샘플링
- 논문에서는 **Ns = 4**를 사용

이렇게 생성된 multiple refined images는  
모두 3DGS 학습에 supervision으로 사용된다.


#### Effect

Multi-sample Diffusion을 통해

- diffusion의 stochastic hallucination이 평균화되고
- 여러 sample에서 **공통적으로 유지되는 구조만** 3DGS에 반영되며
- single-view overfitting이 완화된다.

결과적으로 3DGS는  
특정 view에만 맞는 해가 아니라,  
**multi-view consensus를 만족하는 geometry**를 학습하게 된다.

> **Multi-sample Diffusion은**  
> diffusion refinement를 안정적인 supervision으로 변환하여,  
> view-consistent 3D Gaussian scene을 형성하는 핵심 장치이다.

---

## 3. Training

Skyfall-GS의 training은  
**Stage 1 — Initial Reconstruction**과  
**Stage 2 — Iterative Dataset Update (IDU)**의 두 단계로 나뉘어 진행된다.  
두 단계는 서로 다른 supervision과 loss 구성을 사용하며,  
Stage 2는 Stage 1에서 학습된 3DGS를 초기 상태로 이어받아 수행된다.

---

### Stage 1 Training

Stage 1에서는 satellite imagery만을 사용하여  
**coarse 3D Gaussian scene**을 안정적으로 초기화하는 것이 목표이다.

#### Supervision
- Satellite RGB images
- Pseudo-camera depth supervision (monocular depth prior)

#### Losses
- **Photometric loss**  
  - rendered RGB와 satellite image 간의 color reconstruction loss
- **Depth correlation loss**  
  - 3DGS-rendered depth와 monocular depth prediction 간의  
    scale-invariant correlation loss (Pearson correlation)
- **Opacity regularization**  
  - Gaussian opacity에 대한 binary entropy regularization

#### Optimization Goal
- appearance variation과 geometry ambiguity를 완화
- Stage 2에서 refinement가 가능한 **stable geometry anchor** 형성

---

### Stage 2 Training (IDU)

Stage 2에서는  
Stage 1에서 생성된 3DGS를 기반으로  
**dataset 자체를 반복적으로 갱신**하며 학습을 진행한다.

#### Training Procedure
- 각 episode마다
  - 현재 3DGS로부터 camera curriculum에 따른 view를 렌더링
  - diffusion model을 이용해 renderings를 refinement
  - refined images를 새로운 supervision data로 사용
- 기존 satellite images는 사용하지 않고,
  **diffusion-refined images로 dataset을 대체(update)**

#### Supervision
- Diffusion-refined images (multi-sample, Ns = 4)

#### Losses
- **Photometric loss**  
  - rendered RGB와 diffusion-refined image 간의 color reconstruction loss
- (Stage 1에서 사용한 depth supervision과 opacity regularization은  
  Stage 2에서는 적용되지 않음)

#### Optimization Goal
- curriculum에 따라 점진적으로 어려운 view를 포함
- multi-view consensus를 만족하는 geometry로 수렴

---

### Optimization Strategy

- 3DGS parameters
  - position, scale, rotation
  - opacity
  - SH color coefficients
- Diffusion model (FLUX.1)
  - **fully frozen**
  - training에는 관여하지 않음
- IDU 과정은
  - geometry가 안정화될 때까지
  - 여러 episode에 걸쳐 반복 수행됨

---

## 4. Experimental Results

---

## 5. Limitations

Skyfall-GS는 satellite imagery 기반 3D reconstruction에서  
강력한 성능을 보이지만, 몇 가지 한계를 가진다.

- Diffusion refinement로 인해
  **hallucinated geometry**가 포함될 수 있으며,
  이는 정확한 측정 목적에는 부적합하다.
- IDU 과정은
  - multiple rendering
  - multi-sample diffusion
  을 포함하므로 **computational cost**가 크다.
- ground-level view는
  실제 street-level imagery가 아닌
  diffusion prior에 의존하므로
  **완전한 현실성과는 차이**가 존재한다.

따라서 Skyfall-GS는
정밀한 metric reconstruction보다는,
**large-scale visualization 및 city-level scene understanding**에
더 적합한 접근으로 볼 수 있다.








