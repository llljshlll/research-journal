# Stable Diffusion
: **Latent Diffusion Model (LDM)** 계열의 조건부 생성 모델로,  
고해상도 이미지 생성을 pixel space가 아닌 latent space에서 수행하도록 설계된  
효율 중심의 generative diffusion 구조  
  
본 문서는 Stable Diffusion의 내부 구조와 동작 방식을  
**Architecture / Diffusion Process / Conditioning / Training Objective** 관점에서 정리한다.  
Transformer, Attention 등 범용 개념은 별도의 `concepts/` 문서에서 다룬다.


## 1. Overview
Stable Diffusion의 전체 파이프라인은 다음과 같이 요약할 수 있다.
<img src="../../../docs/assets/models/flux/stalbe_diffusion_architecture.png">  

1. 텍스트 프롬프트를 **Text Encoder (CLIP)** 를 통해 임베딩  
2. 이미지를 **VAE Encoder**를 통해 latent space로 압축  
3. **Denoising UNet**이 latent space에서 noise를 예측하며 이미지 점진적 복원  
4. **Sampler**가 시간 축에서 noise 제거 과정을 제어  
5. 최종 latent를 **VAE Decoder**를 통해 pixel image로 복원  
  
핵심 설계 철학  
- 고차원 pixel space 대신 **저차원 latent space에서 diffusion 수행**  
- 텍스트 조건을 **UNet 내부 cross-attention으로 직접 주입**  

## 2. Architecture
### 2.1 Latent Space & VAE (SD-specific)

Stable Diffusion은 diffusion을 **pixel space가 아닌 latent space**에서 수행.  
이를 위해 **Variational Autoencoder (VAE)** 가 사용됨.  
<img src="../../../docs/assets/models/flux/stalbe_diffusion_VAE.png">   

- **Encoder**
  - 입력 이미지 x (shape: 3 × H × W)
  - latent z (shape: 4 × H/8 × W/8) 로 압축
- **Decoder**
  - diffusion이 완료된 latent를 pixel image로 복원

이 설계로 인해:  
- 연산량의 대폭 감소  
- 고해상도 이미지에 대한 diffusion 가능  
- diffusion model이 전역적인 의미 구조(manifold)에 집중 가능  
  
Stable Diffusion에서 사용되는 VAE는   
입력을 하나의 고정된 latent 코드가 아닌  
**확률 분포 기반의 latent 표현**으로 매핑하도록 학습된다.  
  
이를 통해 latent space 전반에서 의미 있는 표현을 유지하며,  
diffusion 과정에 적합한 연속적 잠재 공간을 형성한다.  
  
> VAE의 확률적 latent space 특성 및 AutoEncoder와의 구조적 차이에 대한 자세한 설명은  
> [concepts/vae.md](../../concepts/vae.md) 참고.  
  
---

### 2.2 Denoising UNet
<img src="../../../docs/assets/models/flux/stalbe_diffusion_UNet.png"> 
Stable Diffusion의 Denoising UNet은  
timestep과 텍스트 조건을 함께 입력으로 받아  
latent space에서 noise를 예측하는 조건부 생성 모델.  
  
입력과 출력은 다음과 같다.

- 입력
  - noisy latent z_t (shape: 4 × H × W)
  - timestep t (time embedding)
  - text embedding (CLIP output, optional)
- 출력
  - 현재 timestep에서의 noise 예측값 ε̂(z_t, t)

UNet은 이미지를 직접 생성하지 않으며,  
각 timestep에서 **제거되어야 할 noise의 방향과 크기**를 예측하는 역할만 수행한다.

#### 전체 구조 개요

UNet은 Encoder–Middle–Decoder 구조를 기반으로 하며,  
downsampling과 upsampling을 통해 다중 해상도에서 latent feature를 처리
  <img src="../../../docs/assets/models/flux/UNet_architecture.png"> 
- Encoder (Down blocks)  
  - 해상도를 점진적으로 감소  
  - 국소적 세부 정보 → 점차 전역적 구조 추출  
- Middle block  
  - 가장 낮은 해상도  
  - 전역 의미 정보 집중 처리  
- Decoder (Up blocks)  
  - 해상도를 점진적으로 복원  
  - Encoder의 feature와 skip connection으로 결합  


#### 핵심 구성 블록  

##### 1 - ResBlock (Time-embedding conditioned)  
ResBlock은 Stable Diffusion UNet의 기본 계산 단위로,  
timestep 정보를 feature에 직접 결합하지 않고  
MLP를 통해 변환된 **channel-wise bias 형태로 주입하는  
time-conditioned residual block 구조**.

입력 feature는 convolution 경로를 따라 처리되며,  
timestep embedding은 SiLU와 Linear layer를 거쳐  
출력 channel 차원으로 투영된 뒤 feature map에 additive bias로 적용된다.

이를 통해 UNet은 각 timestep에 따라  
동일한 구조의 convolution 연산을 서로 다른 방식으로 조절하며,  
diffusion 과정의 시간적 위치를 명시적으로 반영한다.

<img src="../../../docs/assets/models/flux/UNet_ResBlock.png"> 
ResBlock 내부 동작은 다음과 같다.

- 입력 feature
  - GroupNorm → SiLU → 3×3 Conv 경로를 따라 처리
- Timestep embedding
  - 1280-d time embedding
  - SiLU → Linear(1280 → c_out)
  - (c_out × 1 × 1) 형태로 reshape
- 결합 방식
  - 변환된 time embedding을 feature map에 additive bias로 적용
- Residual connection
  - 입력과 출력 channel이 다를 경우 1×1 Conv
  - 동일할 경우 identity shortcut


##### 2 - Attention / Cross-Attention Block
Attention Block은 UNet 내부에서  
latent feature와 조건 정보를 결합하는 핵심 모듈로,  
Self-Attention과 Cross-Attention을 순차적으로 수행하는  
transformer-style block 구조.

입력 latent feature는 공간 차원을 펼쳐  
(h × w) × c 형태의 시퀀스로 변환된 뒤 처리되며,  
Self-Attention을 통해 공간 위치 간의 전역적 관계를 모델링한다.

이후 Cross-Attention 단계에서  
latent feature는 query로 사용되고,  
텍스트 인코더(CLIP)에서 생성된 text embedding은  
key와 value로 사용되어,  
각 공간 위치가 텍스트 토큰과 선택적으로 대응되도록 조절된다.

Attention 연산 이후에는  
FeedForward 네트워크가 적용되며,  
전체 블록은 residual connection을 통해  
입력 feature와 안정적으로 결합된다.

<img src="../../../docs/assets/models/flux/UNet_AttnBlock.png"> 

Attention Block 내부 처리 흐름은 다음과 같다.

- 입력 feature
  - GroupNorm → 1×1 Conv
  - (c × h × w) → (h × w) × c 로 reshape
- Self-Attention
  - latent feature 간 전역 공간 관계 학습
- Cross-Attention
  - Query: latent feature
  - Key / Value: text embedding (77 × 768)
  - 공간 위치별 텍스트 조건 선택적 반영
- FeedForward
  - channel-wise 비선형 변환
- 출력
  - reshape → 1×1 Conv
  - residual connection을 통해 feature 결합


> Self-Attention 및 Cross-Attention의  
> 내부 연산 구조(QKV, scaling, softmax)에 대한 자세한 설명은  
> [concepts/attention.md](../../concepts/attention.md) 참고.


##### 3 - Conv / Downsample / Upsample Block
Downsample 및 Upsample Block은  
UNet의 다중 해상도 구조를 구성하는 핵심 요소로,  
latent feature의 공간 해상도를 단계적으로 조절하는 역할.

이 과정은 pooling 연산이 아닌  
convolution 및 interpolation 기반으로 수행되어,  
공간 정보의 손실을 최소화하도록 설계된다.


**Conv Block (Spatial size preserved)**
<img src="../../../docs/assets/models/flux/UNet_ConvBlock.png"> 
Conv Block은 공간 해상도를 유지한 채  
channel 차원의 변환을 수행하는 기본 convolution 블록.

- 입력 feature: c_in × h × w
- 3×3 convolution을 통해 channel을 c_out으로 변환
- spatial resolution(h, w)은 유지

블록의 위치에 따라:
- 첫 번째 convolution은 identity shortcut 사용
- 마지막 convolution에는 GroupNorm과 SiLU 활성화가 적용된다.

**Downsample Block**
<img src="../../../docs/assets/models/flux/UNet_DownSampleBlock.png">
Downsample Block은 spatial resolution을 절반으로 줄여  
더 넓은 receptive field에서 feature를 추출하도록 한다.

- 3×3 convolution
- stride = 2, padding = 1
- 출력 feature: c × (h/2) × (w/2)

이를 통해:
- 국소적 디테일은 점진적으로 요약
- 전역 구조 정보는 강조

**Upsample Block**
<img src="../../../docs/assets/models/flux/UNet_UpsampleBlock.png">
Upsample Block은 축소된 spatial resolution을 복원하여  
고해상도 feature 생성을 가능하게 한다.

- Nearest-neighbor interpolation (scale factor = 2)
- 이후 3×3 convolution 적용
- 출력 feature: c × (2h) × (2w)

Interpolation 이후 convolution을 적용함으로써  
단순 복제가 아닌, 학습 가능한 해상도 복원이 이루어진다.


##### 4 - Skip Connection
Skip connection은 Encoder와 Decoder의 동일 해상도 feature를 연결한다.

- downsampling 과정에서 손실된 공간 정보 복원
- 세부 질감과 전역 구조의 결합
- 안정적인 gradient 흐름 유지

Stable Diffusion UNet은 skip connection을 통해 저해상도 의미 정보 + 고해상도 디테일 정보를 동시에 활용

---

### 2.3 Text Encoder (CLIP)

Stable Diffusion은 텍스트 조건 처리를 위해  
CLIP Text Encoder를 사용한다.

#### Text processing pipeline
- Text prompt
  → CLIP Tokenizer
  → Token embedding
  → Text Transformer (CLIP)
  → Text embedding

#### Tokenization
- 최대 토큰 수: 77
- 단어 단위가 아닌 subword + 공백 기반 토큰화
- 입력 프롬프트는 고정 길이 토큰 시퀀스로 변환됨

#### Output representation
- 출력 형태: E_text ∈ R^{77 × 768}
- 각 토큰은 768차원의 의미 벡터로 표현됨
- 문장 전체를 하나의 벡터로 압축하지 않음

#### Role in Stable Diffusion
- CLIP은 이미지를 생성하지 않음
- 텍스트의 **시각적 의미 표현(semantic prior)** 만 제공
- 생성 과정에서:
  - UNet Cross-Attention의 key / value로 사용됨
  - latent feature가 텍스트 토큰을 선택적으로 참조

> CLIP은 조건의 **출처(source)** 이며,  
> 조건이 실제로 반영되는 위치는 UNet 내부이다.

> CLIP의 토큰화 방식, 임베딩 의미 공간,  
> 이미지–텍스트 정렬 학습에 대한 자세한 설명은  
> [concepts/clip.md](../../concepts/clip.md) 참고.


---

### 2.4 Conditioning Mechanism (SD-specific)

Stable Diffusion에서 텍스트 조건은  
**UNet 내부의 Cross-Attention**을 통해 주입된다.

- Query (Q): 이미지 latent feature
- Key / Value (K, V): 텍스트 embedding (CLIP 출력)

이를 통해 UNet은:
- “현재 latent 위치에서 어떤 텍스트 토큰이 중요한가”를 학습
- 공간적으로 다른 영역에 다른 텍스트 의미 반영 가능

#### Classifier-Free Guidance (CFG)

CFG는 conditional / unconditional 예측을 동시에 수행하여  
텍스트 조건의 강도를 조절하는 기법이다.

- \( \epsilon = \epsilon_{uncond} + s \cdot (\epsilon_{cond} - \epsilon_{uncond}) \)
- guidance scale \( s \) 로 텍스트 반영 강도 제어

CFG는 **추론 단계에서만 적용**되며,
모델 구조를 변경하지 않고 조건성을 강화할 수 있다.

---

## 3. Diffusion Process

### 3.1 Forward / Reverse Diffusion (SD 관점)

- **Forward diffusion**
  - latent에 점진적으로 Gaussian noise를 추가
- **Reverse diffusion**
  - UNet이 noise를 예측하며 latent를 점진적으로 복원

Stable Diffusion은:
- 데이터 \( x_0 \) 를 직접 복원하지 않고
- 각 timestep에서의 noise \( \epsilon \) 을 예측하도록 학습된다

---

### 3.2 Sampler (DDPM, DDIM, etc.)

Sampler는 **diffusion 과정을 어떻게 시간적으로 적분할지**를 결정한다.

- DDPM: stochastic, 많은 step 필요
- DDIM: deterministic, 빠른 수렴
- 기타 DPM-Solver 계열 등

중요한 점:
- **Sampler는 모델이 아니다**
- UNet이 예측한 noise를 어떻게 사용할지만 정의

같은 모델이라도 sampler에 따라:
- 속도
- 샘플 품질
- 질감 특성이 달라질 수 있다

---

## 4. Training Objective

### 4.1 Noise Prediction Objective

Stable Diffusion은 다음 목적을 학습한다.

- 입력: z_t, t
- 타겟: 실제로 추가된 noise ε
- 출력: UNet의 noise 예측 ε_theta

즉,

> “현재 latent 상태에서 제거해야 할 noise가 무엇인가”를 학습

---

### 4.2 Loss Function

기본 loss는 **Mean Squared Error (MSE)** 이다.  
```
L = E[ || ε - ε_θ(z_t, t, c) ||^2 ]
```
- c: 텍스트 조건
- 단순하지만 안정적인 학습 가능
- 대부분의 Stable Diffusion 변형 모델에서도 동일한 objective 유지