# FLUX

## 1. Overview
FLUX = **Transformer Diffusion** Based Multimodal Generative AI
![Compare Unet to DiT](../../docs/assets/models/flux/UNet_DiT.png)
- 기존의 UNet architecture는 convolution 연산의 특성상 local context를 모델링하는데 강점이 있지만, global context를 이해하는데 한계가 있음
=> 그래서 FLUX는 diffuison transformer architecture를 최초로 적용함
  Transformer 기반의 DiT는 복잡한 시각적 컨텍스트와 레이아웃 변화를 더 효과적으로 모델링할 수 있음

** ControlNet 친화적 모델이 아님, geometry-aware 생성은 아직 하지 못함

```
Input
├── Text Prompt: T5 및 CLIP을 통해 처리
├── Image: 참조 이미지 또는 마스크 (Kontext, Fill, Redux 모델 등에서 사용)
└── Other Modalities:
    ├── Structural Guidance: Canny edge, Depth map (Flux.1 Tools) 
    └── Text-Specific Priors: Glyph map, Position mask (FLUX-Text 변형 모델)
Process
├── Text Encoding → T5 Text Encoder 및 CLIP
├── Image Encoding → VAE Encoder → 16개 채널의 Latent Space z 로 변환
├── Mechanism (Transformer 기반):
    ├── Double Stream Blocks: 텍스트와 이미지 토큰을 개별 가중치로 병렬 처리
    ├── Single Stream Blocks: 결합된 시퀀스를 통합 처리하여 효율성 극대화 (38개 블록)
    └── 3D RoPE (Rotary Positional Embeddings): 시공간 좌표(t, H, W) 인덱싱으로 구조적 정밀도 향상
├── Denoising & Refinement → Rectified Flow Matching (UNet 대신 DiT denoiser 사용)
└── In-Context Integration → Sequence Concatenation: 참조 이미지 토큰을 타겟 토큰에 직접 연결
Output
└── VAE Decoder → 고해상도 이미지 생성 (1024×1024 표준, Pro Ultra의 경우 최대 4K 지원)


```
## 2. Architecture
![FLUX Architecture](../../docs/assets/models/flux/FLUX_architecture.png)
1. Input Section

| Symbol | Description |  
|------|------------|  
| **img** | VAE Encoder를 통해 latent space로 변환된 이미지 토큰 |  
| **timesteps** | diffusion 과정의 단계 나타냄. Sinusoidal Timestep Embedding을 통해 벡터화됨 |  
| **guidance** | 텍스트 조건을 얼마나 강하게 반영할지 결정하는 가이드 스케일 |  
| **y** | CLIP 텍스트 인코더의 출력 (Global 텍스트 정보) |  
| **txt** | T5-XXL 텍스트 인코더의 출력 (시퀀스 단위 의미 정보) |  
| **img_ids** | 이미지 토큰의 공간 좌표 (h, w) 및 시간 인덱스 (used in RoPE) |  
| **txt_ids** | 텍스트 토큰의 위치 인덱스 (used in RoPE) |  

2. Main Architecture Components

| Symbol / Block | Description |
|---------------|------------|
| **CAT** | Concatenation |
| **DoubleStream Block (N = 19)** | 이미지 토큰과 텍스트 토큰을 분리된 스트림으로 병렬 처리하는 블록 |
| **SingleStream Block (M = 38)** | 이미지·텍스트 토큰을 하나의 시퀀스로 통합해 처리하는 블록 |
| **PE** | 3D RoPE 기반 위치 임베딩 (h, w, t 정보 포함) |
| **vec** | timestep 및 txt 전역 정보를 결합한 conditioning 벡터 |

3. Modulation & Conditioning

| Symbol | Description |
|------|------------|
| **Mod** | vec를 입력으로 받아 feature-wise modulation을 수행하는 유닛 |
| **α (gate**) | 연산 결과의 기여도를 조절하는 게이팅 파라미터 |
| **β (shift**) | feature 값을 평행 이동시키는 bias 항 |
| **γ (scale**) | feature 강도를 조절하는 scaling 계수 |
| **Chunk** | 하나의 modulation 벡터를 (α, β, γ)로 분할 |
| **Unsqueeze** | modulation 값을 모든 토큰 차원에 브로드캐스트하기 위한 차원 확장 |

4. Spatial & Dimensional Notation

| Symbol | Meaning |
|------|--------|
| **t** | 참조 인덱스를 구분하는 오프셋 |
| **T** | 시퀀스의 총 길이 (H × W × (N+1) |
| **N** | reference 이미지 수 |
| **h** | Hidden Dimension(채널) |
| **md** | modulation이 적용되는 feature dimension 크기 |
| **d tensors** | chunk 연산을 통해 생성된 개별 modulation 파라미터 텐서 |



T5 (T5-v1.1-XXL): 매우 방대한 파라미터를 가진 텍스트 인코더로, 자연어의 복잡한 문맥과 상세한 지시 사항을 깊이 있게 이해하는 데 사용됩니다. 이는 Flux 모델이 긴 프롬프트를 정확하게 따르고 높은 수준의 '프롬프트 충실도(Prompt Following)'를 보여주는 핵심 이유입니다
CLIP (CLIP-L): 시각 정보와 텍스트 사이의 연관성을 학습한 인코더로, 프롬프트의 의미가 이미지와 시각적으로 얼마나 잘 일치하는지를 조절하는 역할을 합니다.
이 두 가지 인코더를 통해 생성된 텍스트 임베딩은 모델 내부의 Double Stream Blocks에서 이미지 토큰과 병렬로 처리된 후, Single Stream Blocks를 거치며 시각 정보와 깊게 결합됩니다.


Image Encoding → VAE Encoder → 16개 채널의 Latent Space z 로 변환
VAE Encoder는 원래 이미지를 C*H*W 형태로 변환함
그런데 transformer architecture는 입력을 1차원 시퀀스로 처리해야하기 때문에 16차원의 채널 백터를 하나의 토큰으로 처리함
즉, H×W 개의 좌표에 있는 벡터들을 일렬로 Flattening해서 하나의 긴 시퀀스로 만들어서 처리함
512*512 image 넣으면 16*64*64라면, 총 16개 채널의 4,096개의 토큰이 생김



3D RoPE(3-Dimensional Rotary Positional Embedding)
앞에서 펼쳐진 토큰들이 원래 이미지에서의 위치 정보를 잃어버릴 위험이 있기 때문에 3D RoPE 사용
각 토큰에 원래 특징 맵에서의 **공간 좌표 (H, W**)와 시간 정보(토큰의 주소/위치 t)를 인덱싱하여 주입
시공간 좌표((t,H,W))를 트랜스포머 연산에 사용할 수 있는 PE(Positional Embedding) 형태로 변환
------------코드 보기
이를 통해 모델은 토큰들이 시퀀스 형태로 나열되어 있더라도, 어떤 토큰이 이미지의 어느 부분(왼쪽 위, 오른쪽 아래 등)에 해당하는지를 정확히 이해하고 구조적으로 이미지를 생성할 수 있음



Double Stream Blocks
이미지 토큰과 텍스트 토큰에 대해 각각 별도의 가중치(Separate weights)**를 할당하여 병렬로 처리하는 구조
Text Stream : 텍스트 인코더(T5, CLIP 등)를 통해 들어온 언어적 의미 정보를 처리
Visual Stream: VAE를 통해 인코딩된 이미지의 잠재 토큰(Latent tokens)들을 처리
1. 시퀀스 연결(Concatenation): 별도로 흐르던 이미지 토큰 시퀀스와 텍스트 토큰 시퀀스를 하나로 합칩니다.
2. 통합 어텐션 수행: 합쳐진 전체 시퀀스 위에서 어텐션 연산을 수행하여, 이미지 토큰이 텍스트의 맥락을 읽고 텍스트 토큰이 이미지의 구조를 파악하게 합니다.
3. 다시 분리: 어텐션 연산이 끝나면 정보가 교류된 토큰들을 다시 각자의 스트림(가중치)으로 돌려보내 다음 처리를 이어갑니다

vec (Vector Conditioning): 타임스텝 임베딩(Timestep embedding)과 텍스트의 전역적인 의미 정보를 담고 있는 벡터
Modulation 과정: 이 vec는 블록 내부의 Linear 층과 SiLU 활성화 함수를 통과하며 해당 블록에 필요한 구체적인 변조 파라미터들(α,β,γ 등)을 생성합니다. 이 파라미터들이 이미지(img)와 텍스트(tex) 스트림 각각에 적용됩니다.
Shift (β): 정규화된 데이터에 특정 값을 더해주는(Addition) 역할을 합니다. 이는 데이터의 전체적인 '편향(Bias)'을 조절하여 텍스트 조건에 맞게 특징의 기준점을 이동시킵니다
 Scale (γ): 정규화된 데이터에 특정 값을 곱해주는(Multiplication) 역할을 합니다. 특정 특징의 '강도(Intensity)'나 중요도를 증폭하거나 감쇠시켜 지시 사항에 맞는 시각적 요소를 강조합니다
 Gate (α): 어텐션(Attention)이나 MLP 연산의 결과물이 잔차 연결(Residual Connection)을 통해 원래 데이터와 합쳐지기 직전에 적용됩니다. 이는 해당 층에서 계산된 정보가 최종 출력에 얼마나 기여할지를 결정하는 '개폐기' 역할을 하며, 모델의 학습 안정성을 높이고 조건부 생성을 정교하게 제어합니다
트랜스포머 기반의 DiT 블록은 크게 두 부분으로 구성되며, 각 부분 직전에 Modulation이 적용되기 때문에 1과 2로 구분됩니다.
• 1 (Pre-Attention): 어텐션(Attention) 층에 들어가기 전에 적용되는 파라미터들입니다. 이미지와 텍스트 토큰이 서로를 참조하기 전에 각자의 특징을 준비시킵니다.
• 2 (Pre-MLP): 어텐션 이후 MLP(Feed-Forward) 층에 들어가기 전에 적용되는 파라미터들입니다. 어텐션을 통해 융합된 정보를 개별 토큰 단위에서 세밀하게 다듬는 과정을 제어합니다.
tex와 img로 나뉘는 이유 : Double Stream Blocks 아키텍처의 핵심은 이미지와 텍스트가 **서로 다른 가중치(Separate weights)**를 가진다는 점



+한 스텝에서 DoubleStream Block과 SingleStream Block에 각각 들어오는 P.E, vec값은 같지만, 
vec는 각 층의 Modulation유닛의 Linear층이 각 블록마다 다르게 학습된 가중치 가지고 있어서 동일한 vec들어오더라도 서로 다른 α,β,γ 파라미터 생성해냄
P.E도 RoPE Attention을 이용해서

+Skyfall-GS에서는 diffusion을 완전한 생성기가 아니라 구조가 이미 주어진 이미지의 노이즈 제거 및 정제 단계로 사용함
저노이즈, 저스텝 조건에서는 전역 구조를 강하게 유지하는 Transformer 기반 diffusion이 UNet 기반 모델보다 안정적일 수 있음

+ reference image와 noise image가 항상 concat되어 img형태로 함께 사용되다가, 마지막 SingleStream Block을 지나고 다음 단계로 에측된 노이즈를 전달할 때에는 reference image가 슬라이싱되어 버려짐. 그리고 다음 단계에서는 다시 처음 준비된 latent 형태로 noise image와 concat 되어 들어감
즉, Reference image는 처음 준비된 latent 형태로 매 스텝마다 동일하게 재사용됨


해결해야하는 의문
1. timesteps이랑 guidance가 Sinusoidal timestep embedding으로 들어가는데 이게 뭔지
2. 그 이후에 Sinusoidal timestep embedding에서 나온거랑 CLIP에서 나온거랑 다 MLP Emb들어갔다가 더해지는데 이건 뭔지
3. 사진 속에서는 img가 vae가 아니라 linear를 통과하는거처럼 나오는데 뭐가 맞는지
4. 파라미터 개수






- 고해상도에서도 구조적으로 안정적인 이미지 생성(기존 unet based diffusion model은 고해상도에서 구조 붕괴 일어남)
- transformer diffusion의 실무 기준 제시



