# FLUX ARCHITECTURE ANALYSIS

목차
1. Global Architecture
2. Input
3. Core Mechanisms (RoPE, Modulation)
4. Double Stream Block
5. Single Stream Block


## 1. Global Architecture
ex) ![Overall pipeline of Skyfall-GS](../../../docs/assets/papers/flux/FLUX_global_architecture_shape.png)
- hidden_dim = 3072
- num_heads = 24
- head_dim = 128  (3072 = 24 × 128)
- txt_seq_len = 512
- img_seq_len = 4096


## 2. Input

### 2.1 Text Encoders

FLUX uses two text encoders with different roles.
  
- **T5 (T5-v1.1-XXL)**
  - 시퀀스 형태의 텍스트 토큰을 생성
  - Transformer 블록 내부에서 이미지 토큰과 attention을 통해 상호작용
  - Shape: (B, 512, 4096) → Linear → (B, 512, 3072)  

- **CLIP (CLIP-L)**
  - 하나의 global text embedding을 출력
  - timestep, guidance와 함께 global conditioning vector(vec)로 사용
  - Shape: (B, 768) → MLP → (B, 3072)  

### 2.2 Image Encoding

FLUX VAE encoder는 1024×1024 이미지를 (B, 16, 128, 128) 형태의 latent feature map으로 변환함  

Image (B, 3, 1024, 1024)  
→ VAE Encoder (B, 16, 128, 128)  
→ 2×2 spatial packing (B, 64, 64, 64)  
→ flatten (B, 4096, 64)  
→ Linear(64 → 3072)  
→ (B, 4096, 3072)  
  
Transformer 입력을 위해, latent feature map은 2×2 spatial packing을 거쳐 공간 해상도를 줄이고 채널 차원을 확장함  
이후 각 공간 위치를 하나의 image token으로 취급하여 (B, 4096, 64) 시퀀스를 구성하고,  
Linear projection을 통해 hidden_dim 공간으로 변환  


### 2.3 Positional Indices for RoPE (EmbedND)

Transformer model을 사용하면, 이미지 latent가 1D 시퀀스로 펼쳐지면서, 각 토큰의 원래 공간 위치 정보가 사라진다.  
이를 보완하기 위해 FLUX는 3D RoPE를 사용한다.  
  
- 각 토큰에 대해 (t, h, w) 형태의 3차원 인덱스를 정의
- text token index와 image token index를 concat하여 EmbedND에 전달
- EmbedND는 이후 RoPE attention에서 사용되는 회전 파라미터를 생성

**EmbedND** : 
- RoPE(Rotary Positional Encoding)를 위한 보조 모듈
- attention 단계에서 Q와 K를 회전시키기 위한 위치 기반 회전 계수(cos/sin)를 생성하는 역할
- 학습가능한 파라미터가 없음


## 3. Core Mechanisms (RoPE, Modulation)

### 3.1 RoPE Attention

RoPE(Rotary Positional Encoding)는 attention에서 위치 정보를 주입하기 위한 방식.
토큰 feature에 위치 벡터를 더하지 않고, Query(Q)와 Key(K)를 회전시키는 방식으로 위치 정보 반영.
```
P.E : (1, L, 128, 2, 2)
```  
- L : 전체 토큰 길이
- 128 : head_dim
- (2, 2) : 각 feature pair에 대한 2D 회전 행렬(cos / sin)
각 토큰 위치마다, 각 feature pair에 적용할 회전 행렬을 미리 계산해 둔 값  

```
input x : (1, 4068, 3072)
```  

hidden dimension은 다음과 같이 분해
```
3027 = num_heads * head_dim = 24 * 128
```  
따라서 Query와 Key의 shape은 다음과 같음.
```
Q, K : (B, 24, L, 128)
```
각 (batch, head, token)에 대해 128차원 벡터 하나를 의미함  

RoPE에서는 128차원 벡터를 다음과 같이 해석  
```
[x0, x1, x2, x3, x4, x5, ..., x126, x127]
→ (x0, x1), (x2, x3), (x4, x5), ..., (x126, x127)
```
  
128차원 = 64개의 2차원 벡터, 각 2차원 백터가 하나의 회전 단위  
따라서 Q, K는 다음과 같이 분리됨  
```
q_even : (B, 24, L, 64)  
q_odd  : (B, 24, L, 64)
```  
  
EmbedND에서 생성된 P.E는 내부적으로 다음 형태로 사용  
```
cos : (1, L, 64)
sin : (1, L, 64)
```  


Attention 연산 시에는 broadcast  
```
cos, sin → (1, 1, L, 64)
```  
즉, 같은 토큰 위치와 같은 feature index에 대해서는 모든 head가 동일한 cos/sin 값을 공유.


회전 연산은 각 feature pair에 대해 다음과 같이 적용
```
x_even_rot = q_even * cos - q_odd * sin
x_odd_rot  = q_even * sin + q_odd * cos
```

이후 두 값을 다시 interleave하여 원래 차원으로 복원
```
(x0', x1'), (x2', x3'), … → [x0', x1', x2', x3', ..., x127']
```  


최종 결과:
```
Q_rot, K_rot : (B, 24, L, 128)
```
RoPE는 Query(Q)와 Key(K)에만 적용  
위치 정보는 토큰 간의 비교(Q·K 내적)에만 필요하며, 실제 내용을 담고 있는 Value에는 적용하지 않음  
이후 RoPE로 회전된 Q와 K를 사용하여 일반적인 attention을 수행함   


### 3.2 Modulation

Modulation은 diffusion timestep과 텍스트의 전역적 의미 정보를 결합한
conditioning 벡터 `vec`를 이용해 각 블록의 연산을 조건에 맞게 조절하는 메커니즘.

기존 Stable Diffusion의 time embedding은 현재 timestep 정보를 제공하지만,
해당 step에서 어떤 연산을 얼마나 반영할지는 명시적으로 제어하지 않고 학습에 맡김  

Modulation은
timestep + 텍스트 + guidance → vec → shift / scale / gate를 생성하여
연산 입력 자체와 연산 결과의 반영 비율을 직접 조절  
  
즉, 같은 attention, 같은 MLP, 같은 가중치를 사용하면서도
각 step에서 해당 연산을 얼마나 신뢰할지를 구조적으로 제어  

**vec**  
Diffusion timestep embedding과 텍스트의 전역적 의미 정보를 결합한 conditioning 벡터.

`vec`는 블록 내부의 Linear layer와 SiLU 활성화를 거쳐 해당 블록에서 사용할 변조 파라미터 (α, β, γ)를 생성.  
이 파라미터들은 img 스트림과 txt 스트림에 각각 적용.  

**Shift (β)**   
정규화된 데이터에 bias를 더해 특징의 기준점을 이동  

**Scale (γ)**   
정규화된 feature에 스케일을 곱해 특정 특징의 강도를 증폭 또는 감쇠  

**Gate (α)**  
Attention 또는 MLP 연산 결과가 residual connection으로 합쳐지기 직전에 적용되는 가중치  
해당 층에서 계산된 정보가 최종 출력에 기여하는 비율을 조절하며, 모델의 학습 안정성과 조건부 제어 정밀도 향상에 기여   

트랜스포머 기반 DiT 블록은 두 개의 주요 연산 단계로 구성되며, 각 단계 직전에 Modulation이 적용됨  

1. **Pre-Attention**  
   Attention 연산 이전에 적용되어 이미지 및 텍스트 토큰의 특징을 조건에 맞게 정렬  

2. **Pre-MLP**  
   Attention 이후 MLP 이전에 적용되어 융합된 정보를 토큰 단위에서 정밀하게 조정  

이미지(img)와 텍스트(txt) 스트림이 분리되어 Modulation이 적용되는 이유는  
Double Stream Block이 이미지와 텍스트에 서로 다른 weight를 사용하는 구조이기 때문  









