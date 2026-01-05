# FLUX ARCHITECTURE ANALYSIS

목차
1. Global Architecture
2. Input
3. Core Mechanisms (RoPE, Modulation)
4. Double Stream Block
5. Single Stream Block
6. RoPE Attention
7. Modulation

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
q_even : (B, 24, L, 64)  
q_odd  : (B, 24, L, 64)  
  
EmbedND에서 생성된 P.E는 내부적으로 다음 형태로 사용  
cos : (1, L, 64)
sin : (1, L, 64)


Attention 연산 시에는 broadcast  
cos, sin → (1, 1, L, 64)
즉, 같은 토큰 위치와 같은 feature index에 대해서는 모든 head가 동일한 cos/sin 값을 공유.


회전 연산은 각 feature pair에 대해 다음과 같이 적용

x_even_rot = q_even * cos - q_odd * sin
x_odd_rot  = q_even * sin + q_odd * cos

이후 두 값을 다시 interleave하여 원래 차원으로 복원
(x0', x1'), (x2', x3'), … → [x0', x1', x2', x3', ..., x127']


최종 결과:
```
Q_rot, K_rot : (B, 24, L, 128)
```
RoPE는 Query(Q)와 Key(K)에만 적용  
위치 정보는 토큰 간의 비교(Q·K 내적)에만 필요하며, 실제 내용을 담고 있는 Value에는 적용하지 않음  
이후 RoPE로 회전된 Q와 K를 사용하여 일반적인 attention을 수행함   








