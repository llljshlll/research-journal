# Transformer Block

토큰 시퀀스를 입력받아 같은 shape의 토큰 시퀀스를 출력하는 반복 단위. 이 블록을 L번 쌓은 것이 Transformer다.

## Shape

- 입력/출력 모두 `(B, N, C)`. shape이 유지되므로 몇 층이든 쌓을 수 있다.

## 구성 (Pre-Norm 기준)

```
x = x + Attention(Norm(x))   # 토큰 간 정보 교환
x = x + MLP(Norm(x))         # 토큰 내부 채널 가공
```

| 구성 요소 | 역할 |
|---|---|
| [[attention]] | 토큰 **사이**의 관계를 읽어 정보를 섞음 (공간/문맥 혼합) |
| MLP (Linear → GELU → Linear) | 각 토큰을 **독립적으로** 가공 (채널 혼합, 비선형성). 지식 저장소 역할 |
| [[normalization]] | 값의 스케일을 안정화해 깊게 쌓아도 학습되게 함 |
| Residual (`x + ...`) | 입력을 그대로 보존하는 지름길. gradient가 깊은 층까지 흐르게 하고, 블록이 "변화량만" 학습하게 함 |

## 왜 이렇게 설계했나

- Attention만으로는 토큰 간 혼합만 가능하고 비선형 가공이 없다. MLP가 그 역할을 채운다.
- 두 역할(토큰 간 / 토큰 내)을 번갈아 하면서 점점 추상적인 표현을 만든다.
- Residual 덕분에 층이 깊어져도 학습이 안정적이다.

## 변형

- **Self-attention block**: Q, K, V가 같은 토큰에서 나옴.
- **Cross-attention 추가**: 조건(text 등)을 K/V로 주입. Stable Diffusion U-Net 안의 블록.
- **DiT / AdaLN block**: timestep, 조건으로 norm의 scale/shift를 조절. → [[normalization]]
- **MM-DiT (FLUX 등)**: 이미지 토큰과 텍스트 토큰을 이어 붙여 하나의 attention으로 처리. → [[flux_overview]]

## 쓰이는 곳

- ViT, DiT, FLUX, TRELLIS의 sparse transformer 등 거의 모든 현대 생성 모델의 backbone.

## Related

- [[linear]]
- [[patchify-flatten]]
- [[conv-vs-attention]]
