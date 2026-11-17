# Linear (Fully Connected)

입력 벡터에 행렬을 곱하고 bias를 더하는 가장 기본적인 연산. `y = Wx + b`

## Shape

- 입력: `(..., C_in)` → 출력: `(..., C_out)`
- 마지막 채널 축에만 작용하고, 앞쪽 축(batch, 토큰 수 등)은 건드리지 않는다.
- 토큰 시퀀스 `(B, N, C)`에 적용하면 **모든 토큰에 같은 W를 독립적으로** 적용한다 (토큰끼리 섞이지 않음).

## 왜 쓰나

- **차원 변환**: 서로 다른 차원의 표현을 이어 붙이기 위해. 예) patch 픽셀값 → 모델 hidden dim, hidden dim → 출력 채널.
- **Projection**: 같은 입력에서 목적이 다른 표현을 뽑기 위해. attention의 Q, K, V가 각각 다른 Linear로 만들어진다.
- **특징 혼합**: 한 위치 안에서 채널끼리 섞는 유일한 기본 연산 (공간/토큰 간 혼합은 attention이나 conv가 담당).
- Linear만 쌓으면 결국 하나의 행렬이므로, 사이에 비선형(GELU 등)을 넣어야 표현력이 생긴다 → [[transformer-block]]의 MLP.

## 원리 / 직관

- W의 각 행은 "입력에서 어떤 패턴을 찾는 필터". 출력 채널 하나 = 입력과 필터의 내적.
- `1x1 conv`와 수학적으로 동일하다. 이미지 feature map에서 채널만 섞는 연산.
- 차원을 줄이면 bottleneck(압축), 늘리면 expansion(MLP에서 보통 4배).

## 쓰이는 곳

- [[attention]]의 Q/K/V/output projection
- [[patchify-flatten]] 직후의 patch embedding
- Transformer MLP, 조건(timestep, text)을 hidden dim으로 맞추는 embedding
- [[vae]]나 diffusion의 마지막 출력 head

## Related

- [[transformer-block]]
- [[normalization]]
