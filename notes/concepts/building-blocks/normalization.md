# Normalization

feature의 평균/분산을 맞춰서 학습을 안정화하는 연산. 종류에 따라 "무엇을 기준으로 정규화하느냐"가 다르다.

## 왜 쓰나

- 층이 깊어지면 activation 스케일이 폭주하거나 사라져 학습이 불안정해진다.
- 스케일을 맞춰 주면 더 큰 learning rate를 쓸 수 있고 수렴이 빨라진다.

## 종류

| 이름 | 정규화 기준 | 주로 쓰이는 곳 |
|---|---|---|
| BatchNorm | batch 전체에서 채널별 통계 | 초기 CNN. batch 크기에 의존해서 생성 모델에선 잘 안 씀 |
| **LayerNorm** | 한 토큰의 모든 채널 | Transformer 기본 |
| RMSNorm | LayerNorm에서 평균 제거 생략, RMS만 사용 (더 가벼움) | LLM, 최근 DiT 계열 |
| **GroupNorm** | 채널을 그룹으로 묶어 그룹 내 통계 | Stable Diffusion U-Net, [[vae]] (batch 작아도 안정) |
| **AdaLN** | LayerNorm 후 scale/shift를 **조건에서 예측** | DiT 계열 (timestep/class 조건 주입) |

공통 형태: `y = γ · (x − μ) / σ + β`  (γ, β는 학습 파라미터)

## AdaLN (조건 주입 관점)

- 일반 LayerNorm의 γ, β는 고정 파라미터. AdaLN은 조건(timestep, 텍스트 pooled 등)을 [[linear]]에 통과시켜 γ, β를 **샘플마다 동적으로 생성**한다.
- 즉 "이 timestep에서는 feature를 이렇게 변조하라"는 방식. cross-attention보다 가볍게 전역 조건을 넣을 수 있다.
- adaLN-Zero: 출력 gate를 0으로 초기화해서 학습 초반에 블록이 identity처럼 동작하게 함.

## Pre-Norm vs Post-Norm

- Post-Norm: `Norm(x + F(x))`. 원조 Transformer. 깊어지면 학습이 어려움.
- **Pre-Norm**: `x + F(Norm(x))`. residual 경로가 깨끗해서 깊은 모델에서 안정적. 현재 표준 → [[transformer-block]]

## Related

- [[transformer-block]]
- [[linear]]
- [[attention]]
