# Attention

Attention은 query가 key/value 집합에서 필요한 정보를 선택적으로 읽는 연산이다. 생성 모델에서는 이미지 latent, 텍스트 토큰, 다른 view feature 사이의 관계를 학습하는 핵심 메커니즘으로 사용된다.

## Core Idea

- **Query (Q)**: 현재 위치나 토큰이 찾고 싶은 정보.
- **Key (K)**: 참조 대상의 주소 역할을 하는 표현.
- **Value (V)**: 실제로 집계되는 정보.
- **Attention weight**: Q와 K의 유사도를 softmax로 정규화한 가중치.

## Self-Attention

같은 feature 집합 안에서 Q, K, V를 만들고 서로 참조한다. 이미지 latent에서는 공간 위치 간 전역 관계를 학습하고, 텍스트에서는 토큰 간 문맥을 학습한다.

## Cross-Attention

Q와 K/V의 출처가 다르다. Stable Diffusion에서는 이미지 latent가 query가 되고, CLIP text embedding이 key/value가 되어 텍스트 조건을 이미지 생성 과정에 주입한다.

## Related

- [[clip]]
- [[vae]]
- [[multi-view-consistency]]
- [[overview|Stable Diffusion]]
- [[mv-adapter|MV-Adapter]]
