# ControlNet

ControlNet은 pretrained Stable Diffusion UNet의 구조를 보존하면서, depth, normal, segmentation, edge map 같은 spatial condition을 추가로 주입하기 위한 구조이다.

## Related

- [[overview|Stable Diffusion]]
- [[attention]]
- [[ctrLoRA|CtrLoRA]]
- [[multi-view-consistency]]

## Core Idea

- pretrained UNet backbone은 고정하거나 안정적으로 보존한다.
- condition branch가 spatial control feature를 생성한다.
- zero convolution을 사용해 초기 학습 시 기존 diffusion prior를 크게 교란하지 않는다.
- 조건별 ControlNet을 학습하면 강한 제어를 얻지만, condition이 늘어날수록 모델 수와 학습 비용이 커진다.

## Research Relevance

치아 도메인에서는 segmentation, curvature, normal, lighting map 등 여러 condition을 동시에 다루어야 하므로, 단일 ControlNet보다 [[ctrLoRA|CtrLoRA]]처럼 condition별 adapter를 분리하는 방향이 비용과 확장성 측면에서 유리하다.
