# Multi-view Consistency

Multi-view consistency는 서로 다른 camera view에서 같은 geometry, identity, texture, lighting 해석이 유지되는 성질이다. 현재 연구 흐름에서는 diffusion 기반 view 생성 결과가 view마다 달라지면서 NeRF, Gaussian Splatting, texture baking 단계에 오류가 전파되는 핵심 병목으로 등장한다.

## Failure Modes

- view별 albedo 또는 lighting이 달라져 texture가 평균화되거나 얼룩진다.
- diffusion 결과의 stochasticity가 geometry distortion으로 누적된다.
- 보이지 않는 영역을 추정할 때 identity나 구조가 바뀐다.
- image-level 보정 결과를 3D representation에 넣을 때 flickering 또는 duplicate 구조가 생긴다.

## Related Notes

- [[2026_03_25]]
- [[2026_03_31]]
- [[2026_04_07]]
- [[2026_04_17]]
- [[2026_04_19]]
- [[2026_04_28]]
- [[mv-adapter|MV-Adapter]]
- [[tinker|TINKER]]
- [[gaussian-splatting]]
- [[neural-radiance-fields]]
