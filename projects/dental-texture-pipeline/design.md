# Dental Texture Pipeline Design

## Goal

- 입력: texture가 없는 치아/잇몸 mesh
- 출력: geometry와 정렬된 realistic texture appearance

## Constraints

- mesh geometry는 이미 정확함
- 따라서 생성 결과는 plausibility보다 **geometry fidelity**가 더 중요함
- view별 appearance mismatch는 최종 3D reconstruction에 직접적인 artifact로 이어짐

## Current Pipeline

```text
mesh + segmentation
    -> single-lighting renders
    -> per-view appearance generation
    -> wild-NeRF alignment
    -> relighting / intrinsic consistency correction
    -> Gaussian Splatting reconstruction
```

## Key Question

- 어떻게 realistic appearance를 얻으면서도
  view 간 color / lighting / shadow consistency를 유지할 것인가

## Position Of Previous Work

- Stable Diffusion 계열: direct generation 실험
- MV-Adapter: limited multi-view consistency 확보
- Skyfall-GS: sparse-view 보완 아이디어 참고

현재 메인 문제는 sparse-view 자체보다, 최종 reconstruction 이전에
appearance를 얼마나 안정적으로 정렬할 수 있는가에 있음.
