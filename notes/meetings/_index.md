# Meeting Index

## Research Flow

- [[2026_03_25]]-[[2026_04_07]]: diffusion 기반 multi-view 생성 결과의 albedo/lighting inconsistency를 확인하고, NeRF와 Gaussian Splatting의 representation 차이를 비교했다. 핵심 병목은 [[multi-view-consistency]]로 정리된다.
- [[2026_04_15]]-[[2026_04_17]]: Mitsuba 기반 [[inverse-rendering]]으로 single-view texture extraction과 progressive texture filling을 실험했다. 단순 mask 기반 hole filling은 view-dependent shading이 texture에 bake되어 multi-view에서 깨지는 한계를 보였다.
- [[2026_04_19]]-[[2026_04_20]]: lighting 고정, UV visibility mask, freeze mask를 도입했다. sequential optimization은 consistency를 유지하지만 초기 오류를 고치기 어려워, sequential warm-up 후 global optimization으로 전환했다.
- [[2026_04_21]]-[[2026_04_29]]: [[texture-baking]], I2I inpainting, UV resolution, seam, texel sampling 문제를 디버깅했다. 현재 방향은 albedo-like texture를 안정화하고, I2I는 mask-limited hole filling 용도로 제한하는 것이다.

## March 2026

- [[2026_03_25]]
- [[2026_03_31]]

## April 2026

- [[2026_04_07]]
- [[2026_04_15]]
- [[2026_04_16_1]]
- [[2026_04_16_2]]
- [[2026_04_17]]
- [[2026_04_19]]
- [[2026_04_20]]
- [[2026_04_21]]
- [[2026_04_28]]
- [[2026_04_29]]
