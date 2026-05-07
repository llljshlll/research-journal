# Inverse Rendering

Inverse rendering은 관측 이미지와 렌더링 결과가 일치하도록 material, texture, lighting, geometry 관련 파라미터를 역으로 최적화하는 방법이다. 현재 연구에서는 Mitsuba 기반으로 target view에서 texture 또는 unlit color를 최적화하는 흐름에 사용된다.

## Current Use

- Blender/Mitsuba 렌더러 차이를 맞추고 평가 기준을 Mitsuba로 통일한다.
- 단일 view target에서 albedo 또는 texture map을 최적화한다.
- shading bake를 단순 평균 대신 differentiable rendering 기반으로 대체하는 방향을 검토한다.

## Related Notes

- [[intrinsic-decomposition]]
- [[texture-baking]]
- [[2026_04_15]]
- [[2026_04_16_1]]
- [[2026_04_16_2]]
- [[2026_04_21]]
- [[IntrinsicAnything|IntrinsicAnything]]
