# Intrinsic Decomposition

Intrinsic decomposition은 이미지를 albedo, shading, specular/lighting 성분 등으로 분리하는 접근이다. 이 vault에서는 diffusion으로 생성된 multi-view 이미지의 lighting inconsistency를 줄이기 위한 후보 방법으로 정리된다.

## Research Role

- view마다 달라지는 lighting을 texture에 bake하지 않도록 분리한다.
- albedo 중심 representation을 얻어 Gaussian Splatting이나 texture optimization의 입력을 안정화한다.
- inverse rendering과 함께 PBR parameter 추정으로 확장할 수 있다.

## Related Notes

- [[inverse-rendering]]
- [[texture-baking]]
- [[2026_04_07]]
- [[2026_04_15]]
- [[2026_04_21]]
- [[IntrinsicAnything|IntrinsicAnything]]
