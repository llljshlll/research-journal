# Gaussian Splatting

Gaussian Splatting은 3D scene을 explicit Gaussian primitives로 표현하고 differentiable rasterization으로 빠르게 렌더링하는 방식이다. 이 연구에서는 multi-view 생성 결과를 3D representation으로 통합하는 후보지만, view inconsistency에 민감하다는 한계가 반복적으로 관찰된다.

## Key Points

- explicit point/primitive 기반 representation이라 입력 이미지 불일치가 직접 반영된다.
- view-dependent color는 [[spherical-harmonics|Spherical Harmonics (SH)]] 계수로 표현할 수 있다.
- NeRF보다 빠르고 명시적이지만, inconsistent supervision에 의한 artifact가 쉽게 누적된다.

## Related Notes

- [[3d_gaussian_splatting|3D Gaussian Splatting for Real-Time Radiance Field Rendering]]
- [[skyfall-gs|Skyfall-GS]]
- [[multi-view-consistency]]
- [[neural-radiance-fields]]
- [[2026_03_25]]
- [[2026_03_31]]
- [[2026_04_07]]
