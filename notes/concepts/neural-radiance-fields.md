# Neural Radiance Fields (NeRF)

NeRF는 scene을 continuous volumetric radiance field로 표현하고 ray integration으로 view synthesis를 수행하는 방식이다. 이 vault에서는 view alignment와 inconsistency 흡수 능력을 Gaussian Splatting과 비교하는 기준으로 등장한다.

## Research Role

- generated multi-view images를 정렬된 3D 공간으로 통합한다.
- volumetric integration 특성 때문에 view inconsistency를 어느 정도 평균화할 수 있다.
- explicit Gaussian Splatting보다 느리지만, 불완전한 supervision에서 더 안정적인 경우가 있다.

## Related Notes

- [[gaussian-splatting]]
- [[multi-view-consistency]]
- [[2026_03_25]]
- [[2026_03_31]]
- [[2026_04_07]]
