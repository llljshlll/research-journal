# Texture Baking

Texture baking은 view-space image 정보를 UV texture 또는 material map으로 옮기는 과정이다. 현재 연구에서는 single-view 또는 multi-view generated images를 치아 mesh texture로 누적하는 핵심 단계이며, shading과 visibility 처리가 주요 문제로 나타난다.

## Open Issues

- shading이 texture에 같이 bake되면 view가 바뀔 때 lighting inconsistency가 발생한다.
- 보이지 않는 texel이나 count=0 영역은 inpainting 또는 추가 optimization이 필요하다.
- 여러 pixel이 같은 texel에 투영될 때 averaging, overwrite, visibility mask 정책이 결과를 크게 바꾼다.

## Related Notes

- [[inverse-rendering]]
- [[intrinsic-decomposition]]
- [[multi-view-consistency]]
- [[2026_04_16_2]]
- [[2026_04_17]]
- [[2026_04_19]]
- [[2026_04_20]]
- [[2026_04_21]]
- [[2026_04_28]]
- [[2026_04_29]]
