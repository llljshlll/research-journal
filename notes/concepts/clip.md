# CLIP

CLIP은 image-text pair를 같은 embedding space에 정렬하도록 학습된 모델이다. Stable Diffusion에서는 이미지를 직접 생성하지 않고, text encoder가 prompt를 토큰별 의미 벡터로 변환하여 UNet cross-attention의 조건으로 제공한다.

## Stable Diffusion에서의 역할

- prompt를 token sequence로 변환한다.
- 각 token을 의미 embedding으로 인코딩한다.
- UNet의 cross-attention에서 key/value로 사용된다.
- prompt의 시각적 의미를 latent denoising 과정에 주입한다.

## Notes

- CLIP conditioning은 조건의 출처이고, 실제 spatial control은 UNet 내부 attention에서 일어난다.
- 긴 prompt나 세밀한 구문은 tokenization과 attention 분포에 영향을 받는다.

## Related

- [[attention]]
- [[overview|Stable Diffusion]]
- [[mv-adapter|MV-Adapter]]
- [[ctrLoRA|CtrLoRA]]
