# TAESD

***Tiny AutoEncoder for Stable Diffusion*** – Stable Diffusion의 VAE를 매우 경량화한 tiny autoencoder

---

## Overview

- Stable Diffusion이 사용하는 VAE와 동일한 **latent API** 를 사용하는 **초경량 AutoEncoder** 모델
- 원본 Stable Diffusion VAE 대비 **매우 작은 파라미터 수**로 latents → full-size 이미지 decoding 을 거의 실시간에 가깝게 수행할 수 있음

---

## Performance / Characteristics

| 항목 | Stable Diffusion VAE | TAESD |
|------|----------------------|-------|
| Encoder 파라미터 | ~34M | ~1.2M |
| Decoder 파라미터 | ~49M | ~1.2M |
| Real-time preview | ✗ | ✓ |
| 세부 디테일 재현 | 높음 | 떨어짐 |
| 메모리/속도 | 느림 | 빠름 |
| 용도 | 최종 출력 | 빠른 미리보기/latents 시각화 |  

> TAESD는 trade-off를 통해 속도를 얻는 대신 세부 디테일 품질을 희생함.
---

## Architecture
  
TAESD는 **encoder + decoder** 구조이며 다음과 같은 특징이 있음  
- Conv + ReLU 기반 경량 네트워크  
- upsample 레이어로 latent → 이미지 복원  
- Stable Diffusion latents (shape: `[4, H/8, W/8]`) 를 입력으로 처리

---

## Installation & Usage Examples (Diffusers)

아래 예시는 Hugging Face `diffusers`에서 **TAESD** 를 Stable Diffusion pipeline의 VAE로 사용하는 코드이다:

```python
import torch
from diffusers import DiffusionPipeline, AutoencoderTiny

# pipeline 로드
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-1-5-base", torch_dtype=torch.float16
)

# TAESD VAE 로드
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "A technical diagram showing TAESD architecture"
image = pipe(prompt, num_inference_steps=25).images[0]
image.save("taesd_inference.png")
```
> TAESD는 이미지 스케일링 및 range 기대값 [0, 1] 기준으로 처리함
> stable diffusion의 VAE는 [-1, 1] 기준으로 처리함