# Patchify / Flatten (공간축 펼치기)

2D(또는 3D) 공간 구조의 feature를 1D 토큰 시퀀스로 바꾸는 과정. Transformer는 `(B, N, C)` 시퀀스만 받기 때문에 필요하다.

## Shape

```
이미지/latent: (B, C, H, W)
  → patchify (p×p 조각으로 자름): (B, H/p · W/p, p·p·C)
  → Linear (patch embedding):     (B, N, D)     N = H·W / p²
```

- 단순 flatten은 `p=1`인 경우: 픽셀(또는 latent 위치) 하나가 토큰 하나.
- 3D(voxel)는 `(B, C, D, H, W)` → `N = D·H·W / p³`.

## 왜 하나

- **Attention은 순서 있는 토큰 집합**을 입력으로 가정한다. 격자 구조를 그대로 넣을 수 없다.
- 픽셀 단위로 펼치면 N이 너무 커진다. attention 비용이 `O(N²)`이라서 patch로 묶어 N을 줄인다 (p=2이면 N이 1/4, 비용은 1/16).
- patch 하나가 로컬한 정보를 담고, 이후 attention이 patch 간 전역 관계를 담당한다.

## 주의할 점 (원리상 잃는 것)

- 펼치는 순간 **공간 위치 정보가 사라진다** → positional embedding(절대/RoPE 등)을 더해 줘야 한다. 이게 없으면 토큰 순서를 섞어도 결과가 같다.
- patch 크기 p가 클수록 빠르지만 디테일이 뭉개진다. 그래서 diffusion은 [[vae]]로 먼저 압축한 latent 위에서 patchify한다.
- 나중에 이미지로 되돌리려면 **unpatchify**(역방향 reshape + Linear)가 필요하다.

## 쓰이는 곳

- ViT (픽셀 patch), DiT / FLUX (VAE latent patch), 3D 모델의 voxel/sparse 토큰화 → [[TRELLIS]]
- 멀티뷰, UV 텍스처처럼 여러 격자를 토큰으로 이어 붙일 때 → [[multi-view-consistency]]

## Related

- [[transformer-block]]
- [[linear]]
- [[conv-vs-attention]]
