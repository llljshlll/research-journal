# Texture 생성 논문 비교

> 6편(Text2Tex, Paint3D, TEXGen, DirectUV, TexDreamer, Generative Detail Enhancement)의 접근 방식 비교
> 
> 분류 : View 기반 생성(Text2Tex, Paint3D) / UV 직접 생성(TEXGen, DirectUV, TexDreamer) / 기존 material 강화(Generative Detail Enhancement)

## 1. 한눈에 비교

| 항목 | Text2Tex | Paint3D | TEXGen | DirectUV | TexDreamer | Generative Detail Enhancement |
|---|---|---|---|---|---|---|
| 목표 | mesh의 texture 생성 | mesh의 texture 생성 (lighting-less) | mesh의 texture 생성 | mesh의 texture 생성 | 3D human(SMPL UV)의 texture 생성 | 기존 PBR material의 detail 강화 |
| 처리 흐름 | View → UV projection | View → UV refinement | UV 직접 생성 | UV 직접 생성 (latent UV space) | UV 직접 생성 (semantic UV) | View 생성 → inverse rendering |
| 입력 | mesh + text | mesh + text/image | mesh + text + single-view image | mesh + reference image (+ coarse UV) | text 또는 image | mesh + 기존 material + text |
| 출력 | UV texture (RGB) | UV texture (RGB, lighting 제거, 2K) | 1024×1024 UV texture | 1024×1024 UV texture | 1024×1024 UV texture | PBR material (albedo, roughness, normal) |
| 2D diffusion 사용 방식 | Stable Diffusion v2 Depth2Image + inpainting mask | SD 1.5 + ControlNet (depth, inpaint, tile, position) | 사용 안 함 (UV space에서 직접 학습한 UNet) | Flux 구조 DiT를 직접 학습 (Flux VAE는 frozen) | SD 2.1에 LoRA fine-tuning | SD 1.5 + ControlNet (tile, normal) |
| geometry 조건 | depth | depth + position map(UV) | position map(UV) + 3D point | CCM(surface 좌표)을 RoPE에 주입 (SAPE) | 없음 (SMPL UV 구조를 LoRA가 학습) | normal render |
| 추가 학습 | 불필요 | Position Encoder(`tau_p`)만 학습, 기존 diffusion은 freeze | 700M parameter model 전체 학습 | 전체 학습 (A800 8장, 170K step) | LoRA(T2UV), feature translator(I2UV) | 불필요 (training-free) |
| 학습 data | - | Objaverse 105,000 mesh | Objaverse 120,000 mesh | TexVerse + PartNext + Objaverse 약 100K mesh | 소량 sample texture (약 100개에서 포화) + ATLAS (50k texture 합성) | - |
| 추론 방식 | view별 순차 생성 | view별 순차 생성 + UV refinement | feed-forward, DDIM 30 step | multi-view 생성(Kiss3DGen) → coarse UV 투영 → UV latent denoising (28 step) | feed-forward (T2UV 약 10초) | multi-view 동시 생성 + optimization |
| 추론 시간 | 약 15분 (A6000) | 95초 (TEXGen 논문 측정) | 10초 (A100) | 노트에 미기재 | 약 0.17분 (T2UV) | 16 view·512² 기준 diffusion 187초 + inverse rendering |
| multi-view consistency 확보 | 영역 분할(New/Update/Keep) + 자동 view 선택 | 대칭 2 view 동시 샘플링 + UV space hole filling | UV space에서 한 번에 생성 | UV latent에서 직접 denoise (projection 결과는 soft conditioning) | UV space에서 한 번에 생성 | view-correlated noise + attention bias |
| lighting 처리 | 한계로 남음 (shading baked-in) | UVHD로 제거 | 학습 data의 diffuse color 사용 | 학습 시 조명이 다른 coarse UV를 섞어 조건으로만 사용, GT는 albedo | - | 반사 등 view-dependent 효과가 baked-in되는 한계 |
| 한계 | shading 포함, 일관성 보장 안 됨 | coarse 단계의 multi-face problem, PBR 미지원 | 정렬된 condition image만 가능, PBR 미지원 | 심하게 조각난 UV layout에서 blur/color leakage | SMPL UV 기반 human에 한정, 의상 무늬 불일치 가능 | macro geometry 미개선, `w` 수동 tuning |

## 2. UV 직접 생성 계열 : 3D 정보를 넣는 방식

UV space에서 직접 생성하면 UV seam 때문에 "surface에서는 인접하나 UV에서는 먼 texel"이 생기는 점이 공통 문제이고, 논문마다 이를 다르게 푼다.

| 논문 | 3D 정보 주입 방식 | seam 문제 대응 | 적용 범위 |
|---|---|---|---|
| TEXGen | Hybrid 2D-3D block (UV conv + 3D point cloud serialized attention) | point space에서 global consistency 확보 | 일반 object |
| DirectUV | CCM 좌표를 RoPE에 넣음 (SAPE, Multi-Level SAPE) | attention 자체가 surface 상 상대 거리 `S_j - S_i`를 기준으로 동작 | 일반 object |
| TexDreamer | 없음 (고정된 SMPL UV 구조를 LoRA로 학습) | UV layout이 항상 같으므로 문제가 되지 않음 | SMPL UV 기반 human |

- TEXGen, DirectUV : 임의의 UV layout을 다루기 위해 3D 정보를 명시적으로 넣음
- TexDreamer : 고정된 UV layout을 전제하므로 3D 정보 없이 가능하지만, 범위가 human으로 한정됨

## 3. 계보와 연결 관계

**View 기반 → UV 직접 생성**
- Text2Tex의 한계(texture에 shading 포함) → Paint3D의 lighting-less texture 접근(UVHD)의 동기
- Paint3D의 한계(2단계 pipeline의 error 누적, 초기 texture 생성에 per-instance optimization 필요) → TEXGen이 feed-forward 단일 모델로 해결
- TEXGen의 한계(핵심 attention은 UV grid 기준이고 3D 정보는 보조 module) → DirectUV가 3D surface 좌표를 positional encoding에 넣어 해결
  - DirectUV가 GSO에서 TEXGen과 직접 비교 : FID 44.09 → 39.17, PSNR 24.45 → 25.13

**DirectUV는 projection을 완전히 버리지는 않음**
- inference에서 multi-view generator(Kiss3DGen)로 만든 이미지를 UV에 projection하여 coarse UV를 얻음
- 다만 coarse UV는 생성 target이 아니라 conditioning 신호이며, 학습 시 0/1/2/4 view를 무작위로 사용하여 blank UV에서도 동작
- Text2Tex, Paint3D처럼 projection 결과를 inpainting의 target으로 취급하는 방식과 구분됨

**Lighting 문제**
- Text2Tex(한계로 남김) → Paint3D(UVHD로 제거) → TEXGen(학습 data의 diffuse color 사용) → DirectUV(조명이 다른 coarse UV를 일부러 섞어 학습, GT는 albedo)

**PBR**
- Paint3D, TEXGen은 PBR map 생성을 한계 또는 향후 과제로 남김
- Generative Detail Enhancement는 새로 생성하는 대신, 기존 PBR material을 inverse rendering으로 강화하는 별도 목표
  - 2D diffusion의 multi-view consistency 문제를 training 없이 해결한다는 점에서 Text2Tex, Paint3D와 같은 문제의식

**TexDreamer**
- 일반 object 계열과 별개로 human 도메인에서 "UV data 부족"을 소량 sample + LoRA로 해결
- DirectUV는 TexDreamer를 TEXGen과 함께 "UV space 직접 생성" 방식으로 분류
- 일반 object 계열(TEXGen, DirectUV)과 달리 pretrained T2I의 generalization을 UV 구조에 적응시키는 방식

## 4. 핵심 trade-off

| 관점 | 2D diffusion 활용, 반복 생성/optimization (Text2Tex, Paint3D, Detail Enhancement) | UV 직접 생성 (TEXGen, DirectUV) |
|---|---|---|
| 학습 비용 | 낮음 (pretrained model 재사용, Paint3D는 Position Encoder만 학습) | 높음 (TEXGen 700M model·120K mesh, DirectUV 100K mesh·A800 8장·170K step) |
| 추론 속도 | 느림 (view별 반복, optimization) | 빠름 (TEXGen 약 10초). DirectUV는 multi-view 생성 단계가 추가됨 |
| 3D 일관성 | 별도 장치 필요 (mask, noise, attention bias 등) | 구조적으로 확보 (TEXGen : UV + 3D point, DirectUV : SAPE) |
| Janus problem | 발생 가능 | 3D data 학습으로 회피 |
| 표현의 다양성 | 2D prior 덕분에 넓음 | 학습 data 분포에 의존 |
| UV 품질 민감도 | 낮음 (view에서 생성) | 높음 (DirectUV는 조각난 UV layout에서 열화) |
| occluded 영역 | hole이 생겨 별도 inpainting 필요 (Paint3D의 UV Inpainting) | UV 전체를 생성하므로 자연스럽게 채움 |

**TexDreamer의 위치** : UV 직접 생성이지만 human 도메인에 한정하고 소량 data + LoRA로 학습하므로, 위 표의 일반 object용 UV 직접 생성과 학습 비용·범위가 다르다.

## 5. 참고 : 정량 수치 비교 시 주의

각 논문이 서로 다른 test set과 평가 설정을 사용하므로, 논문 간 수치를 직접 비교할 수 없다. 같은 논문 안의 비교만 유효하다.

| 논문 | test set | 비교 대상 |
|---|---|---|
| Text2Tex | Objaverse 410개 | Text2Mesh, CLIPMesh, Latent-Paint |
| Paint3D | Objaverse 301개 + 실제 30개 | Latent-Paint, TEXTure, Text2Tex |
| TEXGen | Objaverse 400개 | TEXTure, Text2Tex, Paint3D |
| DirectUV | Google Scanned Objects 전체 (OOD) | TEXGen, FlexPainter, Hunyuan3D-2.1, UniTEX |
| TexDreamer | SMPL T-pose 렌더링 / ATLAS test set | Text2Tex, TEXTure, Latent-Paint, Fantasia3D, SMPLitex 등 |
| Detail Enhancement | 정량 평가 없음 (정성 비교 중심, Kettle 예시) | DreamMat, Paint-it, TexPainter 등 |

- KID 단위도 논문마다 다름 (Text2Tex, Paint3D : ×10⁻³, TEXGen : ×10⁻⁴)

## 6. 참고 노트

- [Text2Tex](Text2Tex.md)
- [Paint3D](Paint3D.md)
- [TEXGen](TEXGen.md)
- [DirectUV](DirectUV.md)
- [TexDreamer](TexDreamer.md)
- [Generative Detail Enhancement](Generative_Detail_Enhancement.md)
