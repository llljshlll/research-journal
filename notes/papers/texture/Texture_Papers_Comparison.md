# Texture 생성 논문 비교

> 일반 object 5편(Text2Tex, Paint3D, TEXGen, Meta 3D TextureGen, Generative Detail Enhancement)과 human/face 4편(TexDreamer, UV-IDM, UltrAvatar, FreeUV)의 접근 방식 비교

## 1. 일반 object 대상

| 항목 | Text2Tex | Paint3D | TEXGen | Meta 3D TextureGen | Generative Detail Enhancement |
|---|---|---|---|---|---|
| 목표 | mesh의 texture 생성 | mesh의 texture 생성 | mesh의 texture 생성 | mesh의 texture 생성 | 기존 PBR material의 detail 강화 |
| 처리 흐름 | View → UV projection | View → UV refinement | UV 직접 생성 | View(image space) → UV inpainting | View 생성 → inverse rendering |
| 입력 | mesh + text | mesh + text/image | mesh + text + single-view image | mesh + text | mesh + 기존 material + text |
| 출력 | UV texture (RGB) | UV texture (RGB, lighting 제거, 2K) | 1024×1024 UV texture | 1024×1024 UV texture (선택적으로 4k) | PBR material (albedo, roughness, normal) |
| 2D diffusion 사용 방식 | Depth2Image + inpainting | depth-aware diffusion + ControlNet 기반 UV 모델 | 사용 안 함 (UV space에서 직접 학습) | Emu 유사 구조 text-to-image 모델을 fine-tuning | Stable Diffusion 1.5 + ControlNet (tile, normal) |
| geometry 조건 | depth | depth + position map(UV) | position map(UV) + 3D point | position + normal render (4 view) | normal render |
| 추가 학습 | 불필요 | Position Encoder(`tau_p`)만 학습, 기존 diffusion은 freeze | 700M parameter model 전체 학습 | Stage I/II를 각각 fine-tuning (260k object, H100 32장) | 불필요 (training-free) |
| 추론 방식 | view별 순차 생성 (약 15분) | view별 순차 생성 + UV refinement | feed-forward (약 10초) | feed-forward, diffusion 2회 (약 19초) | multi-view 동시 생성 + optimization |
| multi-view consistency 확보 | 영역 분할(New/Update/Keep) + 자동 view 선택 | UV space에서 hole filling | UV space에서 한 번에 생성 | 4 view 동시 생성 + position/normal 조건 + incidence 가중 blending | view-correlated noise + attention bias |
| lighting 처리 | 한계로 남음 (shading baked-in) | UVHD로 제거 | 학습 data의 diffuse color 사용 | 균일 조명 렌더링을 target으로 사용, PBR map은 미지원 | 반사 등 view-dependent 효과가 baked-in되는 한계 |

## 2. Human/Face 대상

| 항목 | TexDreamer | UV-IDM | UltrAvatar | FreeUV |
|---|---|---|---|---|
| 대상 | 3D human (SMPL UV) | 3D face (BFM UV) | 3D face (FLAME UV) | 3D face (FLAME UV) |
| 입력 | text 또는 image | 단일 in-the-wild 이미지 | text 또는 단일 이미지 | 단일 in-the-wild 이미지 |
| 출력 | 1024×1024 UV texture | 256×256 UV texture | PBR texture (diffuse, normal, specular, roughness), 2K | 512×512 UV texture |
| 핵심 방법 | SD 2.1에 LoRA fine-tuning(T2UV) + feature translator(I2UV) | incomplete UV texture를 condition(ICM)으로 쓰는 LDM | DCE(조명 제거) + AGT-DM(photometric/edge guidance) | 외형/구조 network를 따로 학습하고 추론 시 조합(Cross-Assembly) |
| 학습 data | 소량 sample texture (약 100개에서 성능 포화) + ATLAS (50k texture 합성) | BFM-UV (StyleGAN2로 합성한 약 80K texture) | 3DScan 사이트의 face scan (PBR 포함) | FFHQ 이미지 33,419장 (UV ground truth 없음) |
| UV ground truth 필요 여부 | 필요 (소량 + 합성) | 필요 (합성 dataset) | 필요 (scan 기반 PBR) | 불필요 |
| 사용하는 pretrained 모델 | Stable Diffusion 2.1, CLIP | LDM (VAE, U-Net 직접 학습) | SD-2.1-base, SDXL, BiSeNet, MICA, EMOCA | Stable Diffusion 1.5, CLIP, ControlNet |
| 추론 시간 | 약 10초 (0.17분) | 6초 | 2분 이내 (text 입력 기준) | 4.75초 |
| 조명 처리 | - | 조명 attribute를 유지하여 원본에 가까운 texture 생성 | DCE로 diffuse color 추출 (self-attention의 query/key 교체) | 별도 제거 없음, Lab color transfer로 색조 보정 |
| 한계 | SMPL UV 기반 human에 한정, 의상 무늬 불일치 가능 | 합성 data의 FFHQ bias, StyleGAN2 편집 중 identity 손실 | 별도 한계 서술 없음 | 미세 요소(액세서리, 잡티)의 위치 어긋남, segmentation 실패에 취약 |

## 3. 계보와 연결 관계

**일반 object**
- Text2Tex의 한계(texture에 shading 포함) → Paint3D의 lighting-less texture 접근(UVHD)의 동기
- Paint3D의 한계(2단계 pipeline의 error 누적, test-time optimization 필요) → TEXGen이 feed-forward 단일 모델로 해결
- Meta 3D TextureGen도 feed-forward이나 접근이 다름
  - TEXGen은 UV space에서 diffusion을 직접 수행, Meta 3D TextureGen은 image space에서 multi-view를 생성한 뒤 UV space에서 inpainting
  - Meta 3D TextureGen은 depth 대신 position/normal render를 조건으로 사용하여 view 간 대응(position)과 미세한 geometry(normal)를 제공
- Generative Detail Enhancement는 texture를 새로 만드는 대신 기존 material의 detail을 강화하는 별도 목표
  - 2D diffusion의 multi-view consistency 문제를 training 없이 해결한다는 점에서 Text2Tex, Paint3D와 같은 문제의식

**Human/Face**
- Meta 3D TextureGen이 TexDreamer를 "UV space에서 직접 diffusion을 수행하나 human에 한정되어 임의 object로 확장 불가"한 방법으로 언급
- UV-IDM → FreeUV
  - UV-IDM : StyleGAN2로 합성한 UV dataset(ground truth)에 의존하고, 원본 이미지에서 incomplete texture를 뽑아 LDM에 조건으로 줌
  - FreeUV : UV ground truth 자체를 쓰지 않고 UV-IDM과 직접 비교하여 같은 이점(수 초 추론)에 더해 data 의존성 제거
- UltrAvatar의 조명 제거(DCE)는 Paint3D의 UVHD와 같은 문제(texture에 포함된 조명)를 다루나, 학습 대신 diffusion의 self-attention feature를 이용한 training-free 방식
- TexDreamer(소량 sample + LoRA)와 FreeUV(UV ground truth 없음)는 모두 "UV data 부족"을 해결하려는 시도이나 방향이 다름
  - TexDreamer : pretrained T2I의 generalization을 소량 data로 UV 구조에 적응
  - FreeUV : 외형과 구조를 서로 다른 domain에서 따로 학습하여 추론 시 결합

## 4. 핵심 trade-off

**일반 object**

| 관점 | 2D diffusion 활용, 반복 생성/optimization (Text2Tex, Paint3D, Detail Enhancement) | Feed-forward 학습 (TEXGen, Meta 3D TextureGen) |
|---|---|---|
| 학습 비용 | 낮음 (pretrained model 재사용) | 높음 (대규모 3D dataset, TEXGen 700M model, Meta 3D TextureGen 260k object) |
| 추론 속도 | 느림 (view별 반복, optimization) | 빠름 (TEXGen 약 10초, Meta 3D TextureGen 약 19초) |
| 3D 일관성 | 별도 장치 필요 (mask, noise, attention bias 등) | 구조적으로 확보 (TEXGen : UV + 3D point 결합, Meta 3D TextureGen : multi-view 동시 생성 + position/normal 조건) |
| Janus problem | 발생 가능 | 3D data 학습으로 회피 |
| 표현의 다양성 | 2D prior 덕분에 넓음 | 학습 data 분포에 의존 |

**Human/Face : UV data 확보 방식**

| 방식 | 대표 논문 | 장점 | 단점 |
|---|---|---|---|
| 합성 UV dataset로 학습 | UV-IDM, TexDreamer(ATLAS) | 직접 supervision 가능, 빠른 추론 | 합성 data의 편향(StyleGAN2, FFHQ), 구축 비용 |
| scan 기반 PBR dataset로 fine-tuning | UltrAvatar | PBR texture 생성, 정렬이 좋은 학습 data | 고품질 scan 확보 필요 |
| UV ground truth 없이 학습 | FreeUV | data 의존성 제거 | 미세 요소 위치 어긋남, segmentation 품질 의존 |

## 5. 참고 노트

- [Text2Tex](Text2Tex.md)
- [Paint3D](Paint3D.md)
- [TEXGen](TEXGen.md)
- [Meta 3D TextureGen](Meta_3D_TextureGen.md)
- [Generative Detail Enhancement](Generative_Detail_Enhancement.md)
- [TexDreamer](TexDreamer.md)
- [UV-IDM](UV-IDM.md)
- [UltrAvatar](UltrAvatar.md)
- [FreeUV](FreeUV.md)
