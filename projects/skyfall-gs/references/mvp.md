# MaterialMVP: Code-Accurate Notes

`MaterialMVP`는 `3D mesh + reference image`를 입력으로 받아,  
멀티뷰 일관성을 유지하는 `albedo`와 `MR(metallic-roughness)` 이미지를 생성한 뒤 이를 다시 UV texture로 베이킹하는 시스템이다.

이 문서는 논문 개념도를 그대로 옮기는 문서가 아니라, **현재 저장소 코드 기준으로 실제 구현이 어떻게 되어 있는지**를 정리한 메모다.

---

## 1. 현재 코드 기준 전체 파이프라인

엔트리 포인트:

- [demo.py](/home/diglab/workspace/projects/MaterialMVP/demo.py)
- [textureGenPipeline.py](/home/diglab/workspace/projects/MaterialMVP/textureGenPipeline.py)

실행 흐름:

1. mesh 로드
2. `xatlas`로 UV unwrap
3. view selection
4. 선택된 view에서 `normal map`, `position map` 렌더링
5. multi-view diffusion 모델로 `albedo`, `mr` 뷰 이미지 생성
6. super-resolution
7. 각 view를 UV texture space로 back-project
8. cosine-weighted bake
9. 빈 영역 UV inpaint
10. OBJ/GLB 저장

중요:

- 현재 코드 기본값은 `use_remesh=False`
- 즉 기본 동작은 **원본 mesh geometry를 유지하고 texture만 생성**하는 쪽이다
- 단, UV unwrap은 다시 수행하므로 UV는 새로 잡힌다

---

## 2. 출력은 무엇인가

현재 구현에서 diffusion 모델이 직접 생성하는 것은:

- `albedo` multi-view images
- `mr` multi-view images

여기서 `mr`는 metallic과 roughness를 함께 담는 맵이다.

즉 문서나 그림에서 종종 `albedo / metallic / roughness`로 나눠 적지만,  
**현재 코드의 diffusion 출력 채널은 `albedo`와 `mr` 두 종류**다.

관련 코드:

- [textureGenPipeline.py](/home/diglab/workspace/projects/MaterialMVP/textureGenPipeline.py)
- [utils/multiview_utils.py](/home/diglab/workspace/projects/MaterialMVP/utils/multiview_utils.py)

---

## 3. Base Model

학습/추론 베이스는 `Stable Diffusion 2.1` 계열이다.

설정:

- [cfgs/v1.yaml](/home/diglab/workspace/projects/MaterialMVP/cfgs/v1.yaml)

핵심 항목:

- `stable_diffusion_config.pretrained_model_name_or_path: stabilityai/stable-diffusion-2-1`
- `noise_in_channels: 12`

즉 원본 SD2.1을 그대로 쓰는 것이 아니라, **입력 채널과 attention 구조를 바꾼 UNet**을 사용한다.

---

## 4. 코드 기준 모델 구조

핵심 파일:

- [materialmvp/pipeline.py](/home/diglab/workspace/projects/MaterialMVP/materialmvp/pipeline.py)
- [materialmvp/model.py](/home/diglab/workspace/projects/MaterialMVP/materialmvp/model.py)
- [materialmvp/modules.py](/home/diglab/workspace/projects/MaterialMVP/materialmvp/modules.py)

전체 개념:

```text
reference image
  -> VAE latent
  -> DINOv2 feature

normal maps
  -> VAE latent

position maps
  -> VAE latent
  -> voxel index / pose-aware alignment

learned PBR tokens
  -> albedo token
  -> mr token

all conditions
  -> modified SD2.1 UNet
  -> multi-view albedo images + multi-view mr images
```

---

## 5. UNet에서 실제로 바뀐 부분

### 5.1 Input channel 확장

기본 SD UNet 입력은 noisy latent만 받지만, 현재 구현은 여기에 geometry condition latent를 같이 넣는다.

대략:

```text
sample = [noisy_latent | normal_latent | position_latent]
```

그래서 [cfgs/v1.yaml](/home/diglab/workspace/projects/MaterialMVP/cfgs/v1.yaml) 에 `noise_in_channels: 12`가 들어가고,  
[train.py](/home/diglab/workspace/projects/MaterialMVP/train.py) 에서 `conv_in`을 새로 만들어 입력 채널 수를 늘린다.

### 5.2 UNet block 교체

원래 `UNet2DConditionModel` 내부 transformer block들을  
`Basic2p5DTransformerBlock`으로 바꾼다.

관련 코드:

- [materialmvp/modules.py](/home/diglab/workspace/projects/MaterialMVP/materialmvp/modules.py)

즉 backbone을 새로 설계한 게 아니라, **SD2.1 UNet의 attention block들을 2.5D 멀티뷰용으로 확장**한 형태다.

### 5.3 활성화된 기능 플래그

현재 구현은 아래를 모두 켜 둔다.

- `use_ma = True`
- `use_ra = True`
- `use_mda = True`
- `use_dino = True`
- `use_position_rope = True`
- `use_learned_text_clip = True`
- `use_dual_stream = True`

즉 논문 그림의 핵심 요소들은 현재 코드에서도 대부분 활성화되어 있다.

---

## 6. Attention 구조: 그림과 코드의 차이

논문 그림은 기능 블록을 병렬 분기처럼 단순화해서 그린다.  
하지만 **현재 코드 구현은 대부분 residual add를 순차적으로 누적하는 구조**다.

코드 기준 실행 순서는 대략 아래와 같다.

```text
norm
-> material-aware self attention (attn1)
-> + reference attention (attn_refview)
-> + multiview attention (attn_multiview)
-> + cross attention to text / learned PBR tokens (attn2)
-> + dino attention (attn_dino)
-> + FFN
```

즉 제가 이전에 설명한

`material attn -> reference attn -> mv attn -> cross attn -> dino attn -> FFN`

은 **코드 실행 순서 기준으로 맞다**.

반면 그림의

- `Z_albedo`, `Z_mr` 분기
- `MCAA`
- `Reference Branch`
- `MV Attn`

은 이 순차 구조를 개념적으로 압축해서 보여주는 표현이다.

### 6.1 residual add가 의미하는 것

여기서 residual add는 말 그대로 같은 shape의 tensor를 원소별로 더하는 것이다.

예:

```python
h = h + branch_output
```

즉 concat이 아니라, 기존 표현 위에 각 attention branch가 만든 보정값을 누적한다.

### 6.2 MCAA와 plain self-attn의 관계

현재 구현에서는 일반 self-attention 자리가 `MDA / material-aware self-attention`으로 대체된다.

즉:

- 별도로 plain self-attn이 하나 더 있는 게 아님
- `attn1` 자체가 확장된 self-attn 역할을 함

### 6.3 RefAttn은 어디에 붙는가

그림은 RefAttn이 `MCAA 내부 Z_albedo의 K,V`에 직접 들어가는 것처럼 보일 수 있다.  
현재 코드 구현은 그렇게 딱 한 블록 안에 삽입하기보다, **MCAA 이후 별도 reference branch로 계산한 결과를 residual로 더하는 형태**에 가깝다.

다만 기능적으로는 그림과 연결된다.  
왜냐하면 현재 코드도 **reference attention은 albedo branch 기준으로만 사용**하기 때문이다.

관련 주석:

- [materialmvp/modules.py](/home/diglab/workspace/projects/MaterialMVP/materialmvp/modules.py)
  - `Only using albedo features for reference attention`

즉 정리하면:

- 그림: 개념적 표현
- 코드: `attn1` 후 `attn_refview`를 별도 residual branch로 추가
- 하지만 RefAttn이 albedo branch 중심이라는 큰 방향은 맞음

### 6.4 MV / Cross / Dino는 어떻게 합쳐지나

이들도 concat이 아니라 순차적인 residual add다.

개념적으로:

```text
h1 = h0 + MCAA(h0)
h2 = h1 + RefAttn(h1, ref)
h3 = h2 + MVAttn(h1 or h2 계열)
h4 = h3 + CrossAttn(h3, text/pbr token)
h5 = h4 + DinoAttn(h4, dino)
h6 = h5 + FFN(h5)
```

정확한 내부 입력은 block 구현 세부에 따라 조금 다르지만,  
설명 수준에서는 **각 branch가 같은 hidden state 공간에서 residual로 누적된다**고 이해하면 된다.

---

## 7. PBR 채널 분리 방식

현재 구현은 `albedo UNet`과 `mr UNet` 두 개가 따로 있는 구조가 아니다.

대신:

- 하나의 UNet backbone을 공유하고
- tensor를 `B x N_pbr x N_view x ...` 형태로 다루고
- 일부 attention processor와 learned token이 PBR별로 분기된다

즉:

- backbone 공유
- PBR conditioning과 일부 attention 경로만 분화

이다.

이 점은 그림을 볼 때 오해하기 쉽다.  
그림상 `Z_albedo`, `Z_mr`가 완전 독립 분기처럼 보이지만, 코드에서는 **완전 분리 네트워크가 아니라 shared backbone + partial specialization** 구조다.

---

## 8. Learnable Material Embeddings

이전 설명에서 `16 x 1024`라고 적은 건 현재 코드 기준으로 맞지 않는다.

현재 구현은 [materialmvp/modules.py](/home/diglab/workspace/projects/MaterialMVP/materialmvp/modules.py) 에서:

- `learned_text_clip_albedo`
- `learned_text_clip_mr`
- `learned_text_clip_ref`

를 등록한다.

shape은:

- `77 x 1024`

이다.

즉 SD/CLIP text token 길이에 맞춘 learned embedding으로 보는 편이 맞다.

이 learned token들은 일반 문장 prompt 대신 PBR type별 conditioning 역할을 한다.

---

## 9. DINO conditioning

현재 구현은 DINOv2 feature도 사용한다.

관련 코드:

- [materialmvp/modules.py](/home/diglab/workspace/projects/MaterialMVP/materialmvp/modules.py)
- [materialmvp/model.py](/home/diglab/workspace/projects/MaterialMVP/materialmvp/model.py)

흐름:

1. reference image에서 DINO hidden state 추출
2. `image_proj_model_dino`로 cross-attention 차원에 맞게 projection
3. transformer block의 `attn_dino`를 통해 hidden state에 residual add

즉 DINO는 단순 feature concatenation이 아니라 **추가 cross-attention branch**다.

---

## 10. Position-aware multi-view alignment

멀티뷰 attention이 단순히 모든 view 토큰을 다 섞는 것은 아니다.

현재 구현은 `position map`을 바탕으로 다중 해상도의 voxel index를 만들고,  
이를 통해 view 간 비슷한 3D 위치가 서로 attention되도록 유도한다.

관련 코드:

- `calc_multires_voxel_idxs`
- `compute_discrete_voxel_indice`
- `PoseRoPEAttnProcessor2_0`

즉 "2D 이미지 여러 장을 같이 돌리는 것"보다, **3D position 정보를 쓴 2.5D 정렬 attention**에 가깝다.

---

## 11. Consistency-Regularized Training: 현재 코드 기준

핵심 파일:

- [materialmvp/model.py](/home/diglab/workspace/projects/MaterialMVP/materialmvp/model.py)
- [src/data/dataloader/objaverse_loader_forTexturePBR.py](/home/diglab/workspace/projects/MaterialMVP/src/data/dataloader/objaverse_loader_forTexturePBR.py)

현재 데이터셋 로더는 각 object에 대해:

- 서로 다른 lighting 조건의 reference image 2장
- multi-view albedo
- multi-view mr
- multi-view normal
- multi-view position

을 읽는다.

학습 시:

1. reference A로 예측
2. reference B로 같은 target material을 다시 예측
3. 두 prediction끼리도 가깝게 만듦

현재 코드 손실은 대략:

- `albedo_loss`
- `mr_loss`
- `consistency_loss`

최종:

```text
0.85 * (albedo_loss + mr_loss) + 0.15 * consistency_loss
```

즉 문서에 예전처럼 `lambda = 0.1`이라고 고정해서 적는 건 현재 코드 기준으로 정확하지 않다.  
현재 구현상 최종 결합 비율은 `0.15`가 consistency 쪽에 걸려 있다.

---

## 12. 추론 시 실제 생성 흐름

현재 추론에서 일어나는 일:

1. 입력 reference image를 `512 x 512`로 정리
2. 선택된 각 view의 normal / position map 생성
3. diffusion pipeline이 모든 view의 `albedo`, `mr`를 한 번에 생성
4. 결과를 `{"albedo": ..., "mr": ...}`로 분리
5. Real-ESRGAN으로 해상도 향상
6. back-project 후 UV texture bake
7. inpaint 후 저장

관련 코드:

- [utils/multiview_utils.py](/home/diglab/workspace/projects/MaterialMVP/utils/multiview_utils.py)
- [textureGenPipeline.py](/home/diglab/workspace/projects/MaterialMVP/textureGenPipeline.py)

---

## 13. Geometry를 새로 생성하나

아니다. 현재 시스템은 **geometry generation 모델이 아니라 material / texture generation 모델**이다.

즉 diffusion이 새 triangle을 만들어내지는 않는다.

정확히는:

- 바뀌는 것: texture, material map
- 유지되는 것: mesh surface geometry

단, 전처리에서 remesh를 켜면 triangle 배치는 바뀔 수 있다.  
현재 기본값은 `use_remesh=False`이므로, 기본 추론 경로에서는 원본 geometry를 유지한다.

---

## 14. 우리 프로젝트 관점에서 중요 포인트

치아/의료 쪽에서 특히 볼 만한 구현 포인트:

1. reference perturbation에 강한 consistency regularization
2. appearance(`albedo`)와 material(`mr`)을 분리하되 shared backbone으로 묶는 구조
3. position map 기반 multi-view alignment
4. geometry는 유지하고 material만 생성하는 파이프라인

---

## 15. 짧은 정리

현재 코드 기준 MaterialMVP는:

- `Stable Diffusion 2.1` 기반
- input channel 확장(`noise + normal + position`)
- `material-aware self-attn + ref attn + multi-view attn + text attn + dino attn`
- learned PBR tokens(`77 x 1024`)
- DINO conditioning
- position-aware multi-view alignment
- consistency-regularized training
- 최종적으로 `albedo`와 `mr` multi-view를 생성 후 UV bake

로 이해하는 게 가장 정확하다.

---

## 16. `bake_view_selection()`은 어떻게 "잘 칠할 수 있는 view"를 고르나

핵심 함수:

- [utils/pipeline_utils.py](/home/diglab/workspace/projects/MaterialMVP/utils/pipeline_utils.py)
  - `bake_view_selection()`

이 함수는 "이미지 품질이 좋아 보이는 view"를 고르는 것이 아니라,  
**현재 mesh 표면을 얼마나 많이, 얼마나 새롭게 덮을 수 있는지**를 기준으로 view를 고른다.

### 16.1 입력 후보

후보 view는 config에 미리 들어 있다.

- 정면/측면/상하면에 해당하는 6개 기본 view
- 추가로 elevation `+20`, `-20`에서 30도 간격 azimuth view들

즉 후보 view 집합은 고정되어 있고, 그중 일부를 선택하는 구조다.

### 16.2 각 후보 view에서 무엇을 계산하나

각 후보 카메라 `(elev, azim)`에 대해:

1. `render_alpha()`를 돌린다
2. 이때 각 픽셀이 어떤 triangle을 보고 있는지 얻는다
3. `np.unique()`로 그 view에서 보이는 triangle index 집합을 만든다

즉 각 후보 view는 결국 아래 정보로 요약된다.

```text
이 view에서 실제로 관측 가능한 triangle들의 집합
```

### 16.3 왜 triangle "개수"가 아니라 "면적 비율"을 보나

함수는 먼저 mesh 전체의 각 triangle area를 구한다.

```text
face_areas -> total_area -> face_area_ratios
```

이렇게 하면 아주 작은 triangle을 수백 개 더 보는 것보다,  
실제 표면 면적을 많이 덮는 view가 더 유리해진다.

즉 selection 기준은 대략:

```text
새로 커버하는 triangle들의 면적 합
```

이다.

### 16.4 선택 절차

선택은 greedy 방식이다.

1. 처음 6개 canonical view는 무조건 넣는다
2. 그 뒤 남은 후보들에 대해
   - 아직 선택되지 않은 후보를 하나씩 검사
   - 그 후보가 추가로 보여주는 triangle 집합을 계산
   - 그 triangle들의 면적 증가량을 계산
3. 증가 면적이 가장 큰 후보를 하나 추가한다
4. 새로 얻는 면적이 너무 작아지면 중단한다

코드 기준 중단 조건은:

```text
max_inc_area <= 0.01
```

즉 추가 view가 mesh 표면을 거의 새롭게 덮지 못하면 굳이 생성하지 않는다.

### 16.5 왜 이게 texture bake에 유리한가

이 selection의 목적은 "서로 다른 그림 6장 만들기"가 아니라,  
**UV texture를 채우는 데 도움이 되는 view set**을 만드는 것이다.

좋은 bake view는 보통:

- 이미 다른 view가 덮은 부분을 중복해서 보기보다
- 아직 안 보인 triangle을 새롭게 드러내고
- 표면 대부분을 고르게 커버해야 한다

`bake_view_selection()`은 바로 이 관점으로 view를 고른다.

즉 "mesh를 잘 칠할 수 있는 view"라는 말은 정확히는:

```text
UV baking 단계에서 새로운 표면 영역을 더 많이 제공하는 view
```

를 우선한다는 뜻이다.

---

## 17. Mesh-aware baking은 정확히 어떻게 동작하나

핵심 함수:

- [utils/pipeline_utils.py](/home/diglab/workspace/projects/MaterialMVP/utils/pipeline_utils.py)
  - `bake_from_multiview()`
- [DifferentiableRenderer/MeshRender.py](/home/diglab/workspace/projects/MaterialMVP/DifferentiableRenderer/MeshRender.py)
  - `back_project()`
  - `fast_bake_texture()`

핵심 아이디어는 단순하다.

```text
생성된 view 이미지를 그냥 UV에 붙이는 게 아니라,
실제 mesh의 visibility / normal / depth / UV 대응을 이용해
"이 픽셀이 mesh의 어느 UV texel에 해당하는가"를 계산해서 역투영한다.
```

### 17.1 `back_project()`의 입력

각 view 이미지와 그 view의 카메라 파라미터 `(elev, azim)`를 받는다.

즉 이 함수는:

- 생성된 2D view image
- 그 이미지가 어떤 카메라에서 본 것인지
- 현재 mesh의 vertex / face / UV

를 모두 알고 있다.

그래서 이미지 픽셀을 무작정 texture space에 복사하는 게 아니라,  
**카메라-메쉬-UV 관계를 통해 정합된 역투영**을 수행할 수 있다.

### 17.2 우선 현재 view에서 실제로 보이는 표면만 찾는다

`back_project()` 내부에서는 현재 카메라에서 mesh를 rasterize해서:

- `visible_mask`
- per-pixel normal
- per-pixel depth
- per-pixel UV

를 얻는다.

즉 view image의 각 픽셀이 실제 mesh 어느 점을 보고 있는지 다시 계산한다.

이 단계가 중요한 이유는:

- 생성 이미지의 픽셀 중 background는 버리고
- self-occlusion으로 가려진 표면은 제외하고
- 실제 관측 가능한 surface만 texture 후보로 쓰기 때문이다

### 17.3 정면을 보는 픽셀에 더 큰 weight를 준다

`cos_image`는 시선 방향과 surface normal의 cosine similarity다.

대략:

```text
cos = dot(view_direction, surface_normal)
```

의 역할을 한다.

의미:

- 카메라를 정면으로 보는 surface
  - `cos`가 큼
  - 투영 왜곡이 적음
  - 신뢰도 높음
- 비스듬한 surface
  - `cos`가 작음
  - 투영 왜곡, seam, stretch가 커짐
  - 신뢰도 낮음

코드에서는 threshold 아래는 0으로 잘라낸다.

즉 너무 사선인 표면은 아예 bake weight를 주지 않는다.

### 17.4 경계(boundary)와 불안정 영역을 줄인다

코드는 depth 기반 `sketch_image`를 만들고,  
여기에 작은 convolution 커널을 써서 경계 근처 visible region을 더 줄인다.

목적은:

- 실루엣 근처
- depth discontinuity 근처
- rasterization 경계 근처

같은 불안정한 부분을 bake에서 덜 믿도록 하는 것이다.

즉 단순 visible 여부만 보는 것이 아니라,  
**보이기는 하지만 경계라서 믿기 어려운 픽셀**도 제거한다.

### 17.5 실제 UV texel에 어떻게 넣나

현재 기본 bake mode는 `back_sample`이다.

이 모드에서는 UV space의 각 texel이 들고 있는 3D 위치 `self.tex_position`을 다시 현재 카메라로 투영한다.

개념적으로:

```text
UV texel
 -> 그 texel이 대표하는 mesh 상의 3D 위치
 -> 현재 카메라 이미지 좌표로 projection
 -> 해당 위치에서 RGB를 bilinear sample
```

즉 "image pixel -> UV"가 아니라,

```text
UV texel -> camera image
```

방향으로 샘플링하는 셈이다.

이 방식이 중요한 이유는 UV texture space 기준으로 직접 값을 채우기 때문에:

- 각 texel이 정확히 어떤 view image에서 어떤 픽셀을 읽어야 하는지 정해지고
- occlusion/depth check도 함께 가능하고
- mesh geometry와 맞지 않는 픽셀 오염을 줄일 수 있다

### 17.6 depth check는 왜 필요하나

UV texel이 2D image 안에 투영되더라도,  
그 image pixel이 실제로 같은 표면을 보고 있는지는 depth를 비교해 확인해야 한다.

코드는:

- projected texel depth `v_z`
- image rasterization으로 얻은 sampled depth `sampled_z`

를 비교해서, 차이가 작은 경우만 valid로 인정한다.

즉:

```text
같은 화면 위치에 투영되더라도,
실제로는 다른 가려진 표면이면 버린다
```

이게 mesh-aware baking이 단순 projection보다 정확한 핵심 이유 중 하나다.

### 17.7 여러 view를 어떻게 합치나

`bake_from_multiview()`는 각 view마다:

1. `back_project()`로 UV texture 후보 생성
2. cosine map에 view weight와 exponent를 적용
3. `fast_bake_texture()`로 merge

를 수행한다.

여기서 실제 weight는 대략:

```text
weight = candidate_view_weight * (cos_map ^ bake_exp)
```

이다.

즉:

- 기본적으로 중요한 canonical view는 더 큰 prior weight를 가질 수 있고
- 그 안에서도 정면에 가까운 texel이 더 강하게 반영된다

### 17.8 `fast_bake_texture()`는 무슨 merge인가

이 함수는 UV texture 후보들을 weighted average로 합친다.

대략:

```text
texture_merge += texture * cos_weight
trust_map += cos_weight
final = texture_merge / trust_map
```

추가로 이미 거의 다 칠해진 view는 skip한다.

코드상:

```text
if painted_sum / view_sum > 0.99: continue
```

즉 어떤 view가 새롭게 기여하는 texel이 거의 없으면 merge 비용을 아낀다.

정리하면 mesh-aware baking의 핵심은:

1. mesh를 알고 있는 상태에서
2. visibility / depth / normal / UV를 함께 써서
3. 신뢰도 높은 view만 반영하고
4. UV space에서 직접 weighted merge

한다는 점이다.

---

## 18. `uv_inpaint()`는 정확히 어떻게 동작하나

핵심 함수:

- [utils/pipeline_utils.py](/home/diglab/workspace/projects/MaterialMVP/utils/pipeline_utils.py)
  - `texture_inpaint()`
- [DifferentiableRenderer/MeshRender.py](/home/diglab/workspace/projects/MaterialMVP/DifferentiableRenderer/MeshRender.py)
  - `uv_inpaint()`

베이킹이 끝나도 UV texture에는 보통 빈 영역이 남는다.

이유:

- 어떤 texel은 어느 선택된 view에서도 보이지 않았거나
- 사선이라서 weight가 0이 되었거나
- depth/경계 체크에서 제거되었기 때문

그래서 최종 texture는:

- 채워진 영역
- 비어 있는 영역

을 함께 가진다. 이때 `mask`가 바로 "어디가 채워졌는가"를 나타내는 신뢰 맵이다.

### 18.1 입력 mask는 무엇인가

`bake_from_multiview()`는 `ori_trust_map > 1e-8`를 반환한다.

즉 mask는 대략:

- `1`: 적어도 하나의 view가 유효하게 칠한 texel
- `0`: 아직 아무 view도 신뢰 있게 칠하지 못한 texel

이다.

이 mask가 `texture_inpaint()`를 거쳐 `uv_inpaint()`로 들어간다.

### 18.2 1단계: mesh vertex connectivity 기반 보간

기본 옵션은 `vertex_inpaint=True`다.

이때 `uv_inpaint()`는 먼저:

```text
meshVerticeInpaint(texture_np, mask, vtx_pos, vtx_uv, pos_idx, uv_idx)
```

를 호출한다.

이건 단순 2D 이미지 인페인팅 전에,  
**mesh vertex와 triangle 연결 정보를 이용해 UV 빈 영역을 먼저 메워보는 단계**다.

의미적으로는:

- 같은 mesh surface 위에서 인접한 texel들은 비슷한 값을 가질 가능성이 높고
- triangle adjacency를 알면 seam/연결성을 무시한 2D 보정보다 더 surface-aware한 채움이 가능하다

즉 이 단계는:

```text
UV 이미지 한 장만 보고 메꾸는 게 아니라,
mesh의 연결성을 힌트로 먼저 빈 영역을 줄이는 단계
```

다.

### 18.3 2단계: OpenCV Navier-Stokes inpaint

그 다음 남은 빈 영역은 OpenCV의:

- `cv2.inpaint(..., cv2.INPAINT_NS)`

로 채운다.

여기서 `255 - mask`가 실제 인페인트 대상 영역이 된다.

즉:

- mask가 255인 곳: 이미 신뢰 가능한 texture 있음
- mask가 0인 곳: 비어 있으므로 메워야 함

이다.

Navier-Stokes inpaint는 주변 색/구조를 따라 빈 영역을 부드럽게 연장하는 2D 이미지 인페인팅 방식이다.

### 18.4 왜 두 단계를 같이 쓰나

만약 OpenCV inpaint만 쓰면:

- UV seam 구조를 무시하고
- 단순히 2D 이웃 정보만 이용해 채우게 된다

그러면 실제 mesh surface 상으로는 가까운 영역인데 UV상 멀리 떨어진 경우를 잘 활용하지 못한다.

반대로 mesh connectivity 기반 단계만으로는 넓은 hole이나 복잡한 빈 영역을 완전히 메우기 어려울 수 있다.

그래서 현재 구현은:

1. mesh-aware vertex inpaint로 구조적인 빈칸을 먼저 줄이고
2. 2D inpaint로 남은 작은 hole을 마감

하는 하이브리드 전략을 쓴다.

### 18.5 이 단계의 한계

이건 새로운 texture를 "정답처럼 생성"하는 단계가 아니다.

즉 `uv_inpaint()`는:

- 신뢰도 높은 bake 결과를 보존하면서
- 비어 있는 곳을 주변 정보로 자연스럽게 메꾸는 후처리

에 가깝다.

그래서:

- 큰 가려진 영역
- reference에 전혀 단서가 없는 뒷면
- 반복 패턴이 중요한 재질

에서는 완전한 정답 복원보다, seam 감소와 hole filling 역할로 이해하는 게 맞다.

---

## Reference

- Paper: `MaterialMVP: Illumination-Invariant Material Generation via Multi-view PBR Diffusion`
- Repo: https://github.com/ZebinHe/MaterialMVP
