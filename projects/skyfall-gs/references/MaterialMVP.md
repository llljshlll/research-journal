# MaterialMVP: Illumination-Invariant Material Generation via Multi-view PBR Diffusion

3D mesh와 reference image를 입력으로 받아  
**albedo / metallic / roughness(PBR)** 를 multi-view 일관적으로 생성하는 one-stage 방법.

핵심 목표는 세 가지다.

1. **multi-view consistency**
2. **illumination-invariant material generation**
3. **albedo와 MR(metallic+roughness) 정렬**

---

## Overall Pipeline

입력:
- 3D mesh
- reference image

출력:
- albedo map
- metallic map
- roughness map

전체 흐름:

1. mesh를 여러 시점에서 렌더링해 `normal`과 `position` 조건을 만든다
2. multi-view diffusion UNet이 각 view의 PBR 표현을 공동 생성한다
3. reference-conditioned training으로 조명과 시점 변화에 덜 민감하게 학습한다
4. 생성된 multi-view 결과를 UV texture로 bake해 mesh에 입힌다

<img src="../../../docs/assets/projects/skyfall-gs/references/materialMVP_pipeline.png" width="760">

---

## 1. Input Representation

MaterialMVP의 입력은 단순한 image latent 하나가 아니다.  
각 view마다 mesh에서 얻은 기하 정보를 함께 넣어, 생성이 geometry와 정렬되도록 만든다.

각 시점에서 렌더링되는 조건:

- `normal map`
- `position map`

이 둘은 latent space로 인코딩된 뒤, noisy PBR latent와 **채널 방향으로 결합**된다.

개념적으로 입력은:

`x = [z_t^{pbr} ; z^{normal} ; z^{position}]`

으로 쓸 수 있다.

채널 수는:

`4 + 4 + 4 = 12`

가 되어, 각 view 입력은 `12-channel` 조건을 갖는다.

중요한 점은:

- view는 별도 축으로 유지된다
- `normal`, `position`은 view를 합치는 정보가 아니라 각 view 내부의 geometry condition이다
- 따라서 multi-view 구조와 geometry condition이 동시에 유지된다

PBR 생성 대상은 보통 두 종류의 재질 표현으로 다뤄진다.

- `albedo`
- `MR` (`metallic`, `roughness`)

<img src="../../../docs/assets/projects/skyfall-gs/references/albedo_mr.png" width="400">


즉 모델은 단일 RGB texture가 아니라, 서로 의미가 다른 material 표현을 함께 생성한다.

---

## 2. UNet Architecture
<img src="../../../docs/assets/projects/skyfall-gs/references/materialMVP_SD.png" width="760">

기본 backbone은 latent diffusion UNet이지만, 실제 동작은 multi-view PBR generation에 맞게 확장된다.

핵심 아이디어는 두 가지다.

1. 여러 view가 서로 정보를 주고받도록 만든다
2. `albedo`와 `MR`을 구분하면서도 공간적으로 정렬되게 만든다

기본 transformer block이

`self-attn -> cross-attn -> FFN`

이라면, MaterialMVP에서는 이를 더 풍부한 조건 결합 구조로 확장한다.

개념적으로는:

`material-aware self-attn -> reference attn -> multiview attn -> cross-attn -> dino attn -> FFN`

<img src="../../../docs/assets/projects/skyfall-gs/references/UNet.png" width="760">

### 2.1 Material-aware Representation

`albedo`와 `MR`은 통계적 성질과 의미가 다르다.

- albedo는 주로 intrinsic color를 담당한다
- MR은 표면 반사 특성을 담당한다

이를 같은 방식으로 생성하면 정렬 오류나 material artifact가 생기기 쉽다.  
그래서 MaterialMVP는 두 표현을 구분해 다루되, 완전히 분리된 두 네트워크를 쓰지는 않는다.

즉 구조는:

- **shared backbone**
- **PBR-aware separation**

의 조합이다.

이 덕분에 `albedo`와 `MR`은 서로 다른 역할을 유지하면서도, 동일한 geometry와 view 구조 안에서 정렬된다.

### 2.2 Reference Conditioning

reference image는 주로 **albedo 쪽 정렬 기준**으로 작동한다.

그리고 이렇게 얻은 reference-aligned 신호가 residual 형태로 전체 표현에 전파되므로,  
MR도 간접적으로 정렬 이득을 받는다.

즉 reference는 "albedo에만 완전히 제한"된다기보다,

- **직접 효과는 albedo 중심**
- **간접 효과는 shared representation을 통해 MR까지 전파**

된다고 보는 것이 더 정확하다.

### 2.3 Multi-view Interaction

여러 view가 독립적으로 생성되면 앞면과 옆면, 뒷면 사이의 재질이 쉽게 어긋난다.  
이를 막기 위해 view 사이 attention을 넣어 서로의 정보를 읽게 한다.
<img src="../../../docs/assets/projects/skyfall-gs/references/MV_attn.png" width="760">

하지만 MaterialMVP의 핵심은 단순히 view를 섞는 것에 그치지 않는다.  
`position map`으로부터 얻은 3D 위치 정보를 이용해, **서로 비슷한 표면 위치끼리 더 잘 대응되도록** 만든다.
<img src="../../../docs/assets/projects/skyfall-gs/references/MV_attn_voxel.png" width="760">

즉 "같은 픽셀 위치"가 아니라 "비슷한 3D 위치"를 기준으로 view consistency를 강화한다.

### 2.4 DINO Conditioning

reference image의 의미 정보를 더 안정적으로 주입하기 위해, image-level semantic feature도 함께 사용한다.

이 보조 조건은:

- texture의 저수준 색/패턴 복사에만 의존하지 않고
- reference의 의미적 material cue를 유지하도록

돕는다.

따라서 UNet은

- geometry condition
- reference image condition
- semantic image feature

를 함께 사용해 PBR texture를 생성한다.

---

## 3. Consistency-Regularized Training

논문이 다루는 핵심 문제는 두 가지다.

1. **view sensitivity**
2. **illumination entanglement**

즉 reference가 조금만 달라져도 결과가 흔들리고,  
reference의 조명이 albedo나 MR에 섞여 들어갈 수 있다.

이를 줄이기 위해 학습 시 단일 reference 대신 **reference pair** `(I_1, I_2)`를 사용한다.

- 두 이미지는 같은 object를 보지만
- 시점이나 조명이 약간 다르다

이때 같은 latent target에 대해 두 조건이 유사한 예측을 내도록 제약한다.

기본 diffusion loss:

`L_{pbr} = E[ ||epsilon - epsilon_theta(z_t, t, c(I_1))||_2^2 ]`

consistency loss:

`L_{cons} = E[ ||epsilon_theta(z_t, t, c(I_1)) - epsilon_theta(z_t, t, c(I_2))||_2^2 ]`

최종 loss:

`L = (1 - lambda)L_{pbr} + lambda L_{cons}`

여기서 `lambda = 0.1`.

이 설계의 의미는 명확하다.

- `L_pbr`는 생성 품질을 유지하고
- `L_cons`는 reference perturbation에 대한 안정성을 학습한다

결과적으로 모델은 조명과 시점 변화에 덜 민감한 material prior를 얻게 된다.

---

## 4. From Multi-view Outputs to Texture Maps
<img src="../../../docs/assets/projects/skyfall-gs/references/texture.png" width="760">
UNet이 직접 만드는 것은 최종 UV texture map이 아니라,  
각 시점에서 보이는 **multi-view material image**다.

이 결과를 실제 mesh texture로 바꾸기 위해 후처리가 필요하다.

전체 과정은 다음과 같다.

1. mesh의 UV 좌표를 준비한다
2. 여러 view에서 생성된 `albedo`와 `MR` 이미지를 얻는다
3. 각 view 결과를 UV atlas로 역투영한다
<img src="../../../docs/assets/projects/skyfall-gs/references/pixel_sampling.png" width="760">

4. 겹치는 영역을 병합하고 비어 있는 영역을 메운다
<img src="../../../docs/assets/projects/skyfall-gs/references/pixel_select.png" width="760">

5. 최종 `albedo`, `metallic`, `roughness` texture를 저장한다

즉 MaterialMVP는 multi-view image generation과 texture baking을 연결해,

- view-consistent generation
- geometry-aligned projection
- mesh-ready PBR texture output

까지 완성하는 파이프라인이다.

---

## Summary

MaterialMVP는 mesh에서 얻은 `normal`/`position` 조건과 reference image를 함께 사용해,  
multi-view diffusion으로 `albedo`와 `MR`을 공동 생성한다.

핵심은:

- geometry-aware `12-channel` 입력
- `albedo`와 `MR`을 구분하는 shared UNet
- position-aware multi-view attention
- albedo 중심 reference alignment와 semantic conditioning
- reference pair 기반 consistency-regularized training
- multi-view 결과를 UV texture로 bake하는 후처리

이 조합을 통해 조명에 덜 민감하면서도 view-consistent한 PBR texture 생성을 목표로 한다.

---

## Reference

- Paper: `MaterialMVP: Illumination-Invariant Material Generation via Multi-view PBR Diffusion` (arXiv:2503.10289v2)
- Project: https://github.com/ZebinHe/MaterialMVP
