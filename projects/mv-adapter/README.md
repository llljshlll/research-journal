# MV-Adapter (Mesh-conditioned Multi-view Generation)

## Overview

본 실험은  
**mesh 기반 geometry 정보와 single realistic reference image를 입력으로 하여,  
구조적·시각적으로 일관된 multi-view 이미지를 생성하는 것**을 목표로 한다.

Nano Banana는 realistic reference image 생성을 위한 전처리 단계로 사용되었으며,  
본 문서의 초점은 **MV-Adapter를 이용한 multi-view consistency 생성 과정**에 있다.

---

## Pipeline

### 1. Input Preparation

#### 1.1 Mesh 기반 입력

- Input:
  - single-material dental mesh (no texture)

Mesh로부터 기준 view에 대해 geometry 정보를 추출한다.

- normal map  
- position map  

이 geometry map은 이후 multi-view 생성 과정에서  
구조적 정합성을 유지하기 위한 조건으로 사용된다.

<img src="../../docs/assets/projects/mv-adapter/mesh_input.png" alt="mesh input" width="360">

---

#### 1.2 Realistic Reference Image 생성 (Nano Banana)
-ㅔ-=
- Nano Banana 입력:
  - mesh render image
  - geometry maps:
    - normal map
    - lighting map
    - depth map
- Output:
  - single realistic reference image

이 단계의 목적은  
**geometry 구조를 유지한 상태에서 색상, 질감, 조명 정보를 보완한 reference image를 생성하는 것**이다.  
생성된 이미지는 이후 MV-Adapter의 기준(reference)으로 사용된다.

| Nano Banana input maps | Nano Banana output |
|---|---|
| ![nano banana input maps](../../docs/assets/projects/mv-adapter/nano_banana_input_maps.png) | ![nano banana output](../../docs/assets/projects/mv-adapter/nano_banana_output.webp) |

---

### 2. Multi-view Geometry Condition 생성

기준 view를 중심으로 총 **6개 view**에 대해  
mesh로부터 geometry map을 직접 계산한다.

- normal map (6 views)
- position map (6 views)

해당 geometry condition은  
view 간 구조적 일관성을 유지하기 위한 **명시적 기하 제약**으로 사용된다.

| Normal maps | Position maps |
|---|---|
| ![6-view normal maps](../../docs/assets/projects/mv-adapter/normal_maps_6views.png) | ![6-view position maps](../../docs/assets/projects/mv-adapter/position_maps_6views.png) |

---

### 3. MV-Adapter 기반 Multi-view Image Generation

- Input:
  - realistic reference image (Nano Banana output)
  - 6-view geometry maps (depth, position)
- Process:
  - geometry condition을 MV-Adapter attention module에 주입
  - view 간 정보 공유를 통해 구조적 일관성 유지
- Output:
  - **6-view realistic multi-view images**


---

## Result

- 생성된 6-view 이미지는
  - 치아–잇몸 경계가 비교적 명확하게 유지되며
  - view 변화에 따른 형태 왜곡이 제한적으로 나타남
- single-image 기반 multi-view 생성 방식 대비,
  구조적 안정성이 향상됨을 확인하였다.

### View Results (0–5)

| Front (0) | Right (1) | Back (2) |
|---|---|---|
| ![front](../../docs/assets/projects/mv-adapter/controlnet_inference_result_view_0.png) | ![right](../../docs/assets/projects/mv-adapter/controlnet_inference_result_view_1.png) | ![back](../../docs/assets/projects/mv-adapter/controlnet_inference_result_view_2.png) |

| Left (3) | Top (4) | Bottom (5) |
|---|---|---|
| ![left](../../docs/assets/projects/mv-adapter/controlnet_inference_result_view_3.png) | ![top](../../docs/assets/projects/mv-adapter/controlnet_inference_result_view_4.png) | ![bottom](../../docs/assets/projects/mv-adapter/controlnet_inference_result_view_5.png) |

---

## Observation

- mesh 기반 geometry condition은
  multi-view consistency 확보에 핵심적인 역할을 함
- reference image 품질이 최종 결과에 직접적인 영향을 미침
- geometry condition이 없는 경우,
  view 간 형태 불일치가 쉽게 발생함

---

## Limitation

- reference image 생성 단계(Nano Banana)의 품질 한계가
  multi-view 결과에 그대로 전파됨
- geometry map 정확도 및 view sampling 방식에 따라
  결과 품질 편차가 발생할 수 있음

---

## Summary

본 실험을 통해  
**MV-Adapter는 geometry-aware reference image와 결합될 경우,  
dental domain에서도 구조적·시각적 multi-view consistency를 효과적으로 유지할 수 있음**을 확인하였다.




## Artifacts
- MV-Adapter 논문 정리: [notes/papers/mv-adapter.md](../../notes/papers/mv-adapter.md)
