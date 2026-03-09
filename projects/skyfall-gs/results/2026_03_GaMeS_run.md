# 2026_03 GaMeS Run (6-View 적용 실험)

## 1. Motivation

Skyfall-GS 파이프라인에서 6-view 입력에 대해 **GaMeS (Gaussian Mesh Splatting)** 를 그대로 적용했을 때,
치아 도메인(mesh 기반)에서도 안정적으로 surface-aligned reconstruction이 가능한지 확인하고자 함.

---

## 2. Experimental Setup

- 입력 view: 총 6개 (front/right/back/left/top/bottom)

| Front (0) | Right (1) | Back (2) | Left (3) | Top (4) | Bottom (5) |
|---|---|---|---|---|---|
| <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/inputs/controlnet_inference_result_view_0.png" width="180"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/inputs/controlnet_inference_result_view_1.png" width="180"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/inputs/controlnet_inference_result_view_2.png" width="180"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/inputs/controlnet_inference_result_view_3.png" width="180"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/inputs/controlnet_inference_result_view_4.png" width="180"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/inputs/controlnet_inference_result_view_5.png" width="180"> |

- 기본 조건:
  - GaMeS 논문/공개 구현의 기본 설정을 우선 그대로 적용
  - 이후 문제 원인 가설에 따라 mesh 처리 방식만 단계적으로 변경

---

## 3. Trials and Results

### 3.1 Baseline: GaMeS 기본 적용 (as-is)

- 설정: GaMeS에 올라와 있는 기본 방식 그대로 적용
- 결과: 출력이 비정상적으로 나타남 (전체적으로 품질 저하/흐림)

| Stage | Front (0) | Right (1) | Back (2) | Left (3) | Top (4) | Bottom (5) |
|---|---|---|---|---|---|---|
| Input | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/inputs/controlnet_inference_result_view_0.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/inputs/controlnet_inference_result_view_1.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/inputs/controlnet_inference_result_view_2.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/inputs/controlnet_inference_result_view_3.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/inputs/controlnet_inference_result_view_4.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/inputs/controlnet_inference_result_view_5.png" width="140"> |
| GaMeS (as-is) | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/outputs/00000.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/outputs/00001.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/outputs/00002.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/outputs/00003.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/outputs/00004.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/outputs/00005.png" width="140"> |


### 3.2 Mesh 고정 후 적용

- 설정: mesh를 고정한 상태로 GaMeS 적용
- 결과: 여전히 비정상적 결과 (개선 효과 미미)

| Stage | Front (0) | Right (1) | Back (2) | Left (3) | Top (4) | Bottom (5) |
|---|---|---|---|---|---|---|
| Input | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/inputs/controlnet_inference_result_view_0.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/inputs/controlnet_inference_result_view_1.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/inputs/controlnet_inference_result_view_2.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/inputs/controlnet_inference_result_view_3.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/inputs/controlnet_inference_result_view_4.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/inputs/controlnet_inference_result_view_5.png" width="140"> |
| GaMeS (mesh fixed) | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/outputs_fix/00000.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/outputs_fix/00001.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/outputs_fix/00002.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/outputs_fix/00003.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/outputs_fix/00004.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/outputs_fix/00005.png" width="140"> |

### 3.3 Triangle 크기 기반 치아 영역 채움

- 가설:
  - GaMeS는 mesh의 triangle 내부에 Gaussian을 배치하므로,
    triangle 분포 불균형이 곧 Gaussian 분포 불균형으로 이어짐
- 관찰:
  - 치아 영역은 vertex가 촘촘함
  - 잇몸 영역은 상대적으로 성김 (큰 triangle 다수)
  - 실제 결과에서도 치아보다 잇몸 영역에서 흐림/붕괴가 두드러짐

| Mesh density observation |
|---|
| <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/mesh2.png" width="560"> |

- 적용:
  - triangle 크기 중간값(median)을 기준으로
  - 중간값보다 큰 triangle을 분할(subdivision)하여 밀도 보강

| Triangle split (median threshold) |
|---|
| <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/mesh.png" width="560"> |

- 결과:
  - 여전히 치아 앞부분만 생성 안됨

| Stage | Front (0) | Right (1) | Back (2) | Left (3) | Top (4) | Bottom (5) |
|---|---|---|---|---|---|---|
| Input | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/inputs/controlnet_inference_result_view_0.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/inputs/controlnet_inference_result_view_1.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/inputs/controlnet_inference_result_view_2.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/inputs/controlnet_inference_result_view_3.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/inputs/controlnet_inference_result_view_4.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/inputs/controlnet_inference_result_view_5.png" width="140"> |
| GaMeS (triangle split) | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/outputs_mesh/00000.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/outputs_mesh/00001.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/outputs_mesh/00002.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/outputs_mesh/00003.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/outputs_mesh/00004.png" width="140"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/outputs_mesh/00005.png" width="140"> |

---


---

## 4. Artifacts

- GaMeS 개념 정리: [projects/skyfall-gs/references/GaMeS.md](../references/GaMeS.md)

