# 2026_03 GaMeS Run (6-View 적용 실험)

## 1. Motivation

Skyfall-GS 파이프라인에서 6-view 입력에 대해 **GaMeS (Gaussian Mesh Splatting)** 를 그대로 적용했을 때,
치아 도메인(mesh 기반)에서도 안정적으로 surface-aligned reconstruction이 가능한지 확인하고자 함.

---

## 2. Experimental Setup

- 입력 view: 총 6개 (front/right/back/left/top/bottom)
- 기본 조건:
  - GaMeS 논문/공개 구현의 기본 설정을 우선 그대로 적용
  - 이후 문제 원인 가설에 따라 mesh 처리 방식만 단계적으로 변경

---

## 3. Trials and Results

### 3.1 Baseline: GaMeS 기본 적용 (as-is)

- 설정: GaMeS에 올라와 있는 기본 방식 그대로 적용
- 결과: 출력이 비정상적으로 나타남 (전체적으로 품질 저하/흐림)

### 3.2 Mesh 고정 후 적용

- 설정: mesh를 고정한 상태로 GaMeS 적용
- 결과: 여전히 비정상적 결과 (개선 효과 미미)

### 3.3 Triangle 크기 기반 치아 영역 채움

- 가설:
  - GaMeS는 mesh의 triangle 내부에 Gaussian을 배치하므로,
    triangle 분포 불균형이 곧 Gaussian 분포 불균형으로 이어짐
- 관찰:
  - 치아 영역은 vertex가 촘촘함
  - 잇몸 영역은 상대적으로 성김 (큰 triangle 다수)
  - 실제 결과에서도 치아보다 잇몸 영역에서 흐림/붕괴가 두드러짐
- 적용:
  - triangle 크기 중간값(median)을 기준으로
  - 중간값보다 큰 triangle을 분할(subdivision)하여 밀도 보강
- 결과:
  - 일부 개선됨 (특히 빈 영역 완화)
  - 그러나 품질이 완전히 회복되지는 않음

---

## 4. Analysis

GaMeS는 point cloud 기반 3DGS와 달리,  
**mesh triangle 구조 자체**가 Gaussian 배치 밀도를 결정함.

- 기존 point cloud 기반 3DGS:
  - 전체 표면에 균등한 point를 추가하는 방식이 비교적 단순하게 가능
- GaMeS (mesh 기반):
  - triangle topology/크기 분포의 영향을 직접 받음
  - 따라서 단순히 "전체 표면 균등 point 추가"와 같은 접근을 그대로 적용하기 어려움

즉, 현재 실패 원인은 단순 학습 불안정성보다  
**치아/잇몸 영역 간 mesh 해상도 불균형**에서 발생한 구조적 한계일 가능성이 큼.

---

## 5. Interim Conclusion

- GaMeS 기본 적용 및 mesh 고정만으로는 6-view 치아 도메인에서 안정적인 결과를 얻기 어려웠음
- triangle 크기 기반 분할로 일부 개선은 가능했으나 충분하지 않음
- 후속으로는 다음 방향이 필요:
  1. 영역별 adaptive remeshing (잇몸/경계부 우선 고밀도화)
  2. triangle 면적 가중 Gaussian budget 재배치
  3. 치아/잇몸 분리 마스크 기반의 영역별 다른 배치 전략

---

## 6. Artifacts

- GaMeS 개념 정리: [projects/skyfall-gs/references/GaMeS.md](../references/GaMeS.md)
- 관련 LoRA/Refinement 실험: [projects/skyfall-gs/lora/flow-edit_lora_training.md](../lora/flow-edit_lora_training.md)
- 비교 참고 실험: [projects/skyfall-gs/results/2026_01_idu_iterative_refinement_test.md](./2026_01_idu_iterative_refinement_test.md)
