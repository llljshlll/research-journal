# Skyfall-GS on Dental Domain (Stage1 / Stage2 Results)

## 1. Background

- **Skyfall-GS**는 원래 **위성(satellite) 데이터 도메인**을 대상으로 설계된 모델임.
- 본 실험에서는 이를 **치아(dental) 도메인**에 적용하여,
  - MV-Adapter로 생성한 **소수 view (6 views)** 조건에서
  - Stage1 / Stage2의 동작 특성과 한계를 분석함.

---

## 2. Experimental Setup

### 2.1 Input Conditions
- Domain: Dental (tooth)
- Input views: 52 views dental image


### 2.2 Stage1 Strategy
- 초기 Gaussian point cloud를 mesh surface 기반으로 sampling
- Densification 과정에서:
  - mesh 근처가 아닌 Gaussian → pruning
  - mesh 근처에서만 split 발생하도록 mesh constraint 적용

---

## 3. Stage1 Result

- 전체 구조는 빠르게 형성되나,
  - floating Gaussian
  - streaking artifact
  - 표면 불안정성
  이 일부 영역에서 관찰됨.

### Stage1 Visualization

| View | Image |
|----|----|
| Front | ![](../../docs/assets/projects/skyfall-gs/stage1_front.png) |
| Back | ![](../../docs/assets/projects/skyfall-gs/stage1_back.png) |
| Top | ![](../../docs/assets/projects/skyfall-gs/stage1_top.png) |
| Side | ![](../../docs/assets/projects/skyfall-gs/stage1_side.png) |

---

## 4. Stage2 Strategy

- Stage1 checkpoint를 초기값으로 사용
- Stage1과 동일한 camera path로 결과 비교

---

## 5. Stage2 Results (Image Comparison)

### 5.1 Checkpoint-wise Comparison

| Checkpoint | Front View | Back View | Top View | Side View | Observation |
|----|----|----|----|----|----|
| ckpt 40000 | ![](../../docs/assets/projects/skyfall-gs/stage2_40k_front.png) | ![](../../docs/assets/projects/skyfall-gs/stage2_40k_back.png) | ![](../../docs/assets/projects/skyfall-gs/stage2_40k_top.png) | ![](../../docs/assets/projects/skyfall-gs/stage2_40k_side.png) | 노이즈 감소 시작 |
| ckpt 50000 | ![](../../docs/assets/projects/skyfall-gs/stage2_50k_front.png) | ![](../../docs/assets/projects/skyfall-gs/stage2_50k_back.png) | ![](../../docs/assets/projects/skyfall-gs/stage2_50k_top.png) | ![](../../docs/assets/projects/skyfall-gs/stage2_50k_side.png) | floating Gaussian 감소 |
| ckpt 60000 | ![](../../docs/assets/projects/skyfall-gs/stage2_60k_front.png) | ![](../../docs/assets/projects/skyfall-gs/stage2_60k_back.png) | ![](../../docs/assets/projects/skyfall-gs/stage2_60k_top.png) | ![](../../docs/assets/projects/skyfall-gs/stage2_60k_side.png) | 표면 안정성 가장 양호 |
| ckpt 70000 | ![](../../docs/assets/projects/skyfall-gs/stage2_70k_front.png) | ![](../../docs/assets/projects/skyfall-gs/stage2_70k_back.png) | ![](../../docs/assets/projects/skyfall-gs/stage2_70k_top.png) | ![](../../docs/assets/projects/skyfall-gs/stage2_70k_side.png) | 일부 over-smoothing |


---

## 6. Stage1 vs Stage2 (ckpt 60000) – Video Comparison

| Stage | Video |
|----|----|
| Stage1 | ![](../../docs/assets/projects/skyfall-gs/stage1_video.mp4) |
| Stage2 (ckpt 50000) | ![](../../docs/assets/projects/skyfall-gs/stage2_50k_video.mp4) |

---

