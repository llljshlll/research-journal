# 3D Gaussian Splatting for Real-Time Radiance Field Rendering

GS 기반 **explicit scene representation** 논문  
NeRF 계열의 volumetric rendering 품질을 유지하면서,  
**real-time rendering이 가능한 radiance field 표현을 최초로 제안**한 foundational work  

---

## 1. Motivation

### Problem
- **Neural Radiance Field (NeRF)** 기반 방법의 구조적 한계
  - volumetric ray marching 기반 rendering → 높은 계산 비용
  - 고품질 결과를 위해 긴 학습 시간 필요
  - 고해상도 환경에서 real-time rendering 불가능

- 기존 **fast NeRF** 계열의 trade-off
  - Instant-NGP, Plenoxels 등은 속도는 개선되었으나
  - grid / hash 기반 구조로 인해
    - empty space 비효율
    - 대규모 장면에서 품질 한계

### Limitations of Existing Methods
- **NeRF / Mip-NeRF360**
  - image quality는 SOTA
  - training time 수십 시간
  - rendering 수 초 / frame

- **Fast NeRF (Instant-NGP, Plenoxels)**
  - 빠른 학습 및 렌더링
  - grid discretization으로 인한 표현력 제한

결과적으로,
**연속적 image formation을 유지하면서**
GPU 친화적인 explicit 장면 표현 방식이 요구됨

---

## 2. Mechanism

3D Gaussian Splatting은  
**radiance field를 neural network가 아닌,
explicit한 3D Gaussian set으로 표현**하는 방식 제안.

핵심 목표:
- NeRF와 동일한 image formation 유지
- ray marching 제거
- real-time rasterization 기반 rendering

---

## 2.1 Scene Representation — 3D Gaussians

장면은 **N개의 3D Gaussian primitive** 집합으로 표현됨.

각 Gaussian은 다음 파라미터를 가짐:

- **Position (μ)**  
  Gaussian의 중심 위치 (3D)

- **Covariance (Σ)**  
  Gaussian의 공간적 확산 정도  
  → anisotropic covariance 허용

- **Opacity (α)**  
  해당 Gaussian의 투과도

- **Color (SH coefficients)**  
  view-dependent appearance 표현을 위한  
  **Spherical Harmonics(SH)** 계수

> Scene representation = **(μ, Σ, α, SH)** 의 explicit parameter set

---

## 2.2 Image Formation Model

3DGS는 **NeRF와 동일한 volumetric α-blending image formation**을 따름.

Pixel color C는
front-to-back order로 정렬된 Gaussian에 대해 다음과 같이 계산됨:
```
C = Σ T_i · α_i · c_i
T_i = Π_{j<i} (1 − α_j)
```

- NeRF: ray marching 기반 샘플링
- 3DGS: **screen-space Gaussian splatting**

→ image formation은 유지하면서
rendering efficiency를 극적으로 향상

---

## 2.3 Differentiable Gaussian Parameterization

Covariance matrix Σ를 직접 최적화할 경우
positive semi-definite 제약 유지가 어려움.

이를 해결하기 위해,
Gaussian covariance를 다음과 같이 재파라미터화:

- **Scale vector (s)**
- **Rotation quaternion (q)**
```
Σ = R(q) · S(s) · S(s)^T · R(q)^T
```

장점:
- anisotropic Gaussian 표현 가능
- gradient descent 기반 안정적 최적화
- surface-aligned Gaussian 형성 가능

---

## 3. Optimization

### 3.1 Training Objective

학습은 **analysis-by-synthesis** 방식으로 수행.

- multi-view RGB images 입력
- 현재 Gaussian set으로 rendering
- rendered image와 GT image 간 오차 최소화

Loss function:
```
L = (1 − λ) · L1 + λ · D-SSIM
```

- λ = 0.2
- geometry / appearance에 대한 직접 supervision 없음
- **RGB reconstruction만으로 모든 파라미터 학습**

---

### 3.2 Adaptive Density Control

3DGS는 **Gaussian 개수를 고정하지 않고,
학습 중 동적으로 조절**함.

Gaussian의 view-space position gradient를 기준으로
다음 두 경우를 구분:

#### Under-reconstruction
- 작은 Gaussian
- 구조가 충분히 표현되지 않은 영역
- → **clone**
  - Gaussian 복제
  - gradient 방향으로 이동

#### Over-reconstruction
- 큰 Gaussian
- 하나의 Gaussian이 넓은 영역 커버
- → **split**
  - 두 개의 Gaussian으로 분해
  - scale 감소

추가 규칙:
- opacity α가 매우 작은 Gaussian 제거
- 과도하게 큰 Gaussian pruning

→ sparse SfM initialization으로부터
dense Gaussian scene으로 자동 확장

---

## 4. Rendering — Fast Differentiable Rasterization

3DGS는 **tile-based rasterization** 기반 rendering 사용.

### Rendering Pipeline
- 화면을 **16×16 tile**로 분할
- Gaussian을 overlap되는 tile 단위로 복제
- `(tile_id, depth)` 기준으로 전역 radix sort
- tile 내부에서 front-to-back α-blending

특징:
- visibility-aware
- anisotropic splatting 지원
- **gradient 개수 제한 없음**

→ training 및 inference 모두에서 높은 안정성 확보  
→ real-time rendering 가능

---

## 5. Experimental Results

- Mip-NeRF360과 동등하거나 일부 장면에서 더 높은 PSNR / SSIM
- training time:
  - 수 분 ~ 수십 분
- rendering speed:
  - 수십 ~ 수백 FPS (scene / 해상도에 따라)

특히:
- NeRF 계열 중 **최초로 real-time high-quality novel view synthesis 달성**

---

## 6. Limitations

3D Gaussian Splatting의 한계:

- **Memory footprint**
  - 수십~수백 MB 규모의 Gaussian parameter 저장 필요

- **Unobserved regions**
  - 입력 view에서 관측되지 않은 영역은 artifact 발생 가능

- **Popping artifacts**
  - 큰 anisotropic Gaussian으로 인한 시점 변화 시 artifact

- **No explicit geometry prior**
  - mesh / depth / semantic constraint 없음
  - 후속 연구에서 개선 여지 존재

