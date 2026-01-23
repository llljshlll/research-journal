# MV-Adapter + Skyfall-GS Design Document
---

## 1. Problem Statement

- MV-Adapter는 제한된 수의 multi-view images만 생성  
- sparse view 조건에서 vanilla 3DGS는:  
  - SfM 기반 초기화가 불안정    
  - geometry drift 발생  
- 특히 dental mesh와 같이 **GT geometry가 이미 존재하는 경우**,    
  SfM 및 depth supervision은 불필요하거나 오히려 conflict를 유발  

---

## 2. Key Idea

- Sparse view 조건에서도 동작하는 Skyfall-GS 구조를 활용  
- 단, Skyfall-GS의 Stage1을 다음과 같이 변경:  
  - SfM 제거  
  - mesh를 명시적인 geometry constraint로 사용  
- Stage2는 diffusion-assisted refinement 구조를 유지하되,  
  view generation 전략만 변경   

---

## 3. Overall Pipeline

### Original Skyfall-GS

- Stage1:  
  - satellite images 기반 initial 3DGS 학습  
- Stage2:  
  - diffusion으로 stable view 생성  
  - GS parameter refinement  

### Proposed Pipeline

- Stage1:  
  - mesh-constrained vanilla 3DGS initialization   
- Stage2:  
  - (추후 상세 설계 추가)  

---

## 4. Dataset Specification

```
images/                  # RGB images (per-tooth)  
masks/                   # foreground mask (tooth + gum = 1, background = 0, optional)  
transforms_train.json    # camera parameters (PINHOLE camera model)  
transforms_test.json     # optional   
points3D.ply             # mesh (replacing sparse point cloud)  
```  
> 확인 필요:  
> MV-Adapter에서 사용하는 camera convention  
> Skyfall-GS loader와의 호환성  


## 5. Stage1: Mesh-Constrained 3DGS
mesh를 GT geometry로 간주하고, 모든 Gaussian이 surface에 정렬되도록 제약  

### 5.1 Initialization

#### Position (x, y, z)
- SfM 기반 point cloud 초기화는 사용하지 않음  
- mesh surface에서 Gaussian center를 직접 샘플링  
- Sampling 방식:  
  - Area-weighted surface sampling  
    1. 각 triangle의 면적 A 계산  
    2. 전체 mesh 면적 대비 비례적으로 샘플 수 분배  
    3. triangle 내부에서 uniform barycentric sampling  
- 초기 Gaussian 개수: 약 50,000  
- mesh는 학습 중 **고정된 geometry constraint**로 유지  


#### Rotation 
- Gaussian rotation은 mesh surface normal을 기준으로 초기화  
- Gaussian의 **minor axis**이 normal 방향을 따르도록 정렬  

#### Opacity, Scale, Color (SH)
기존 방식 유지  


### 5.2 Rendering
- 입력:  
  - Gaussian parameters (μ, Σ, opacity, SH)  
  - camera pose (intrinsic + extrinsic)  
- 과정:  
  - world → camera → screen space 변환  
  - tile-based rasterization  
  - front-to-back alpha blending  
- 출력:  
  - rendered RGB image  

> Rendering pipeline은 vanilla 3DGS와 동일하며 수정하지 않음  

### 5.3 Optimization
#### Photometric Loss
- Skyfall-GS 구조를 유지하며 L1 + SSIM 사용  
- mask 사용해서 mask에 대해서만 loss 적용, 배경은 0으로 넘겨서 Loss 적용x  
  
---

#### Depth Supervision 제거
- 기존 Skyfall-GS에서 사용하던 다음 loss 제거  
  - Ground Truth Depth Loss  
  - Pseudo-Depth Loss  
- 이유:  
  - mesh가 이미 명시적인 geometry를 제공  

---

### 5.4 Densification (Mesh-Aware)

#### Split

- split은 **tangent plane 방향으로만 수행**
- normal 방향 perturbation은 금지  
- 새 Gaussian은:  
  - mesh surface로 projection  
  - 항상 surface 위에 위치하도록 유지  

---

#### Prune

- 다음 조건에서 Gaussian 제거:  
  - opacity가 매우 낮음  
  - screen-space contribution이 거의 없음  
  - mesh surface에서 일정 거리 이상 벗어남  

---

#### Clone

- gradient가 크고 screen-space 영향이 큰 Gaussian을 복제  
- 위치는 유지  
- scale, opacity만 조정  

++
일정 iteration 이후
- hard projection (업데이트되는 xyz를 항상 surface에 부착)
- xyz freeze 
중 하나를 선택
