# Research History

## Executive Summary

본 리포지토리는  
**geometry-aware diffusion 기반의 real-time 생성 파이프라인**을 구축하기 위한  
연구 및 구현 과정을 정리한 기록이다.   
  
주요 목표는 다음과 같다.  
- 단일 mesh 기반 입력에서 **일관된 시각적 결과 (consistency)** 를 유지하는 생성. 
- rendering / simulation 환경에서도 사용 가능한 **real-time inference**. 
- diffusion, multi-view generation, 3D scene reconstruction을 하나의 파이프라인으로 연결

### 구현 및 실험한 주요 내용
- Stream Diffusion + ctrLoRA 기반 real-time post-processing pipeline 구성. 
- LCM 및 stream batch 처리로 inference latency **약 6초 → 약 0.1초 수준으로 감소**
- MV-Adapter를 이용한 mesh-conditioned multi-view image generation  
- multi-view 결과를 활용한 3D Gaussian Splatting 기반 scene reconstruction 시도. 
- sparse view 환경에서의 한계를 분석하고 Skyfall-GS 및 FLUX 구조를 대안으로 탐색 중

---

## Timeline (High-level)

| Phase | Topic | Key Focus |
|------|------|-----------|
| Phase 1 | Diffusion Foundations | Stable Diffusion, ControlNet, LoRA |
| Phase 2 | Structure Conditioning | ctrLoRA, multi-condition |
| Phase 3 | Real-time & Consistency | StreamDiffusion |
| Phase 4 | Scene-level Generation | MV-Adapter, Skyfall-GS, FLUX |

---

## Key Research History

### 1. Background: Stable Diffusion 구조 이해

- Stable Diffusion 전체 파이프라인을 구조적으로 학습
  - CLIP, VAE, UNet
  - DDPM, DDIM
  - Transformer 기반 attention 구조
- ControlNet, LoRA 구조를 포함한 SD 확장 기법들을 상세히 분석

→ 이후 프로젝트에서 **모델 선택과 구조 판단의 기반**이 됨  
🔗 관련 정리: [notes/models/stable_diffusion/overview.md](notes/models/stable_diffusion/overview.md)

---

### 2. Project Goal Definition (Dental Domain)

- 입력: **단일 머티리얼 치아 mesh**
- 목표:
  1. **일관성 (Consistency)**
  2. **실시간 처리 (Real-time inference)**
- 문제:
  - 렌더링 엔진에서 회전 시
  - 단일 머티리얼 → 리얼리스틱 텍스처 표현이 어려움

🔗 프로젝트: [projects/ctrlora-streamdiffusion](projects/ctrlora-streamdiffusion/README.md)

---

### 3. Approach 1: Post-processing via Diffusion

#### 3.1 Strategy Selection

- 렌더링 결과를 **사실화(post-processing)** 하는 방향 선택
- 실시간 요구사항으로 인해 **Stream Diffusion** 방식 채택
- 구조 조건 적용을 위해 **ctrLoRA** 선택
  - ControlNet 계열 논문들과 비교 분석 후 적합하다고 판단

🔗 ctrLoRA 구조 분석: [notes/models/stable_diffusion/ctrLoRA.md](notes/models/stable_diffusion/ctrLoRA.md)

---

#### 3.2 Stream Diffusion + ctrLoRA 구조 분석

- Stream Diffusion 코드 구조 분석
- ctrLoRA 구조 및 conditioning 방식 분석
- 두 구조를 결합하기 위한 전체 파이프라인 파악

---

#### 3.3 TAESD 적용 및 한계

- Stream Diffusion에서 사용하는 **TAESD** 기반으로
  - ctrLoRA + TAESD 조합을 먼저 시도
- 문제 발생:
  - latent 차원 불일치 문제
  - 원인: TAESD와 ctrLoRA가 사용하는 **기본 Stable Diffusion 모델 차원 차이**
- 해결:
  - 차원 문제는 해결했으나
  - **TAESD 자체가 치아 원본을 충분히 복원하지 못함**
- 결론:
  - TAESD 기반 접근 방식 포기

🔗 TAESD 분석: [notes/models/stable_diffusion/TAESD.md](notes/models/stable_diffusion/TAESD.md)

---

#### 3.4 Stream Diffusion + ctrLoRA 직접 병합

- TAESD 제거
- Stream Diffusion + ctrLoRA 직접 병합
  - LCM 적용
  - stream batch 처리
- 결과:
  - Inference 시간  
    **약 6초 → 약 0.1초**

---

#### 3.5 Multi-condition Interference 문제

- ctrLoRA에서 **condition 2개 이상 적용 시**
  - 디테일 붕괴
  - 일관성 유지 실패
- 해결 시도:
  - segmentation 기반 가중치 강화
  - segmentation + lighting map을 하나의 condition으로 합성
- 결과:
  - 여전히 **temporal consistency 문제 해결 실패**
- 판단:
  - post-processing diffusion 방식은 multi-condition 및 temporal consistency를 구조적으로 보장하기 어렵다고 판단
  - 단일 frame 품질 개선에는 효과적이나, scene-level consistency 확보에는 한계가 있음을 명확히 인식

---

### 4. Approach 2: Scene-level Reconstruction

#### 4.1 MV-Adapter 도입

- 방향 전환:
  - 결과를 보정하는 방식이 아니라
  - **씬 자체를 일관되게 생성하는 방식**
- 선택:
  - mesh + single image를 입력으로
  - multi-view 일관 이미지를 생성하는 **MV-Adapter** 사용

🔗 MV-Adapter 정리: [notes/papers/mv-adapter.md](notes/papers/mv-adapter.md)

---

#### 4.2 View 부족 문제

- MV-Adapter로 생성한 다중 뷰를
  - Gaussian Splatting 기반 씬 재구성에 사용하려 시도
- 문제:
  - view coverage 부족
  - 안정적인 3D 씬 재구성 실패
- 결과: 
  - MV-Adapter의 multi-view 생성 결과만으로는 안정적인 3D Gaussian Splatting 최적화에 필요한 view coverage가 부족함
- 판단:
  - sparse view 환경에서의 scene reconstruction 문제를 별도의 구조로 해결할 필요성을 확인
  
---

### 5. Skyfall-GS 및 FLUX 모델 탐색

- sparse view 문제 해결 방안 탐색 중 **Skyfall-GS** 발견
  - 적은 위성 사진 뷰로도 도시 씬 생성
  - diffusion + 3D Gaussian Splatting 결합
- 공통점:
  - sparse view
  - scene reconstruction
- Skyfall-GS에서 사용하는 diffusion 모델이 **FLUX**
- 이에 따라:
  - FLUX 모델 구조 분석
  - 현재는 **FLUX ↔ MV-Adapter 구조적 관계**를 분석 중

🔗 Skyfall-GS 정리: [notes/papers/skyfall-gs.md](notes/papers/skyfall-gs.md)
🔗 FLUX 구조 분석: [notes/models/flux/flux_overview.md](notes/models/flux/flux_overview.md)

---

## Related Notes & Projects

- Stable Diffusion 계열 구조 정리  
  → [notes/models/stable_diffusion/](notes/models/stable_diffusion/)
- ControlNet / LoRA / ctrLoRA 분석  
  → [notes/models/stable_diffusion/](notes/models/stable_diffusion/)
- MV-Adapter, Skyfall-GS 논문 정리  
  → [notes/papers/](notes/papers/)
- Featured Projects  
  → [projects/](projects/)
