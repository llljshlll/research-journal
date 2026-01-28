# Research Portfolio

단일 머티리얼 치아 mesh를 입력으로, **real-time**과 **consistency**를 동시에 만족하는 **scene reconstruction** 파이프라인을 만들기 위한 연구 및 구현 기록  

## Featured Projects

| Project | Summary | Key Result | Link |
|---|---|---|---|
| ctrLoRA + StreamDiffusion | 실시간 post-processing 파이프라인 | **~6s → ~0.1s** | [projects/ctrlora-streamdiffusion](projects/ctrlora-streamdiffusion/README.md) |
| MV-Adapter | mesh-conditioned multi-view generation | view coverage 한계 확인 | [projects/mv-adapter](projects/mv-adapter/README.md) |
| Skyfall-GS | sparse-view scene reconstruction 탐색 | FLUX 구조 분석 진행 | [projects/skyfall-gs](projects/skyfall-gs/README.md) |

## Research Notes

- Stable Diffusion 계열 구조 정리 (overview, ControlNet, LoRA, ctrLoRA, TAESD)  
  → [notes/models/stable_diffusion/](notes/models/stable_diffusion/)
- FLUX 모델 구조/아키텍처 분석  
  → [notes/models/flux/](notes/models/flux/)
- 핵심 논문 정리 (MV-Adapter, Skyfall-GS, 3D Gaussian Splatting, Tinker)  
  → [notes/papers/](notes/papers/)

## Timeline (High-level)

| Phase | Topic | Key Focus |
|------|------|-----------|
| Phase 1 | Diffusion Foundations | Stable Diffusion, ControlNet, LoRA |
| Phase 2 | Structure Conditioning | ctrLoRA, multi-condition |
| Phase 3 | Real-time & Consistency | StreamDiffusion |
| Phase 4 | Scene-level Generation | MV-Adapter, Skyfall-GS, FLUX |

## Current Status (Detailed Summary)

### Background
- Stable Diffusion 구조 전반 정리: CLIP, VAE, UNet, DDPM, DDIM, Transformer
- ControlNet/LoRA 구조 심화 분석 및 관련 논문 정리

### Project Goal
- 단일 머티리얼 치아 mesh를 입력으로 **일관성 있는 리얼리스틱 결과**를 생성
- 핵심 제약: **real-time** 추론 + **scene reconstruction**까지 확장 가능

### Approach A: Post-processing (real-time diffusion)
- StreamDiffusion 채택, 다중 condition 적용을 위해 ctrLoRA 선정
- ctrLoRA 및 StreamDiffusion 구조 파악 후 코드 병합
- TAESD 기반 ctrLoRA 결합 시도 -> 차원 불일치 원인 분석 및 해결
- TAESD 인코딩/디코딩에서 치아 원본 복원 한계로 TAESD 경로는 보류
- StreamDiffusion + ctrLoRA 통합 (LCM, stream batch 처리) 완료
  - 추론 시간: ~6s -> ~0.1s
  - multi-condition interference 문제로 디테일/일관성 저하 발생
  - 실험: segmentation 가중치 조정, segmentation+lighting map 블렌딩 등


### Approach B: Scene-level Reconstruction
- mesh + 단일 이미지 -> multi-view 일관 생성: MV-Adapter 사용
- view 부족 문제 확인
- sparse-view scene reconstruction 대안으로 Skyfall-GS 탐색
  - Skyfall-GS에서 FLUX 사용 -> FLUX 모델 구조 학습

| Front (0) | Right (1) | Back (2) | Left (3) | Top (4) | Bottom (5) |
|---|---|---|---|---|---|
| ![front](./docs/assets/projects/mv-adapter/controlnet_inference_result_view_0.png) | ![right](./docs/assets/projects/mv-adapter/controlnet_inference_result_view_1.png) | ![back](./docs/assets/projects/mv-adapter/controlnet_inference_result_view_2.png) | ![left](./docs/assets/projects/mv-adapter/controlnet_inference_result_view_3.png) | ![top](./docs/assets/projects/mv-adapter/controlnet_inference_result_view_4.png) | ![bottom](./docs/assets/projects/mv-adapter/controlnet_inference_result_view_5.png) |


### Current Work
- MV-Adapter 6-view 결과를 Skyfall-GS에 적용 예정
- 치아 도메인 적용 가능성 검증 위해 Skyfall-GS stage1 실험 진행 완료

## Detailed Research History

- 전체 연구 흐름과 실험 기록: [research_history.md](research_history.md)
