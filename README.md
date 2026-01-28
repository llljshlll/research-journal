# Research Portfolio

**geometry-aware diffusion 기반 real-time 생성 파이프라인**을 구축하기 위한 연구 및 구현 기록.  
목표는 단일 mesh 입력에서의 **일관성 유지**와 **실시간 추론**, 그리고 multi-view/scene-level 생성까지의 통합이다.

## Featured Projects

| Project | Summary | Key Result | Link |
|---|---|---|---|
| ctrLoRA + StreamDiffusion | 실시간 post-processing 파이프라인 | **~6s → ~0.1s** | [projects/ctrlora-streamdiffusion](projects/ctrlora-streamdiffusion/README.md) |
| MV-Adapter | mesh-conditioned multi-view generation | view coverage 한계 확인 | [projects/mv-adapter](projects/mv-adapter/README.md) |
| Skyfall-GS | sparse-view scene reconstruction 탐색 | FLUX 구조 분석 진행 | [projects/skyfall-gs](projects/skyfall-gs/README.md) |

## Research Notes

- Stable Diffusion 계열 구조 정리  
  → [notes/models/stable_diffusion/](notes/models/stable_diffusion/)
- MV-Adapter, Skyfall-GS 논문 정리  
  → [notes/papers/](notes/papers/)
- FLUX 모델 정리  
  → [notes/models/flux/](notes/models/flux/)

## Timeline (High-level)

| Phase | Topic | Key Focus |
|------|------|-----------|
| Phase 1 | Diffusion Foundations | Stable Diffusion, ControlNet, LoRA |
| Phase 2 | Structure Conditioning | ctrLoRA, multi-condition |
| Phase 3 | Real-time & Consistency | StreamDiffusion |
| Phase 4 | Scene-level Generation | MV-Adapter, Skyfall-GS, FLUX |

## Detailed Research History

- 전체 연구 흐름과 실험 기록: [research_history.md](research_history.md)
