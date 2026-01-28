# ctrLoRA + StreamDiffusion (Real-time Post-processing)

## Summary
단일 머티리얼 치아 mesh 렌더링 결과를 **실시간으로 사실화(post-processing)** 하기 위한 파이프라인을 구축했다. 
일관성(Consistency)과 실시간 추론(Real-time inference)을 동시에 만족하는 것을 목표로 했다.

## Goals
- 단일 mesh 입력에서 일관된 시각적 결과 유지
- 렌더링/시뮬레이션 환경에서도 실시간 추론 가능

## Approach
- Stream Diffusion을 기반으로 실시간 파이프라인 구성
- 구조 조건 적용을 위해 ctrLoRA 선택
- LCM 및 stream batch 처리 적용

## Results
- 추론 시간: **약 6초 → 약 0.1초**

## Issues & Insights
- multi-condition 적용 시 디테일 붕괴 및 temporal consistency 문제 발생
- 단일 프레임 품질 개선에는 효과적이나, scene-level consistency 확보에는 구조적 한계 확인

## References
- ctrLoRA 구조 분석: [notes/models/stable_diffusion/ctrLoRA.md](../../notes/models/stable_diffusion/ctrLoRA.md)
- TAESD 분석: [notes/models/stable_diffusion/TAESD.md](../../notes/models/stable_diffusion/TAESD.md)
- Stable Diffusion overview: [notes/models/stable_diffusion/overview.md](../../notes/models/stable_diffusion/overview.md)
