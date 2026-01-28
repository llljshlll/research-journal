# ctrLoRA + StreamDiffusion (Real-time Post-processing)

## Summary
단일 머티리얼 치아 mesh 렌더링 결과를 **실시간으로 사실화(post-processing)** 하기 위한 파이프라인을 구축했다.  
목표는 **real-time**과 **consistency**를 동시에 만족하는 결과를 만드는 것이다.

## Goals
- 단일 mesh 입력에서 일관된 시각적 결과 유지
- 렌더링/시뮬레이션 환경에서도 실시간 추론 가능

## Approach
- Stream Diffusion을 기반으로 실시간 파이프라인 구성
- 다중 조건을 가볍게 적용하기 위해 ctrLoRA 선택
- Stream Diffusion + ctrLoRA 결합, LCM 및 stream batch 처리 적용

## Results
- 추론 시간: **약 6초 → 약 0.1초**
- LCM 적용 단일 이미지 기준 **1.53s** 확인 (stream batch 적용 시 추가 개선 기대)

## Issues & Insights
- multi-condition 적용 시 디테일 붕괴 및 temporal consistency 문제 발생
- post-processing 기반 접근은 **consistency 확보에 구조적 한계**가 있음을 확인

## Experiments
- [experiments.md](experiments.md)

## References
- ctrLoRA 구조 분석: [notes/models/stable_diffusion/ctrLoRA.md](../../notes/models/stable_diffusion/ctrLoRA.md)
- TAESD 분석: [notes/models/stable_diffusion/TAESD.md](../../notes/models/stable_diffusion/TAESD.md)
- Stable Diffusion overview: [notes/models/stable_diffusion/overview.md](../../notes/models/stable_diffusion/overview.md)
