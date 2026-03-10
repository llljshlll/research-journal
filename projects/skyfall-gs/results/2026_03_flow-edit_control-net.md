# FLUX.1-dev FlowEdit + ControlNet (Asymmetric CN) Experiment

> FlowEdit의 장점(고품질 denoise)을 유지하면서 segmentation/normal/depth 조건을 직접 주입하기 위해
> FLUX.1-dev + Union ControlNet 결합을 시도한 실험 기록

---

## Overview

FlowEdit 단독은 시각 품질은 좋지만, 치아/잇몸 경계나 표면 형태를 구조적으로 강제하기 어려움.
이를 보완하기 위해 ControlNet condition을 FlowEdit에 결합해 구조 정보를 직접 주입하고자 함.

목표:
- 경계 교정: segmentation/canny 조건 반영
- 표면 디테일 보강: depth/normal 계열 조건 반영

---

## Method

### Asymmetric ControlNet FlowEdit

FlowEdit은 inversion 없이 이미지를 편집하는 ODE 기반 방법:

`Δv_t = v_t_tar(z_t, p_tar) - v_t_src(z_t, p_src)`

여기에 ControlNet을 **비대칭(asymmetric)** 으로 결합:

```
delta = (transformer(x, p_tar) + CN_residuals) - transformer(x, p_src)
```
&&delta라고 적혀잇거나 이런거 다 실제 문자로 변환 그리고 비대칭으로 contorlNet을 결합한다는 거 무슨 의미인지 더 설명

- `v_t_src` (source): 일반 transformer — ControlNet 없음
- `v_t_tar` (target): transformer + ControlNet residuals

residuals가 source에서 상쇄되지 않고 target에만 작용하여 condition이 반영


---

## Results

### normal
| Input (render) | Input (normal) | No ControlNet | ControlNet (`cn_scale=0.3`) | ControlNet (`cn_scale=0.7`) |
|---|---|---|---|---|
| <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/ori/00000.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/normals/controlnet_inference_result_nor_view_0.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/NoCn/00000.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnNormal03/00000.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnNormal07/00000.png" width="150"> |
| <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/ori/00001.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/normals/controlnet_inference_result_nor_view_1.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/NoCn/00001.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnNormal03/00001.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnNormal07/00001.png" width="150"> |
| <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/ori/00002.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/normals/controlnet_inference_result_nor_view_2.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/NoCn/00002.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnNormal03/00002.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnNormal07/00002.png" width="150"> |
| <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/ori/00003.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/normals/controlnet_inference_result_nor_view_3.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/NoCn/00003.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnNormal03/00003.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnNormal07/00003.png" width="150"> |
| <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/ori/00004.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/normals/controlnet_inference_result_nor_view_4.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/NoCn/00004.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnNormal03/00004.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnNormal07/00004.png" width="150"> |
| <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/ori/00005.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/normals/controlnet_inference_result_nor_view_5.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/NoCn/00005.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnNormal03/00005.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnNormal07/00005.png" width="150"> |

### segmentation
| Input (render) | Input (seg) | No ControlNet | ControlNet (`cn_scale=0.2`) | ControlNet (`cn_scale=0.5`) | ControlNet (`cn_scale=0.7`) |
|---|---|---|---|---|---|
| <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/ori/00000.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/segmentation/seg_0000.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/NoCn/00000.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnSeg02/00000.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnSeg05/00000.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnSeg07/00000.png" width="150"> |
| <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/ori/00001.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/segmentation/seg_0001.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/NoCn/00001.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnSeg02/00001.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnSeg05/00001.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnSeg07/00001.png" width="150"> |
| <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/ori/00002.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/segmentation/seg_0002.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/NoCn/00002.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnSeg02/00002.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnSeg05/00002.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnSeg07/00002.png" width="150"> |
| <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/ori/00003.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/segmentation/seg_0003.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/NoCn/00003.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnSeg02/00003.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnSeg05/00003.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnSeg07/00003.png" width="150"> |
| <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/ori/00004.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/segmentation/seg_0004.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/NoCn/00004.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnSeg02/00004.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnSeg05/00004.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnSeg07/00004.png" width="150"> |
| <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/ori/00005.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/segmentation/seg_0005.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/NoCn/00005.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnSeg02/00005.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnSeg05/00005.png" width="150"> | <img src="../../../docs/assets/projects/skyfall-gs/results/2026_03/FlowEditCnSeg07/00005.png" width="150"> |

---
## 결과 분석

| 상황 | 결과 |
|------|------|
| `cn_scale` 높음 | FlowEdit delta를 압도 → 이미지 파괴 |
| `cn_scale` 낮음 | CN 효과 미미 → FlowEdit 단독과 비슷 |
| 마스크 합성 방식 | CN 품질 자체가 낮아 의미 없음 |

FlowEdit 품질 자체는 우수하지만 ControlNet과 결합 시 trade-off가 너무 극단적.

추가로, FLUX용 Union ControlNet은 segmentation/normal 등 특정 condition에 특화된 모델이 아니라
하나의 모델이 모든 condition을 처리하는 범용 구조이기 때문에, 각 condition에 대한 응답 품질이 전용 모델에 비해 낮음.

=> **SD1.5 기반으로 전환**

SD1.5는 `control_v11p_sd15_seg` (segmentation 전용), `control_v11p_sd15_normalbae` (normal 전용) 등
**condition별 전용 ControlNet 모델**이 공개되어 있어 각 condition을 훨씬 정확하게 반영할 수 있음.
또한 SDEdit (img2img) 방식으로 `strength` 파라미터를 통해 원본 보존 정도를 세밀하게 조절 가능.

> SD 기반 sequential ControlNet: [2026_03_sd_edit](./2026_03_sd_edit.md)
