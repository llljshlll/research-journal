# Skyfall-GS Stage2: FLUX 기반 Refinement Ablation

## Stage1 결과에 대한 FLUX / FLUX + ControlNet 적용 실험

## 1. Experimental Context

Skyfall-GS Stage1은 6개의 고정된 camera view를 사용하여 3D Gaussian Splatting(3DGS)을 학습한다.  
  
Stage1 학습 이후, GS로부터 렌더링한 6개의 view 세트를 대상으로  
FLUX 및 FLUX + conditioning을 적용하여 view 일반화 특성과 conditioning 효과를 비교한다.  

## 2. View Definition

모든 실험 조건에서 동일하게 6개의 view를 사용한다.  

- Seen Views (6 views):  
  Stage1에서 3DGS 학습에 사용된 camera views  
  
- Unseen Views (6 views):  
  Stage1 학습에는 사용되지 않은 camera views로,  
  기존 view에서 각도를 변경하여 GS로부터 새롭게 렌더링함  

## 3. Compared Settings

다음 설정들에 대해 각각 6개의 view 결과를 비교한다.  
1. Seen Views + FLUX (no conditioning)
2. Unseen Views + FLUX (no conditioning)
3. Seen Views + FLUX + ControlNet (Depth)
4. Seen Views + FLUX + Normal conditioning (weight = 0.3)
5. Seen Views + FLUX + Normal conditioning (weight = 0.6)
6. Seen Views + FLUX + Normal conditioning (weight = 0.9)

## 4.1 Seen vs Unseen Views (FLUX, no conditioning)

### 4.1.1 GT / Stage1 GS Rendering / FLUX

| View | GTimage | rendering (Stage1 GS) | FLUX |
|------|---------|------------------------|------|
| View 1 | ![](../../docs/assets/projects/skyfall-gs/images/controlnet_inference_result_view_0.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_ori_view/render/00000.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_ori_view/refine/00000.png) |
| View 2 | ![](../../docs/assets/projects/skyfall-gs/images/controlnet_inference_result_view_1.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_ori_view/render/00001.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_ori_view/refine/00001.png) |
| View 3 | ![](../../docs/assets/projects/skyfall-gs/images/controlnet_inference_result_view_2.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_ori_view/render/00002.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_ori_view/refine/00002.png) |
| View 4 | ![](../../docs/assets/projects/skyfall-gs/images/controlnet_inference_result_view_3.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_ori_view/render/00003.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_ori_view/refine/00003.png) |
| View 5 | ![](../../docs/assets/projects/skyfall-gs/images/controlnet_inference_result_view_4.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_ori_view/render/00004.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_ori_view/refine/00004.png) |
| View 6 | ![](../../docs/assets/projects/skyfall-gs/images/controlnet_inference_result_view_5.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_ori_view/render/00005.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_ori_view/refine/00005.png) |


### 4.1.2 Unseen Views (Novel Views)

| View | Rendering (Stage1 GS) | FLUX |
|------|------------------------|------|
| View 1 | ![](../../docs/assets/projects/skyfall-gs/output_refine_new_view/render/00000.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_new_view/refine/00000.png) |
| View 2 | ![](../../docs/assets/projects/skyfall-gs/output_refine_new_view/render/00001.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_new_view/refine/00001.png) |
| View 3 | ![](../../docs/assets/projects/skyfall-gs/output_refine_new_view/render/00002.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_new_view/refine/00002.png) |
| View 4 | ![](../../docs/assets/projects/skyfall-gs/output_refine_new_view/render/00003.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_new_view/refine/00003.png) |
| View 5 | ![](../../docs/assets/projects/skyfall-gs/output_refine_new_view/render/00004.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_new_view/refine/00004.png) |
| View 6 | ![](../../docs/assets/projects/skyfall-gs/output_refine_new_view/render/00005.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_new_view/refine/00005.png) |


## 4.2 Depth Input vs FLUX + Depth ControlNet

| View | Input Depth | FLUX (no conditioning) | FLUX + Depth ControlNet |
|------|-------------|------------------------|-------------------------|
| View 1 | ![](../../docs/assets/projects/skyfall-gs/depths/depth_00.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_ori_view/refine/00000.png) | ![](../../docs/assets/projects/skyfall-gs/output_depth/refine/00000.png) |
| View 2 | ![](../../docs/assets/projects/skyfall-gs/depths/depth_01.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_ori_view/refine/00001.png) | ![](../../docs/assets/projects/skyfall-gs/output_depth/refine/00001.png) |
| View 3 | ![](../../docs/assets/projects/skyfall-gs/depths/depth_02.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_ori_view/refine/00002.png) | ![](../../docs/assets/projects/skyfall-gs/output_depth/refine/00002.png) |
| View 4 | ![](../../docs/assets/projects/skyfall-gs/depths/depth_03.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_ori_view/refine/00003.png) | ![](../../docs/assets/projects/skyfall-gs/output_depth/refine/00003.png) |
| View 5 | ![](../../docs/assets/projects/skyfall-gs/depths/depth_04.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_ori_view/refine/00004.png) | ![](../../docs/assets/projects/skyfall-gs/output_depth/refine/00004.png) |
| View 6 | ![](../../docs/assets/projects/skyfall-gs/depths/depth_05.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_ori_view/refine/00005.png) | ![](../../docs/assets/projects/skyfall-gs/output_depth/refine/00005.png) |


## 4.3 Normal Input vs FLUX / Normal-conditioned FLUX

| View | Input Normal | FLUX (no conditioning) | Normal w=0.3 | Normal w=0.6 | Normal w=0.9 |
|------|--------------|------------------------|--------------|--------------|--------------|
| View 1 | ![](../../docs/assets/projects/skyfall-gs/normals/controlnet_inference_result_nor_view_0.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_ori_view/refine/00000.png) | ![](../../docs/assets/projects/skyfall-gs/output_normal_0_3/refine/00000.png) | ![](../../docs/assets/projects/skyfall-gs/output_normal_0_6/refine/00000.png) | ![](../../docs/assets/projects/skyfall-gs/output_normal_0_9/refine/00000.png) |
| View 2 | ![](../../docs/assets/projects/skyfall-gs/normals/controlnet_inference_result_nor_view_1.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_ori_view/refine/00001.png) | ![](../../docs/assets/projects/skyfall-gs/output_normal_0_3/refine/00001.png) | ![](../../docs/assets/projects/skyfall-gs/output_normal_0_6/refine/00001.png) | ![](../../docs/assets/projects/skyfall-gs/output_normal_0_9/refine/00001.png) |
| View 3 | ![](../../docs/assets/projects/skyfall-gs/normals/controlnet_inference_result_nor_view_2.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_ori_view/refine/00002.png) | ![](../../docs/assets/projects/skyfall-gs/output_normal_0_3/refine/00002.png) | ![](../../docs/assets/projects/skyfall-gs/output_normal_0_6/refine/00002.png) | ![](../../docs/assets/projects/skyfall-gs/output_normal_0_9/refine/00002.png) |
| View 4 | ![](../../docs/assets/projects/skyfall-gs/normals/controlnet_inference_result_nor_view_3.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_ori_view/refine/00003.png) | ![](../../docs/assets/projects/skyfall-gs/output_normal_0_3/refine/00003.png) | ![](../../docs/assets/projects/skyfall-gs/output_normal_0_6/refine/00003.png) | ![](../../docs/assets/projects/skyfall-gs/output_normal_0_9/refine/00003.png) |
| View 5 | ![](../../docs/assets/projects/skyfall-gs/normals/controlnet_inference_result_nor_view_4.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_ori_view/refine/00004.png) | ![](../../docs/assets/projects/skyfall-gs/output_normal_0_3/refine/00004.png) | ![](../../docs/assets/projects/skyfall-gs/output_normal_0_6/refine/00004.png) | ![](../../docs/assets/projects/skyfall-gs/output_normal_0_9/refine/00004.png) |
| View 6 | ![](../../docs/assets/projects/skyfall-gs/normals/controlnet_inference_result_nor_view_5.png) | ![](../../docs/assets/projects/skyfall-gs/output_refine_ori_view/refine/00005.png) | ![](../../docs/assets/projects/skyfall-gs/output_normal_0_3/refine/00005.png) | ![](../../docs/assets/projects/skyfall-gs/output_normal_0_6/refine/00005.png) | ![](../../docs/assets/projects/skyfall-gs/output_normal_0_9/refine/00005.png) |
