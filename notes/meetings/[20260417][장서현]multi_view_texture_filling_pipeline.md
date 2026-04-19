## Multi-view Texture Filling Pipeline

1. Nanobanana로 single-view 이미지 생성 후 inverse rendering으로 texture 추출
2. 점진적으로 뷰 전환하여 unseen view에서 비어있는 영역 확인
3. 기존 영역 mask 후 nanobanana로 추가 생성
4. 다시 inverse rendering → 반복

### 1. Single-view image generation and Texture extraction by inverse rendering

| Original Image | Nanobanana Image | Mitsuba Rendering | Inverse Rendering |
| --- | --- | --- | --- |
| ![](./20260416/reference_image1.webp) | ![](./20260416/nanobanana_image1.webp) | ![](./20260416/mitsuba_rendering_refine1.webp) | ![](./20260416/inverse_rendering_nanobanana1.webp) |


### 2. Progressive view change
카메라를 점진적으로 회전시켜 기존 이 전에 만든 텍스처를 다른 뷰에서 렌더링한 결과를 확인하였다.
**Progressively rotated view image**  
| Original view | view1 | view2 | 
| --- | --- | --- | 
| ![](./20260416/inverse_rendering_nanobanana1.webp) | ![](./20260417/rendering_view1.webp) | ![](./20260417/rendering_view2.webp) | 


### 3. Additional filling with Nanobanana

초기에는 기존 texture를 유지하기 위해  
unseen 영역만 mask로 열고 Nanobanana로 보완하는 방식을 사용하였다.

하지만 다음 문제가 발생했다:

- mask가 부분적으로 끊겨 생성됨 (fragmented)
- unseen 영역 외에도 기존 영역이 다른 view에서 깨짐
- 즉, local 기준(mask)으로는 multi-view inconsistency를 해결할 수 없음

**Mask visualization**  
<img src="./20260417/mask_visualization.png" width="420" />
*빨간 부분이 unseen 영역(mask 적용 영역)

→ 따라서 mask 기반 partial filling 대신,  
전체 이미지를 대상으로 prompt 기반 refinement로 전환하였다.

**Input vs Nanobanana result**  
| input image | nanobanana image |
| --- | --- |
| ![](./20260417/rendering_view1.webp) | ![](./20260417/nanobanana_image1.webp) | | ![](./20260417/rendering_view2.webp) | ![](./20260417/nanobanana_image2.webp) | 

사용한 프롬프트
```python
PROMPT = (
    "This is a dental/teeth image. "
    "Preserve the exact same tooth and gum shape, size, position, and overall geometry completely unchanged. "
    "Keep all well-formed enamel texture and correct tooth details exactly as they are. "
    
    "First, remove all small gray dots, dark speckles, and artifact-like noise on both teeth and gums. "
    "These small speckle artifacts must be completely cleaned and replaced with natural texture matching the surrounding area. "
    
    "Then, for the gum regions, correct any unnatural gray or dull coloration. "
    "Restore a natural, healthy pink gum tone with subtle variation. "
    
    "Ensure the cleaned areas and corrected regions blend seamlessly with surrounding texture, color, translucency, and shading. "
    "Do not oversmooth or blur the surface. "
    "Do not remove valid fine texture. "
    "Do not add, remove, move, or reshape any teeth or gums. Preserve the overall composition exactly."
)
```
+추후 실험해보면 좋을 사항 : image를 denoising한 이미지 한 장만 주지말고, segmentation이나 lighting map이나 다른 Map들도 같이 주면 더 섬세한 결과가 나올 수 있을 것으로 추정

### 4. Re-optimization by inverse rendering
보완 생성된 이미지를 target으로 사용하여 inverse rendering을 수행하고,  
UV texture를 업데이트하였다.

이때 기존 texture를 보호하기 위해 mask를 사용하였다.

- 기준: 초기 albedo가 흰색(1.0)이므로, deviation으로 판단
- `deviation < threshold` → 미최적화 (free)
- `deviation ≥ threshold` → 최적화됨 (freeze)

threshold는 0.2 / 0.5 / 0.7로 실험하였다.

| No Mask | 0.2 | 0.5 | 0.7 |
| --- | --- | --- | --- |
| ![](./20260417/no_mask_train_view.webp) | ![](./20260417/0_2_train_view.webp) | ![](./20260417/0_5_train_view.webp) | ![](./20260417/0_7_train_view.webp) |

### 5. Progressive view change2

| Original view | Updated UV rendered on original view |
| --- | --- |
| ![](./20260416/inverse_rendering_nanobanana1.webp) | ![](./20260417/no_mask_original_view.webp) |


4번에서 업데이트한 UV texture를 다시 1번의 뷰로 보았다.(기존의 텍스처가 많이 깨지지 않는지 확인해보기 위함)

마스크 없을 때, 0.2일 때, 0.5일 때, 0.7일 때 결과 비교 표
| No Mask | 0.2 | 0.5 | 0.7 |
| --- | --- | --- | --- |
| ![](./20260417/no_mask_original_view.webp) | ![](./20260417/0_2_original_view.webp) | ![](./20260417/0_5_original_view.webp) | ![](./20260417/0_7_original_view.webp) |

*마스크 없이 덧씌우면 이전 뷰가 다시 흐트러짐


### 6. 전체 텍스트 만들기
elevation -30°의 30도 간격으로 배치된 12view를 순회하며 위 과정(2~4)을 반복해보았다.
shading이 영향이 있을까 싶어 shading이 없도록 나노바나나로 만들어진 정면 뷰를 첫 뷰로 선택하였다.

<img src="./20260417/first_view.webp" width="420" />

| 0.2 | 0.5 | 0.7 |
| --- | --- | --- |
| ![](./20260417/result_0_2.webp) | ![](./20260417/result_0_5.webp) | ![](./20260417/result_0_7.webp) | 


mask로 학습된 영억을 고정해서 노이즈가 생김  
=> inverse rendering 과정에서 mask를 적용 안할 시 이전에 학습했던 텍스처가 망가짐. 마스크 적용시 이전 뷰에서 생성된 텍스처 일부 고정하면서 현재 텍스처가 제한적으로 생성되어, 노이즈 발생


mask 0.5 텍스처 전체 적용 결과
| Result 1 | Result 2 | Result 3 |
| --- | --- | --- |
| ![](./20260417/mask_0_5_result1.png) | ![](./20260417/mask_0_5_result2.png) | ![](./20260417/mask_0_5_result3.png) | 
