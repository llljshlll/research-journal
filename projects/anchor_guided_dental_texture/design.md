**Phase 1은 pseudo-dataset 생성 단계입니다.**

먼저 textureless dental mesh를 frontal view에서 렌더링하고, 해당 rendered image와 segmentation map을 Gemini에 입력하여 realistic anchor view를 생성합니다.

이 anchor view는 texel gathering 방식으로 mesh surface에 projection되어 초기 partial texture를 형성합니다.

이후 camera view를 변경하면 hole이 나타나는데, Phase 1에서는 이 hole filling도 Gemini가 직접 수행합니다.

이 과정을 소규모 dental case에 대해 모든 view에서 반복하여 complete pseudo-dataset을 만듭니다.

Phase 1 완료 결과에서 SD3 학습용 training pairs를 추출합니다.

Input은 segmentation map + hole mask + partial render, Target은 Gemini-completed view입니다.

**Phase 2는 SD3 fine-tuning 단계입니다.**

Phase 1 training pairs로 SD3, 즉 DiT 기반 모델을 fine-tuning합니다.

학습된 SD3는 dental-domain semantic-aware hole filling 능력을 갖추게 됩니다.

**Inference는 Phase 2 완료 후의 구조입니다.**

Gemini가 첫 anchor view를 생성하고, 이후의 모든 hole filling은 학습된 SD3가 담당합니다.

생성된 view는 다시 mesh surface에 projection되고, 이 과정을 여러 view에 대해 반복하여 최종적으로 realistic하고 multi-view consistent한 textured dental mesh를 생성합니다.


## Slide 21. Method Overview: Anchor-guided Dental Texture Synthesis

<!-- [수정] Phase 1 / Phase 2 / Inference 3단 구조로 전면 재작성 -->

이제 본 연구의 method를 설명드리겠습니다.

앞서 설명한 것처럼, 본 연구는 Phase 1, Phase 2, Inference 세 단계로 구성됩니다.

Phase 1은 Gemini가 소규모 dental case 전체 파이프라인을 담당해 pseudo-dataset과 training pairs를 만드는 단계, Phase 2는 이 training pairs로 SD3를 fine-tuning하는 단계, Inference는 학습된 SD3가 hole filling을 담당하는 단계입니다.

지금부터 각 단계를 순서대로 살펴보겠습니다.

---

## Slide 22. Input Representation: Semantic Dental Mesh

본 연구의 입력은 texture가 없는 dental mesh입니다.

하지만 단순한 geometry만 있는 mesh가 아니라, 각 vertex에 anatomical label이 존재합니다.

따라서 camera view가 정해지면 해당 view에서 segmentation map과 tooth/gum label map을 렌더링할 수 있습니다.

이 점이 본 연구에서 중요합니다.

Gemini나 open diffusion model이 생성하는 image는 view마다 달라질 수 있지만, semantic dental mesh는 모든 view에서 동일한 geometry와 anatomical label을 제공합니다.

따라서 semantic mesh는 단순 input condition이 아니라, multi-view texture synthesis에서 structural consistency를 유지하기 위한 기준으로 사용됩니다.

이후 단계에서 생성 모델은 단순히 image만 보고 생성하는 것이 아니라, segmentation map, hole mask, partial texture rendering과 같은 mesh-derived conditions를 함께 사용하게 됩니다.

---

## Slide 23. Phase 1 — Step 1: Gemini-based Realistic Anchor View Generation

<!-- [수정] 제목 변경 + Phase 1에서 Gemini의 full role 명시 -->

Phase 1의 첫 번째 단계는 Gemini-based realistic anchor view generation입니다.

Textureless dental mesh를 frontal view에서 렌더링하고, 이 rendered image와 segmentation map을 Gemini에 입력하여 realistic anchor view를 생성합니다.

여기서 Gemini의 역할을 명확히 해야 합니다.

Phase 1에서 Gemini는 anchor view 생성, 즉 이 단계뿐 아니라, 이후 모든 novel view에서 발생하는 hole filling도 직접 수행합니다.

즉, Phase 1에서 Gemini는 전체 파이프라인의 데이터 생성을 담당하는 주체입니다.

소규모 dental case에 대해 Gemini가 이 역할을 수행하여 complete pseudo-dataset을 구축합니다.

반면 Inference에서는 Gemini가 첫 anchor view 생성에만 사용됩니다.

이후의 hole filling은 Phase 1 데이터로 학습된 SD3가 담당합니다.

이 구분이 본 연구의 핵심 전략입니다.

---

## Slide 24. Phase 1 — Step 2: Texel Gathering from Anchor View

<!-- [수정] 제목 변경 (내용 동일) -->

두 번째 단계는 texel gathering입니다.

Gemini가 생성한 anchor view는 2D image이기 때문에, 이를 그대로 두면 textured mesh를 만들 수 없습니다.

따라서 생성된 image의 색을 mesh surface로 다시 projection해야 합니다.

구체적으로는 현재 camera pose에서 보이는 mesh surface point를 image plane으로 projection하고, 해당 image pixel의 color를 sample합니다.

그 다음 sample된 color를 해당 texel에 할당합니다.

이 과정을 통해 anchor view에서 보이는 영역에만 texture가 들어간 partial textured mesh가 만들어집니다.

중요한 점은 한 번의 view로 mesh 전체를 채울 수 없다는 것입니다.

정면 view에서 보이지 않는 옆면이나 안쪽 영역은 여전히 texture가 없는 상태로 남습니다.

이 영역은 이후 novel view rendering 단계에서 hole로 나타나게 됩니다.

---

## Slide 25. Phase 1 — Step 3: Novel View Rendering and Hole Detection

<!-- [수정] 제목 변경 (내용 동일) -->

세 번째 단계는 novel view rendering and hole detection입니다.

앞 단계에서 만든 partial textured mesh를 새로운 camera view에서 렌더링하면, anchor view에서는 보이지 않았지만 현재 view에서는 보이는 surface region이 나타납니다.

이 영역은 mesh geometry상으로는 존재하지만 아직 texture가 할당되지 않았기 때문에 partial render에서는 비어 있는 영역, 즉 hole로 나타납니다.

이때 hole은 단순히 보이지 않는 occlusion 영역이 아니라, 현재 view에서 visible하지만 아직 texture가 없는 영역입니다.

우리는 이 영역을 hole mask로 정의합니다.

또한 같은 camera view에서 segmentation map을 함께 렌더링할 수 있습니다.

따라서 다음 단계의 generation model은 partial render만 보는 것이 아니라, hole mask와 semantic map을 함께 condition으로 받아 어떤 치아 영역을 어떤 appearance로 채워야 하는지 판단하게 됩니다.

---

## Slide 26. Phase 1 — Step 4: Gemini-based Hole Filling (Pseudo-dataset 생성)

<!-- [수정] 전면 재작성. "SD3 hole filling"에서 "Phase 1: Gemini hole filling"으로 변경. training pair 구성이 핵심 -->

Phase 1의 네 번째 단계는 Gemini가 hole filling을 직접 수행하는 단계입니다.

이 단계에서 중요한 것은, Slide 23에서 Gemini가 anchor view를 생성한 것처럼, 여기서는 novel view에서 발생한 hole도 Gemini가 직접 채운다는 점입니다.

새로운 camera view에서 렌더링하면 partial render, hole mask, segmentation map이 생성됩니다.

Phase 1에서는 이 조건들을 Gemini에 입력하여 hole이 채워진 완성된 view를 생성합니다.

이 과정의 핵심 목적은 단순히 realistic mesh를 만드는 것이 아니라, SD3 학습을 위한 training pairs를 구축하는 것입니다.

Input은 segmentation map + hole mask + partial render이고, Target은 Gemini가 완성한 이 view 이미지입니다.

즉, Gemini의 hole filling 결과가 SD3의 학습 target이 됩니다.

소규모 dental case에 대해 이 과정을 모든 view에서 반복하면 SD3 학습에 충분한 training pairs가 구축됩니다.

Inference에서는 이 역할을 Phase 2에서 학습된 SD3가 대신합니다.

Gemini는 비용이 높고 재현성이 낮으며 직접 fine-tuning이 불가능하기 때문에, Phase 1에서 pseudo-supervision을 만드는 데만 활용하고 이후에는 학습된 open model로 대체합니다.

---

## Slide 27. Phase 1 — Step 5: Iterative Texture Update → Pseudo-dataset Complete

<!-- [수정] 전면 재작성. Phase 1 루프 완성 + pseudo-dataset 산출물 강조 -->

Phase 1의 다섯 번째이자 마지막 단계는 iterative texture update입니다.

Gemini가 완성한 novel view를 texel gathering으로 mesh surface에 projection하여 texture를 누적합니다.

업데이트된 mesh에서 다음 view를 렌더링하면 새로운 hole이 나타나고, 다시 Gemini가 채웁니다.

이 루프를 모든 view에 대해 반복하면 하나의 complete realistic dental mesh가 완성됩니다.

이 과정을 소규모 dental case에 대해 수행하면 Phase 1 pseudo-dataset이 완성됩니다.

Phase 1의 핵심 산출물은 두 가지입니다.

첫째, 소규모 케이스의 complete realistic dental mesh입니다.

둘째, SD3 학습을 위한 training pairs입니다.

이 training pairs가 Phase 2 SD3 fine-tuning의 유일한 학습 데이터 소스입니다.

---

## Slide 28. Phase 2: SD3 Fine-tuning on Pseudo-dataset

<!-- [수정] 전면 재작성. "Training Strategy" → "Phase 2: SD3 Fine-tuning". Phase 1→2 연결, SD3 명시, ~20케이스 bootstrapping -->

Phase 2는 Phase 1에서 구축한 pseudo-dataset으로 SD3를 fine-tuning하는 단계입니다.

Phase 1에서 Gemini가 소규모 dental case 전체 파이프라인을 실행하여 training pairs를 생성했습니다.

Phase 2에서는 이 training pairs로 SD3를 dental texture completion 태스크에 맞게 fine-tuning합니다.

SD3는 DiT(Diffusion Transformer) 기반의 open-source 모델로, fine-tuning과 inference 제어가 가능합니다.

Gemini는 dental realism이 높지만 비용이 크고 재현성이 낮으며 fine-tuning이 불가능합니다.

SD3는 처음에는 dental realism이 부족하지만, Gemini가 만든 pseudo-supervision으로 학습하면 그 품질을 모방할 수 있습니다.

소규모 데이터만으로도 fine-tuning이 가능한 이유는 Gemini의 pseudo-supervision이 고품질이기 때문입니다.

Phase 2 학습이 완료되면 Inference에서는 Gemini를 hole filling에 사용하지 않습니다.

Gemini는 첫 anchor view 생성에만 한 번 호출되고, 이후 모든 hole filling은 fine-tuned SD3가 담당합니다.

이 구조를 통해 Gemini의 realism, semantic mesh의 consistency, SD3의 scalability를 각 단계에서 최적으로 활용합니다.
