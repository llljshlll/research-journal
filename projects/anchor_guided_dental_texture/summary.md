# 1. 연구 개요

## 1.1 전체 구조: Phase 1 / Phase 2 / Inference

- Phase 1 — Gemini가 소규모 dental case 전체 파이프라인을 수행하여 pseudo-dataset과 SD3 학습용 training pairs 구축
- Phase 2 — Phase 1의 training pairs로 SD3(DiT 기반 모델) fine-tuning
- Inference — Gemini는 첫 anchor view 생성에만 1회 사용, 이후 모든 hole filling은 fine-tuned SD3가 전담

## 1.2 Input Representation: Semantic Dental Mesh

- Texture 없는 mesh이나 vertex 단위 anatomical label 보유
- Camera view 결정 시 segmentation map, tooth/gum label map 렌더링 가능
- 생성 image는 view마다 달라지지만, semantic mesh는 모든 view에서 동일한 geometry·label 제공 → multi-view structural consistency의 기준
- 이후 생성 모델의 condition: segmentation map, hole mask, partial texture rendering 등 mesh-derived 정보 활용

# 2. Phase 1: Pseudo-Dataset 생성

## 2.1 Step 1 — Gemini 기반 Realistic Anchor View 생성

- Textureless mesh를 frontal view에서 렌더링
- Rendered image와 segmentation map을 Gemini에 입력하여 realistic anchor view 생성
- Phase 1에서 Gemini의 역할: anchor view 생성뿐 아니라 이후 모든 novel view의 hole filling까지 전담 (pseudo-dataset 구축의 핵심 주체)
- Inference에서는 Gemini의 역할이 anchor view 생성 1회로 축소

## 2.2 Step 2 — Texel Gathering

- Anchor view(2D image)의 색상을 mesh surface로 projection
- 현재 camera pose에서 보이는 surface point를 image plane으로 projection 후 해당 pixel color를 texel에 할당
- 결과: anchor view에서 보이는 영역만 채워진 partial textured mesh
- 한계: 한 번의 view로 mesh 전체 커버 불가, 비가시 영역은 이후 hole로 잔존

## 2.3 Step 3 — Novel View Rendering & Hole Detection

- Partial textured mesh를 새로운 camera view에서 렌더링
- Anchor view에서 비가시였으나 현재 view에서 가시인 영역은 texture 미할당 빈 영역
- Hole의 정의: 현재 view에서 visible하지만 texture가 없는 영역 (occlusion 영역과 구분)
- 동일 view의 segmentation map을 함께 렌더링하여 다음 단계의 hole mask + semantic map condition으로 활용

## 2.4 Step 4 — Gemini 기반 Hole Filling (Training Pair 구축)

- Novel view의 partial render, hole mask, segmentation map을 Gemini에 입력
- Gemini가 hole이 채워진 완성 view 생성
- Training pair 구성 — Input: segmentation map + hole mask + partial render / Target: Gemini-completed view
- 목적: realistic mesh 생성이 아닌 SD3 학습용 데이터 구축
- Gemini 활용 범위 한정 이유: 높은 비용, 낮은 재현성, fine-tuning 불가능 → pseudo-supervision 생성에만 활용

## 2.5 Step 5 — Iterative Texture Update → Pseudo-Dataset 완성

- Gemini-completed view를 texel gathering으로 mesh에 재투영하여 texture 누적
- 업데이트된 mesh에서 다음 view 렌더링 시 새로운 hole 발생, Gemini가 재차 hole filling
- 모든 view에 대한 반복 수행으로 소규모 case 단위의 complete realistic dental mesh 완성
- Phase 1 최종 산출물: complete realistic dental mesh + SD3 학습용 training pairs (Phase 2의 유일한 데이터 소스)

# 3. Phase 2: SD3 Fine-Tuning

## 3.1 학습 설정

- 학습 데이터: Phase 1에서 구축한 training pairs
- 대상 모델: SD3 (DiT 기반 open-source model), fine-tuning 및 inference 제어 가능

## 3.2 Fine-Tuning 근거

- Gemini의 한계: dental realism은 높으나 비용·재현성·fine-tuning 불가 문제 존재
- SD3의 보완: 초기 dental realism 부족하나 Gemini pseudo-supervision 학습을 통해 해당 품질 모방 가능
- 소규모 데이터 학습 가능 근거: Gemini pseudo-supervision의 고품질

## 3.3 Fine-Tuning 결과

- Dental-domain semantic-aware hole filling 능력 확보
- Inference에서 Gemini의 hole filling 역할 대체

# 4. Inference

## 4.1 파이프라인

- Gemini가 첫 anchor view 생성 (1회 호출)
- 이후 모든 novel view의 hole filling은 fine-tuned SD3가 전담
- 생성된 view를 texel gathering으로 mesh에 projection, 여러 view에 대해 반복 수행

## 4.2 결과 및 의의

- Realistic하고 multi-view consistent한 textured dental mesh 생성
- Gemini의 realism, semantic mesh의 consistency, SD3의 scalability를 단계별로 최적 활용
