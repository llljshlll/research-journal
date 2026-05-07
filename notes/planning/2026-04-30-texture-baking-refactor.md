# Research Plan: Texture Baking Pipeline Refactor

**Date:** 2026-05-01  
**Source:** [[2026_04_29]]  
**Status:** Active

---

## 1. 현재 연구 문제 (Problem Statement)

현재 texture baking 파이프라인은 **splatting 기반 UV 업데이트 + post-processing** 구조로,
근본적인 매핑 오류를 후처리로 덮고 있어 i2i → inverse rendering loop의 신호가 왜곡된다.

### 핵심 증상
- unseen texel이 nearest-neighbor fill로 채워져 실제 미관측 영역을 식별할 수 없음
- count=0 texel이 "미관측"이 아닌 파이프라인 오류의 증거임에도 데이터 문제로 오해됨
- 서로 다른 anatomical region (치아 vs 잇몸)이 동일 texel로 매핑되는 UV 버그 존재

---

## 2. 제약 조건 (Constraints)

| 항목 | 내용 |
|------|------|
| 해상도 | 512는 현재 UV 분포 대비 부족 → 1024 또는 2048 필요 |
| UV 공간 메트릭 | screen space와 UV space의 metric이 달라 splatting 시 kernel 정규화 불가 |
| Seam 처리 | per-iteration dilation은 금지; unwrap 단계에서 padding으로 해결해야 함 |
| 검증 기준 | view4 white noise 소멸 + view3 cliff artifact 유지 여부로 이분 검증 |

---

## 3. 실패 사례 및 잘못된 가정 (Failure Cases)

### 3-1. count=0 해석 오류
- **기존 주장:** count=0은 미관측이며, nearest fill로 색이 채워진다
- **실제 원인:** 관측 가능한 영역임에도 baking 로직 오류 또는 sampling 방식 문제로 누락
- **영향:** post-processing이 파이프라인 오류를 은폐

### 3-2. 두 픽셀 → 하나의 texel 해석 오류
- **기존 주장:** grazing angle + UV 압축비 29:1로 인한 sampling sparsity
- **실제 원인:** 서로 다른 face (치아 vs 잇몸) → UV overlap (unwrap bug) 또는 ray hit 오류
- **영향:** geometry/UV 매핑 버그를 sampling 문제로 잘못 분류

### 3-3. Splatting 방식 자체의 구조적 문제
- camera pixel → UV → texel (scatter) 방향: quantization 충돌, aliasing, hole 발생
- 정규화된 kernel 없이 splat → texel overwrite / skip
- **결론:** texture baking에서 splatting은 부적절한 접근 방식

---

## 4. 다음 연구 방향 (Research Directions)

### Direction A: Gathering 기반 Baking으로 전환 (우선순위 1)
표준 방식인 **UV texel → screen color sampling (gathering)** 으로 전환.  
Blender, Substance, nvdiffrast 모두 동일 방식을 사용하며 hole이 발생하지 않음.

### Direction B: UV Unwrap 버그 수정 (우선순위 2)
치아 vs 잇몸이 동일 texel에 매핑되는 UV overlap 버그를 근본적으로 수정.  
xatlas padding ≈ 8 또는 Blender margin ≈ 8 적용.

### Direction C: Post-processing 완전 제거 (우선순위 3)
distance_transform_edt, binary_dilation, nearest neighbor fill 전부 제거.  
unseen texel은 초기값(magenta)으로 유지하여 i2i 대상 영역을 명확히 보존.

---

## 5. 구현 계획 (Implementation Plan)

### Step 1. Baking 구조 변경
- [ ] nvdiffrast UV-space rasterization 도입 (권장)
  - UV texel 좌표를 world space로 역매핑
  - 각 view에서 color를 screen sampling (gathering)
  - 대안: UV→world mapping precompute 후 reuse

### Step 2. Post-processing 제거
- [ ] `distance_transform_edt` 제거
- [ ] `binary_dilation` 제거
- [ ] `nearest neighbor fill` 제거
- [ ] unseen texel = 초기값 그대로 유지

### Step 3. UV Unwrap 검증 및 수정
- [ ] UV overlap 여부 시각화 (texel당 face count heat map)
- [ ] xatlas padding = 8 로 재설정
- [ ] ray hit 로직 점검 (depth buffer 기반 hit 검증)

### Step 4. 해상도 업그레이드
- [ ] 1024 × 1024 적용 (우선)
- [ ] coverage heat map으로 texel 활용도 확인 후 2048 여부 결정

---

## 6. 검증 프로토콜 (Verification Protocol)

### 설정
- 초기 texture = **magenta (1, 0, 1)**
- post-processing **OFF**
- 해상도 **1024**

### 각 view별 출력물
1. raw texture (fill 없음) — magenta 영역이 unseen texel
2. render
3. target (gemini)
4. unseen mask (magenta 영역 binary mask)

### 판정 기준
| 현상 | 기대 결과 | 판정 |
|------|-----------|------|
| view4 white noise | gathering 전환 후 소멸해야 함 | ☐ |
| view3 cliff artifact | 유지되어야 함 (실제 오류 신호) | ☐ |

두 현상이 분리되면 원인 분석 가능.

---

## 7. 참고 노트 (References)

- [[texture-baking]]
- [[inverse-rendering]]
- [[2026_04_28]] — 이전 미팅 (gemini init 이미지 등)
