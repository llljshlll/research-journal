# Freeze 마스크 방식 정리

## 이전 방식 — 흰색 편차 기반 (폐기)

### 개념

`FROZEN_TEXTURE` (기준 EXR)를 로드한 뒤, 각 UV 텍셀의 값이 흰색(1.0)에서
`FREEZE_THRESHOLD` 이상 벗어나면 "이미 최적화된 픽셀"로 판단해 고정.

```python
deviation = np.max(np.abs(frozen_np - 1.0), axis=2)
is_frozen = (deviation >= FREEZE_THRESHOLD).astype(np.float32)
```

gradient update 후 frozen 픽셀을 원래 값으로 덮어씌움:

```python
opt[albedo_key] = freeze_tensor * frozen_tensor + (1 - freeze_tensor) * updated
```

### 관련 상수

| 상수 | 역할 |
|---|---|
| `FROZEN_TEXTURE` | freeze 판단 기준 EXR |
| `FREEZE_THRESHOLD` | 흰색과의 편차 임계값 (기본 0.2~0.5) |
| `USE_FREEZE_INIT` | 1번 리파인에 freeze 적용 여부 |
| `USE_FREEZE_LOOP` | 루프 리파인에 freeze 적용 여부 |

### 문제점

- **shading 결과 자체가 흰색일 수 있다.** 치아 법랑질은 흰색에 가까운 색이므로,
  최적화 후에도 텍셀 값이 1.0에 가깝게 남을 수 있음.
- 이 경우 "아직 최적화 안 됨"과 "최적화 됐는데 흰색"을 구분할 수 없어
  freeze 마스크가 부정확해짐.
- 텍스처 값으로 최적화 여부를 판단하는 것 자체가 근본적으로 잘못된 접근.

---

## 이전 방식 — UV 커버리지 기반 (폐기)


### 개념

텍스처 값과 무관하게, **어떤 뷰에서 렌더링됐는지(= UV 텍셀이 카메라에 보였는지)**
를 별도 마스크 파일로 추적.

각 뷰 리파인 완료 후 해당 뷰의 UV 커버리지를 계산해 누적 마스크에 OR 합산.
마스크 값 `1.0` = frozen(고정), `0.0` = free(최적화 가능).

### UV 커버리지 계산 방법

1. UV 좌표 자체를 색상으로 갖는 텍스처 생성  
   텍셀 `(row, col)` → color `(u, v, 0)` 여기서 `u = col/W`, `v = (H-row)/H`

2. 해당 뷰 카메라로 ambient-only 씬 렌더  
   (directional 라이트 없음, constant ambient = 1.0, diffuse 재질)  
   → 렌더 결과 `(R, G) ≈ (U, V)` (라이팅 곱이 1이므로)

3. `alpha > MASK_ALPHA_THRESHOLD` 인 픽셀만 선택  
   → 해당 픽셀의 `(R, G)` 로 UV 좌표 복원  
   → 대응하는 텍셀 위치를 마스크에서 `1.0` 으로 마킹

4. 뷰가 쌓일수록 `np.maximum(current_mask, coverage)` 로 누적

```python
visible  = alpha > MASK_ALPHA_THRESHOLD
tex_col  = np.clip((u_vals[visible] * W).astype(int), 0, W - 1)
tex_row  = np.clip(((1.0 - v_vals[visible]) * H).astype(int), 0, H - 1)
mask[tex_row, tex_col] = 1.0
```

### 관련 상수

| 상수 | 역할 |
|---|---|
| `MASK_ALPHA_THRESHOLD` | UV 커버리지 판단용 alpha 임계값 (기본 0.5) |

- **낮출수록** → 희미하게 보이는 텍셀도 frozen 등록 → frozen 영역 증가  
- **높일수록** → 확실히 보이는 텍셀만 frozen → frozen 영역 감소

### 마스크 파일

`output/pipeline/{RUN_ID}/freeze_mask.exr`  
각 루프 완료 시마다 덮어씀. 뷰가 추가될수록 frozen 영역 단조 증가.

### 파이프라인 내 적용 시점

| 단계 | 동작 |
|---|---|
| STEP 1 (초기 리파인) | 마스크 전부 0 → freeze 없이 자유 최적화 |
| STEP 1 완료 후 | INIT_REF_VIEW_NAME 커버리지 계산 → 마스크 누적 저장 |
| STEP 4 (루프 리파인) | 누적 마스크로 freeze 적용 |
| STEP 4 완료 후 | 해당 루프 뷰 커버리지 계산 → 마스크 누적 저장 |

### 장점

- 텍스처 값에 의존하지 않음 → 흰색 치아도 정확히 frozen 처리
- 뷰별 UV 가시성을 직접 추적하므로, 실제로 최적화에 기여한 텍셀만 고정
- 마스크 파일이 별도로 누적 저장되어 파이프라인 재시작 시 활용 가능

### 문제점

- **V 축 이중 플립 버그**: OBJ UV 좌표(`v = (H-row)/H`)로 텍스처를 만들고 렌더 결과에서
  `(1-V)` 변환을 다시 적용 → 텍셀 위치가 수직으로 뒤집혀 마스크가 부정확해짐.
- **sparse 샘플링**: 스크린 픽셀 하나당 텍셀 하나만 마킹 → 텍스처 공간에 빈 구멍 발생.
- **간접 조명 gradient 누락 처리 불가**: PRB max_depth=8 렌더에서 간접 bounce로
  gradient가 비가시 텍셀에 흘러도, 이 방식으론 그 텍셀을 "보임"으로 잡지 못함.

---

## 현재 방식 — AOV integrator 기반 (사용 중)

### 개념

Mitsuba3의 `aov` integrator와 `aovs: uv:uv`를 사용해 **스크린 픽셀별 UV 좌표를 직접 추출**.
UV 텍스처를 별도로 만들 필요 없음.

### UV 커버리지 계산 방법

1. `aov` integrator (`aovs: uv:uv`, inner: `direct`) 로 렌더
   - 출력 텐서 shape: `(H, W, 6)` — `[R, G, B, alpha, U, V]`
   - `direct` integrator 사용 → 카메라에서 **직접 hit된 텍셀만** 커버리지에 포함
     (indirect bounce 텍셀 제외)

2. `alpha > COVERAGE_ALPHA_THRESHOLD` 인 픽셀만 선택
   → `(U, V)` 로 텍스처 공간 `(row, col)` 계산
   → 마스크에서 해당 텍셀 `= 1.0` 으로 마킹

3. `COVERAGE_DILATION` 픽셀만큼 dilation → 경계 텍셀 누락 보완

```python
visible = alpha > COVERAGE_ALPHA_THRESHOLD
tex_col = np.clip((u_vals[visible] * W).astype(int), 0, W - 1)
tex_row = np.clip((v_vals[visible] * H).astype(int), 0, H - 1)
#  ↑ (1-V) 뒤집기 없음: Mitsuba3 AOV 'uv'는 이미 이미지 좌표계(V=0 위, V=1 아래)로 출력
mask[tex_row, tex_col] = 1.0
```

### effective_freeze 계산 (run_refine 내부)

각 뷰 리파인 시작 시 아래 두 마스크를 합산해 실제 freeze 마스크 결정:

```
effective_freeze = max(accumulated_mask, 1 - current_coverage)
```

| 항목 | 의미 |
|---|---|
| `accumulated_mask` | 이전 뷰들에서 직접 hit된 텍셀 누적 → 이전 최적화 결과 보호 |
| `1 - current_coverage` | 현재 뷰에서 안 보이는 텍셀 → max_depth=8 gradient leakage 차단 |
| **업데이트 허용 텍셀** | 현재 뷰에서 보이면서 아직 accumulated_mask에 없는 텍셀만 |

### gradient leakage 문제 (미해결)

PRB max_depth=8 에서 간접 bounce 경로를 통해 **비가시 텍셀에도 gradient가 흐름**.
비가시 텍셀 값 자체는 freeze로 보호되지만, **가시 텍셀이 비가시 텍셀의 간접 조명을
보정하는 방향으로 최적화**되는 부작용 발생.

```
예) 카메라 레이 → 앞면 텍셀 B(보임) → bounce → 뒷면 텍셀 A(안 보임, frozen)
    → A 값이 고정된 상태에서 B가 "A의 간접광을 보정"하는 값으로 수렴
    → 다른 뷰에서 B를 보면 A의 간접광이 없어 B가 약간 이상하게 보임
```

**근본 해결**: `REFINE_MAX_DEPTH = 1` (direct illumination only)로 설정하면
간접 bounce 자체가 없어져 gradient leakage 완전 차단. 단, 간접 조명 품질 손실.

### 관련 상수

| 상수 | 역할 |
|---|---|
| `COVERAGE_ALPHA_THRESHOLD` | AOV 커버리지 판단용 alpha 임계값 (기본 0.5) |
| `COVERAGE_DILATION` | 경계 텍셀 누락 보완용 dilation 픽셀 수 (기본 2) |
| `REFINE_MAX_DEPTH` | PRB integrator max bounce depth (1=direct only, 8=full indirect) |

### 마스크 파일

`output/pipeline/{RUN_ID}/freeze_mask.exr`  
각 루프 완료 시마다 덮어씀. 뷰가 추가될수록 frozen 영역 단조 증가.

### 파이프라인 내 적용 시점

| 단계 | 동작 |
|---|---|
| STEP 1 (초기 리파인) | `freeze_mask_np=None` → `effective_freeze = 1 - current_coverage` only |
| STEP 1 완료 후 | AOV 커버리지 계산 → `accumulated_mask` 에 누적 저장 |
| STEP N (루프 리파인) | `effective_freeze = max(accumulated_mask, 1 - current_coverage)` |
| STEP N 완료 후 | AOV 커버리지 계산 → `accumulated_mask` 에 누적 저장 |
