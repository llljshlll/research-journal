# 텍스처 채우기 연구 정리

**작성일:** 2026-05-06  
**연구자:** Seohyeon Jang  
**상태:** 진행 중

---

## 이 연구가 풀려는 문제

3D 모델(예: 치아)을 여러 각도에서 찍은 사진으로 텍스처(표면 색상)를 복원할 때,
**텍스처의 각 칸(texel)에 올바른 색을 채우는 방법**이 핵심 과제입니다.

```
3D 모델 표면 → UV 맵 (2D 평면으로 펼침) → 각 칸에 색 채우기
```

간단히 말하면: "3D 물체의 색을 사진에서 긁어와 2D 텍스처 지도에 정확히 붙이는 것"

---

## 현재 방식과 문제점

### 기존 방식: Splatting (뿌리기)
```
카메라 픽셀 → UV 좌표 계산 → 해당 texel에 색 기록
```
- 카메라에서 본 색을 텍스처 칸에 "뿌리는" 방식
- 직관적이지만 구조적 결함이 있음

### 발견된 문제들

**문제 1: 안 채워진 칸을 덮어버림**  
관측되지 않은 texel(unseen)을 주변 색으로 자동 채우는 후처리를 쓰고 있었는데,
이게 오히려 진짜 오류를 숨기고 있었음.

**문제 2: 서로 다른 부위가 같은 칸에 매핑됨**  
치아와 잇몸(다른 3D 면)이 같은 texel에 매핑되는 UV 버그가 존재.
처음엔 "카메라 각도가 좁아서 생기는 현상"으로 오해했지만,
실제론 UV unwrap(3D→2D 펼치기) 과정의 버그였음.

**문제 3: Splatting 자체의 한계**  
UV 공간과 화면 공간의 단위(metric)가 달라서,
정규화 없이 뿌리면 일부 칸은 겹쳐서 덮어쓰이고, 일부는 아예 빠짐.

---

## 해결 방향

### 핵심 전환: Splatting → Gathering (모으기)

| | Splatting (기존) | Gathering (새 방식) |
|---|---|---|
| 방향 | 카메라 픽셀 → texel | texel → 카메라 화면 |
| 방식 | 색을 뿌림 | 색을 가져옴 |
| 문제 | 충돌·누락 발생 | 모든 칸을 빠짐없이 처리 |
| 사용처 | (이 연구의 기존 방식) | Blender, Substance, nvdiffrast 표준 |

**Gathering 방식:** "texel이 먼저 자기 위치를 world 좌표로 역변환하고,
그 위치가 카메라에서 어디 픽셀에 보이는지 찾아서 색을 가져옴"

---

## 구체적인 할 일

### 1단계: Baking 구조 교체 ← **지금 여기**
- [ ] nvdiffrast UV-space rasterization 도입
  - texel 좌표 → world 좌표 역매핑
  - 각 view에서 해당 위치의 색을 screen sampling

### 2단계: 후처리 완전 제거
제거할 코드:
```python
# 아래 3가지 전부 제거
distance_transform_edt(...)   # 거리 변환
binary_dilation(...)          # 팽창 처리
baked[~filled] = baked[nearest...]  # nearest neighbor fill
```
- unseen texel은 초기값(magenta)으로 그대로 유지
- 그래야 i2i(image-to-image) 모델이 채워야 할 영역을 정확히 알 수 있음

### 3단계: UV Unwrap 버그 수정
- [ ] texel당 face 개수 heat map으로 overlap 시각화
- [ ] xatlas padding = 8 적용 (seam 처리)
- [ ] ray hit 로직 점검

### 4단계: 해상도 업그레이드
- [ ] 512 → 1024 (현재 UV 분포 대비 512는 부족)
- [ ] coverage heat map 확인 후 2048 여부 결정

---

## 검증 방법

**설정:**
- 초기 texture = **magenta (1, 0, 1)** — 안 채워진 칸이 분홍색으로 보임
- post-processing **OFF**
- 해상도 **1024**

**각 view마다 확인할 것:**
1. raw texture — magenta 영역이 진짜 unseen texel
2. render 결과
3. target(gemini 생성 이미지)
4. unseen mask

**판정 기준:**

| 현상 | 기대 결과 |
|------|-----------|
| view4에서 보이던 white noise | gathering 전환 후 **사라져야 함** |
| view3의 cliff artifact | **유지되어야 함** (진짜 오류 신호이므로) |

두 현상이 분리되면 각각의 원인을 독립적으로 분석할 수 있음.

---

## 현재 상태 한 줄 요약

> 기존 파이프라인은 splatting + 후처리로 UV 버그를 숨기고 있었음.  
> **nvdiffrast gathering 방식으로 재구성하고, 후처리를 제거하는 것이 현재 목표.**

---

## 참고
- [[2026_04_29]] — 이 문서의 핵심 분석 출처
- [[texture-baking]] — 텍스처 베이킹 개념
- [[inverse-rendering]] — 역렌더링 개념
