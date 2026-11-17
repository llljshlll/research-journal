# Conv vs Attention

공간 정보를 섞는 두 가지 대표 방식. "어디를 보고 얼마나 가중할지"를 정하는 방법이 다르다.

## 한 줄 요약

- **Conv**: 고정된 로컬 창을 모든 위치에 똑같이 적용 (가중치가 입력과 무관).
- **Attention**: 입력 내용에 따라 가중치를 동적으로 계산해 전역을 참조 (가중치가 입력에 의존).

## 비교

| | Conv | Attention |
|---|---|---|
| 수용 영역 | 로컬 (커널 크기), 층을 쌓아야 넓어짐 | 한 층에서 전역 |
| 가중치 | 고정 커널 (content-independent) | Q·K로 동적 계산 (content-dependent) |
| 귀납 편향 | 강함: locality, translation equivariance | 약함: 구조를 데이터로 학습 |
| 비용 | `O(N)` | `O(N²)` (토큰 수 N) |
| 위치 정보 | 구조에 내장 | positional embedding 필요 → [[patchify-flatten]] |
| 데이터 요구 | 적어도 잘 됨 | 많이 필요 |

## 왜 둘 다 쓰나

- 이미지는 로컬 구조가 강해서 conv가 효율적이고, 전역 일관성(구도, 멀티뷰 대응)은 attention이 잘한다.
- **Stable Diffusion U-Net**: ResBlock(conv)로 로컬 처리 + 낮은 해상도에서 attention으로 전역/텍스트 조건 처리.
- **DiT / FLUX**: conv를 빼고 patchify + [[transformer-block]]만 사용. 데이터와 규모가 충분하면 단순한 구조가 더 확장성이 좋다.
- [[vae]] 인코더/디코더는 보통 conv 기반 (고해상도에서 attention은 비쌈).

## 원리 관점

- Conv = 모든 위치가 같은 [[linear]]를 로컬 패치에 적용한 것. 가중치 공유가 곧 translation equivariance.
- Attention = 위치별로 가중치가 바뀌는 "동적 conv". 전역 범위의 커널을 입력이 직접 결정한다.
- 그래서 attention은 [[multi-view-consistency]]처럼 **떨어진 위치 사이의 대응**이 필요한 문제에 유리하다.

## Related

- [[attention]]
- [[transformer-block]]
- [[overview|Stable Diffusion]]
- [[flux_overview|FLUX]]
