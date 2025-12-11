GS 와 diffusion 의 조합 논문
우리와 같이 diffusion 모델을 쓰면서 GS로 뷰를 재구성하기 위해 논문 읽음

핵심 아이디어
: 

위성영상 -> 3D 도시 생성은 측면이 안보이고, 계절도 달라서 조명이랑 색감도 불일치함  
기존 방법
Sat-NeRF: geometry 흐림, 파사드 뭉개짐
CityDreamer / GaussianCity: 의미지도+height map 기반 → 텍스처 다소 synthetic, 구조 단순화
기존 NeRF/GS: satellite → ground-level view generalization 안 됨

(1) Appearance Modeling
멀티 뷰 위성 이미지가 서로 다른 날짜, 계절, 시간대에서 찍혀 있기 때문에 Multi-Date 위성 이미지의 조명/계절 변화 보정.
조명/날짜 차이 = 별도의 latent로 흡수 실제 underlying albedo/재질은 유지하도록 하는 모듈임

구현 방식 : 
논문은 WildGaussians 스타일로 세 가지 정보를 MLP에 넣어 색 보정 계수를 뽑음
* per-image embedding
  - 이미지 j마다 하나씩 존재.
  - “이 사진은 오후 3시, 겨울, 살짝 노랗고 그림자 긴 상태” 이런 걸 latent로 들고 있음.
* per-Gaussian embedding
  - Gaussian i마다 하나씩
  - “이 Gaussian은 그림자가 자주 걸리는 부분”, “나무라서 계절에 따라 색이 많이 바뀜” 같은 지역적 appearance 변화를 담음.
* 기본 색 정보
  - 0차 Spherical Harmonics(DC 성분)로 표현된 base color.
이 세 개를 MLP 𝑓에 넣어서 affine color transform 파라미터를 얻음(saturation, contrast 역할하는 scale과 밝기, 색상 정보 가지는 bais가 나옴)
(원래 3DGS가 SH를 통해 계산한 view-dependent color*scale + baise)가 (조명/날짜 보정까지 반영된 최종 색)으로 들어감


* 작은 MLP가 색상 affine transform (β, γ) 학습 → 각 이미지 조명 상태가 달라도 일관된 색성분을 유지하게 만듦.




