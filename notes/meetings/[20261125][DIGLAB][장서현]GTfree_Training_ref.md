# TRELLIS.2 semantic 조건 주입 reference

## 배경

치아 mesh에 TRELLIS.2를 적용해 보면, 생성된 텍스처의 segmentation은 vertex의 semantic label이 아니라 모델의 prior에 의존함. 그래서 일부 치아에서 semantic이 다르게 생성됨.

현재 상황:
- **semantic label 있음**: mesh vertex마다 잇몸인지, 몇 번 치아인지 명시적인 label이 있음
- **GT 텍스처 없음**: 학습에 쓸 정답 텍스처가 없음
- **도메인 한정**: 치아

목표는 semantic label을 기존 ControlNet 방식으로 TRELLIS.2에 주입해 학습시키는 것임. 단, GT 없이 학습해야 하므로 이때 참고할 만한 reference를 조사함.


## 1. Direct Preference Optimization (DPO) 방식
치아 mesh의 vertex마다 semantic label은 있지만, 그 mesh에 대응하는 정답 텍스처는 없음 
다만, 같은 mesh와 참조 이미지로 생성한 후보 A와 B를 보면, 어느 후보가 실제 치아·잇몸 경계를 더 잘 따르는지는 비교할 수 있음

DPO의 학습 데이터는 같은 입력 조건, 선호 후보 A, 비선호 후보 B. 학습 전 모델을 기준 모델로 고정하고, 학습 모델이 B에 비해 A를 생성하는 쪽으로 기준 모델보다 더 기울도록 업데이트 하는 방식임
A가 B보다 낫다는 비교 정보 사용, flow matching으로 백터의 방향을 학습하니까 더 그럴 수 있음
모델의 prior이 이미 어느정도 GT에 가까운 이미지를 만들어 낼 수 있는 능력이 있을 때 사용 가능.

### SAM 3D: 3Dfy Anything in Images (Meta, CVPR 2026)
- **참고 위치**: Appendix의 Algorithm 1 — SAM 3D Basic Alignment (Texture, Shape)
- **목적**: 사진 속 물체에 대해 여러 3D 후보를 만들고, 사람이 사진과 더 잘 맞는다고 선택한 후보로 3D 생성 모델을 개선함 (post-training 단계)
- **흐름**: ① 사람이 3D 후보 비교 → ② 선택된 후보로 지도 미세조정(SFT) → ③ 선택·탈락 후보 쌍으로 선호 학습
  - 수집 → SFT → DPO를 여러 번 반복하고, 품질 기준 α를 점점 올림

#### 1. 후보 생성
1. **여러 후보 생성**: 같은 입력에서 여러 후보를 생성함. 본문에서는 물체 하나당 후보 8개를 비교함.
2. **두 후보씩 비교**: 평가자가 원본 이미지를 참고해 두 3D 후보를 회전·확대하며 왼쪽 우세 / 오른쪽 우세 / 동등 중 하나를 선택함. 우세한 후보는 남기고 탈락 후보는 다음 후보로 교체하며, 동등이면 무작위로 하나를 남김.
3. **최종 승자의 품질 확인**: 모든 후보가 나빠도 상대적 1등은 생기므로, 마지막 승자가 학습에 쓸 만큼 좋은지 별도로 평가함. 품질 기준을 통과한 경우에만 선호 샘플 x_w로 채택함.
4. **DPO용 쌍 구성**: 채택된 승자와 탈락 후보를 짝지어 (x_w, x_l)을 구성함.

#### 2. SFT — 사람이 고른 후보 자체를 학습
품질 기준을 통과한 승자 x_w를 학습 타깃으로 DPO 전에 flow matching 학습을 먼저 수행함. Shape 모델은 선택된 형상을, texture 모델은 선택된 텍스처를 생성하도록 학습함. 이 단계에서 쓰는 정보는 "**이 입력에 대해 이 결과가 좋은 예시**"라는 것임.

#### 3. DPO — 승자와 패자의 차이를 학습
1. **후보별 velocity 예측**: 승자와 패자 각각의 noisy latent에 대해, 학습 모델과 고정된 SFT 기준 모델이 각각 velocity를 예측함.
2. **후보별 오차 계산**: 각 예측을 해당 후보의 flow matching 목표 velocity와 비교해, 승자와 패자 각각의 기준 모델 오차와 학습 모델 오차를 얻음.
3. **승자 쪽의 상대적 개선량을 높이도록 업데이트**: 후보 하나의 개선량을 I(x)=E_ref(x)-E_train(x)로 두면, DPO는 I(x_w)-I(x_l)이 커지도록 학습함.

즉, 사람이 같은 물체에서 texture A를 B보다 선호했다면, SFT 모델 대비 A의 velocity 예측 개선량이 B보다 커지도록 Texture & Refinement 모델을 업데이트함.

#### 우리 논문에 반영할 방법
**실제 vertex label을 후보 선택 기준으로 쓰기**
- 동일 mesh와 동일 reference image를 고정하고 seed만 바꿔 텍스처 후보를 생성한 뒤, 각 후보의 렌더링에 실제 치아·잇몸 경계선을 겹쳐 봄
- SFT용 후보와 DPO용 쌍의 채택 기준을 구분, SFT에는 단독으로도 학습 타깃으로 쓸 만큼 좋은 후보를 사용, DPO에는 후보 간 우열이 명확한 쌍을 사용
- → 정답 RGB 없이도 어느 후보에서 잇몸색이 실제 경계를 넘어갔는지 판단할 수 있음


### Flow-DPO (Improving Video Generation with Human Feedback, NeurIPS 2025)
**목적**: 같은 프롬프트에서 사람이 어떤 영상을 더 선호하는지 학습해, 사전학습된 비디오 생성 모델의 출력을 개선함

| 모델 | 구성 | 역할 |
|---|---|---|
| **기준 모델** | 원래의 비디오 생성 base model | LoRA 적용 전 예측 오차 측정. 가중치 고정 |
| **학습 모델** | 동일한 base model + LoRA | LoRA 적용 후 예측 오차 측정. LoRA 가중치만 업데이트 |

| 단계 | 학습 대상 | 학습 목표 |
|---|---|---|
| **1단계 · VideoReward 학습** | 사전학습된 **Qwen2-VL-2B 기반 평가 모델** | 사람이 어떤 영상을 더 선호하는지 예측 |
| **2단계 · Flow-DPO 학습** | 사전학습된 **비디오 생성 모델에 추가한 LoRA** | VideoReward가 선호한 영상 쪽으로 생성 모델의 예측을 조정 |

#### 1. VideoReward 학습과 영상 순위 결정
1. **영상 쌍 구성**: 같은 프롬프트로 서로 다른 생성 모델이 만든 영상 A와 B를 한 쌍으로 구성함.
2. **사람의 선호 수집**: 전문 평가자가 시각 품질(VQ), 움직임 품질(MQ), 텍스트 일치도(TA) 각각에 대해 A 우세 / 동률 / B 우세 중 하나를 선택함. 영상 쌍 하나당 선호 label 3개를 수집함.
3. **VideoReward 학습**: 사람의 선호 label로 Qwen2-VL-2B 기반 평가 모델을 학습함. VideoReward는 프롬프트와 영상 하나를 입력받아 세 기준의 점수를 출력함.
4. **Flow-DPO용 승자·패자 결정**: VideoReward로 A와 B를 각각 채점함. 세 기준을 함께 쓰는 실험에서는 영상별 세 점수의 산술평균이 높은 영상을 선호 영상 x_w로, 낮은 영상을 비선호 영상 x_l로 지정함.

이렇게 얻은 (프롬프트, 선호 영상, 비선호 영상)이 Flow-DPO의 학습 데이터임. 학습에는 어느 영상이 이겼는지만 쓰고, VideoReward의 점수 차이는 loss에 직접 넣지 않음.

#### 2. Flow-DPO 학습
원래 생성 모델 대비, 선호 영상에서의 예측이 비선호 영상에서의 예측보다 상대적으로 더 개선되도록 LoRA를 학습함 (여기서 예측은 velocity 예측).

**① 영상마다 정답 velocity 생성**
영상 latent를 x, random noise를 ε이라고 하면, 둘을 섞은 모델 입력은 x_t=(1-t)x+tε임. 이 직선 경로에서 t가 변할 때 latent가 변하는 방향과 크기, 즉 정답 velocity는 v=ε-x임.

**② 각 모델의 예측을 같은 정답과 비교**
영상 하나에 대해 기준 모델과 학습 모델에 같은 입력을 주고, 각 모델이 정답 velocity를 얼마나 정확히 맞혔는지 측정함.
즉, 기존 모델이 만든 velocity를 이용해 모델 오차 구함 여기서 말하는 정답 velocity는 Flow-DPO의 학습 데이터로 만든 데이터를 말함

- 학습 모델 오차: E_train(x) = ‖v − v_θ(x_t, c, t)‖²
- 기준 모델 오차: E_ref(x) = ‖v − v_ref(x_t, c, t)‖²
- 개선량: I(x) = E_ref(x) − E_train(x)

(c는 조건(프롬프트), v_θ는 학습 모델, v_ref는 고정된 기준 모델의 velocity 예측)

**③ 선호 영상과 비선호 영상의 개선량 비교**

| 영상 | 원래 모델 오차 | LoRA 적용 모델 오차 | 개선량 |
|---|---:|---:|---:|
| 선호 영상 x_w | 10 | 7 | **3** |
| 비선호 영상 x_l | 6 | 5 | **1** |

선호 영상의 최종 오차(7)는 비선호 영상(5)보다 크지만, 원래 모델 대비 개선량은 선호 영상이 2만큼 더 큼. Flow-DPO가 loss에 쓰는 값은 이 차이(3 − 1 = 2)임.

논문의 loss는 다음을 최소화함.

- L = −log σ( −(β/2) · [ (E_train(x_w) − E_ref(x_w)) − (E_train(x_l) − E_ref(x_l)) ] )
- 개선량으로 쓰면 L = −log σ( (β/2) · (I(x_w) − I(x_l)) )

I(x_w) − I(x_l)이 클수록 σ 안의 값이 커져 loss가 작아짐. 위 예시라면 σ 안의 값은 (β/2)·2.

- **β는 시점과 상관없는 상수**(논문 β=500): Diffusion-DPO를 flow로 그대로 옮기면 β_t = β(1−t)²가 되는데, 이 경우 noise가 큰 시점에 학습이 쏠려 reward hacking이 생겨서 상수 β로 바꿈
- **같은 noise, 같은 시점**: 선호·비선호 영상에 같은 noise와 시점을 쓰지만, 원래 latent가 다르므로 정답 velocity도 각각 달라짐. 따라서 이상적인 정답 영상이 없어도 두 후보 각각의 정답 velocity는 계산할 수 있음
- **LoRA를 쓴 이유**: 전체 파라미터를 fine-tuning하면 성능이 떨어지거나 모델이 붕괴함. DPO를 너무 오래 학습해도 품질이 나빠짐(limitation)


#### 우리 논문에 반영할 방법
- 이 논문은 해당 loss로만 LoRA 학습 전체를 직접 수행함
- 우리도 controlNet이나 lora로, 해당 loss를 처음부터 사용하여 base model fine tuning

### Mesh-RFT (Enhancing Mesh Generation via Fine-grained Reinforcement Fine-Tuning, NeurIPS 2025)
포인트 클라우드를 보고 mesh생성하는 모델
**Mesh DPO**: 같은 point cloud에서 생성한 두 mesh를 비교하되, 선호 mesh의 좋은 면과 비선호 mesh의 나쁜 면에 해당하는 토큰만 손실에 반영, 기존 모델에 post-training
- 선호 쌍을 사람 없이 객관적 지표로만 자동생성

#### 1. 같은 point cloud에서 선호 쌍 구성
사전학습된 모델에 같은 point cloud P를 넣어 후보 mesh 8개를 생성함. 가능한 28개 조합을 비교하고, 한쪽이 다음 세 지표 모두에서 엄격하게 우수한 경우에만 학습 쌍으로 채택함. 지표 하나라도 동률이거나 우열이 엇갈리는 쌍은 제외함.

| 지표 | 선호 기준 | 측정 대상 |
|---|---|---|
| BER | 낮을수록 좋음 | 면 하나에만 연결된 경계 edge의 비율. 구멍이나 표면 단절을 파악하는 지표 |
| TS | 높을수록 좋음 | 삼각형들을 사각형으로 병합했을 때의 변환 비율과 사각형의 품질 |
| HD | 낮을수록 좋음 | 생성 mesh에서 샘플링한 점들과 입력 point cloud 사이의 기하학적 차이 |

#### 2. 선호 쌍의 각 mesh에 면별 마스크 생성
전체적으로 더 좋은 M+에도 M-보다 품질이 낮은 부분이 존재함. 그래서 전체 순위와 별도로 두 mesh의 삼각형 면을 각각 평가함.
- 각 삼각형이 사각형으로 병합되는지 확인
- 병합된 사각형의 내각이 90°에 가까운 정도, 변 길이 비율, 인접 사각형과의 비율 일관성을 평가
- → 병합에 성공하고 품질 점수가 임계값을 넘으면 해당 삼각형에 1, 그 외에는 0을 부여

삼각형 하나는 정점 3개 × XYZ 좌표 3개 = 9개 토큰으로 표현되므로, 면 하나의 라벨을 그 면에 대응하는 좌표 토큰 9개에 적용함.

| 후보 | 면별 품질 판정 | 실제 손실에 쓰는 마스크 | 선택된 면 |
|---|---|---|---|
| 선호 mesh (M+) | A면 좋음 `1`, B면 나쁨 `0` | 그대로 `[1, 0]` | A면 토큰 9개 |
| 비선호 mesh (M-) | C면 좋음 `1`, D면 나쁨 `0` | 뒤집어서 `[0, 1]` | D면 토큰 9개 |

B와 C는 loss에서 제외됨. 
즉, 선호 mesh의 좋은면, 비선호 mesh의 나쁜면 중심으로 학습 신호 줌
(선호 mesh에서는 좋은 면을 더 잘 생성하도록, 비선호 mesh에서는 나쁜 면을 덜 생성하도록)

#### 3. 같은 토큰을 학습 모델과 기준 모델이 각각 채점
- **두 모델**: 같은 사전학습 체크포인트에서 시작함. Reference는 고정하고, Policy는 이 체크포인트로 초기화해 fine-tuning함 (논문에 LoRA·ControlNet 같은 adapter 사용 언급은 없음)
- **채점 방식**: 이 모델은 이전 좌표 토큰으로 다음 좌표 토큰을 생성하는 모델임. 그래서 두 모델에 같은 point cloud와 같은 후보 mesh 토큰 시퀀스를 넣으면, 각 토큰 위치에서 "앞선 토큰들이 주어졌을 때 후보에 실제로 등장한 다음 좌표 토큰에 부여한 확률"이 계산됨
- **점수 집계**: 마스크로 선택된 위치의 확률을 합치고, Policy 합계 ÷ Reference 합계에 로그를 취함
  - 확률을 먼저 합산하고 그 비율에 로그를 취한다는 점이 중요함. 일반적인 autoregressive DPO의 시퀀스 점수는 토큰별 로그 확률을 합산함
  - 따라서 논문의 Masked DPO는 토큰 선택뿐 아니라 점수 집계 방식도 바뀌어야 함

- L+ = log( M+의 좋은 면 토큰에서 Policy 확률 합 ÷ Reference 확률 합 )
- L− = log( M−의 나쁜 면 토큰에서 Policy 확률 합 ÷ Reference 확률 합 )
- loss = −log σ( β·(L+ − L−) )

즉, 두 값 모두 Reference 대비 Policy의 비율이고, L+ − L−가 클수록 손실이 작아짐. 따라서 선호 mesh의 좋은 면은 Reference보다 높은 확률을, 비선호 mesh의 나쁜 면은 Reference보다 낮은 확률을 주는 방향으로 작동함.

#### 우리 논문에 반영할 방법
- 후보 A와 B의 차이가 16번 치아의 잇몸 경계에 집중되어 있다면, 두 후보의 전체 texture latent에 동일한 선호를 적용하는 방법과 16번 경계에 대응하는 영역에 학습 가중치를 집중하는 방법을 비교할 수 있음
- vertex 라벨은 어느 위치에 선호 신호를 적용할지를 제공함. 어느 후보의 외관이 더 나은지는 별도의 판단이 필요함
- Mesh-RFT는 autoregressive 모델이라 "토큰 확률 합"을 씀. TRELLIS.2는 flow 모델이라 이 식을 그대로 쓸 수 없고, 대신 Flow-DPO 식의 velocity 오차를 복셀 단위로 마스킹하는 형태로 바꿔야 함


## 2. Reward 모델 기반 조건 일치 loss

### ControlNet++
생성 결과 이미지를 segmentation 모델로 예측한 뒤, 그 예측과 입력 조건으로 준 segmentation mask의 차이를 loss(segmentation loss)로 줘서 segmentation 조건을 반영함.

- **reward 모델**: 사전학습된 segmentation 모델(UperNet-R50)을 고정해서 사용. loss는 픽셀별 cross-entropy
- **효율적인 reward 계산**: noise에서 끝까지 생성하면 메모리가 너무 큼. 그래서 학습 이미지에 작은 noise(t ≤ t_thre, t_thre=200)를 더한 뒤 한 번만 denoise한 이미지 x'_0로 segmentation loss를 계산함
- **전체 loss**: L_train + λ·L_reward (segmentation은 λ=0.5). reward loss만 쓰면 이미지가 심하게 망가져서 원래 diffusion loss를 같이 씀
- **학습 대상**: base 모델과 reward 모델은 고정하고 ControlNet만 학습
- **reward hacking 확인**: 학습에는 일부러 약한 segmentation 모델을, 평가에는 더 강한 모델(Mask2Former)을 씀. reward 모델이 강할수록 조건 일치도도 올라감

#### 우리 논문에 반영할 방법
ControlNet++의 학습 방식을 그대로 쓰지 않고, segmentation loss만 가져와 DPO 학습에 더함. 모델은 flow matching 기반 TRELLIS.2임.

- **전체 loss**: L = L_DPO + λ·L_seg
- **GT 이미지 대신 DPO 후보 사용**: ControlNet++는 학습 이미지(GT)에 noise를 더했지만, 우리는 GT가 없으므로 모델이 생성한 후보 텍스처 latent x_w를 그 자리에 씀 (x_l에도 적용할지는 ablation 필요)
  - x_t = (1−t)·x_w + t·ε
  - flow matching의 한 번 추정: x̂0 = x_t − t·v_θ(x_t, c, t)
- **segmentation loss 계산 경로**: x̂0 → 텍스처 디코더 → 복셀 PBR → 원본 mesh에 입혀 렌더링 → segmentation 모델 → 같은 뷰에서 렌더링한 vertex semantic label map과 cross-entropy
  - vertex semantic label이 ControlNet++의 입력 조건 mask 역할을 함
  - 우리는 2D 이미지가 아니라 복셀이므로 렌더링한 이미지에서 비교함. 6개 정도의 뷰에서 비교한 뒤 loss를 적용 (뷰 개수와 시점은 ablation 필요)
  - 이 경로 전체가 미분 가능해야 함 (디코더, 렌더링 포함)
- **작은 t에서만 적용**: t가 크면 x̂0 추정이 부정확해지므로, ControlNet++처럼 noise가 작은 구간에서만 L_seg를 걸음 (threshold는 ablation 필요)
- **L_train 역할**: ControlNet++는 diffusion loss로 원래 생성 능력을 유지했지만, 우리는 GT가 없어 이 loss를 그대로 쓸 수 없음. DPO loss의 기준 모델 항이 base 모델에서 너무 멀어지지 않게 잡아줄 것으로 봄. 부족하면 x_w에 대한 flow matching loss(SAM 3D의 SFT와 같은 형태) 추가를 고려
- **segmentation 모델**: 렌더링된 치아·잇몸 이미지를 분할하는 미분 가능한 모델이 필요함. ControlNet++처럼 학습용과 평가용 모델을 분리해 reward hacking 여부를 확인


## 3. Vertex semantic 기반 치아·잇몸 색 분포 loss
(추가 예정)
