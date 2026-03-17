질문과 답

1. 그러면 각 view들은 b, n, c, h, w에서 n이 view개수로 처리되는거지?, 그럼 position이랑 normal은 어캐 처리돼? 채널 수가 늘어나? 아니면 그냥 h, w에 concat돼?

답:

네. view 차원은 기본적으로 `n`으로 처리됩니다.

하지만 현재 코드에서 실제 생성 latent는 view 하나만 있는 게 아니라 PBR 차원까지 같이 묶여서 다뤄집니다.

대표 shape는:

- 생성 대상 latent: `B x N_pbr x N_view x C x H x W`
- normal latent: `B x N_view x C x H x W`
- position latent: `B x N_view x C x H x W`

그리고 [materialmvp/modules.py](/home/diglab/workspace/projects/MaterialMVP/materialmvp/modules.py) 의 `UNet2p5DConditionModel.forward()`에서:

- `embeds_normal`을 `unsqueeze(1).repeat(1, N_pbr, ...)`
- `embeds_position`도 같은 방식으로 `N_pbr`축으로 복제
- 그 다음 `torch.cat(sample, dim=-3)`로 concat

합니다.

즉 정리하면:

- `normal`, `position`은 `H, W` 방향 concat이 아님
- latent로 인코딩된 뒤 `channel(C)` 방향 concat임
- 그래서 `noise_in_channels=12`가 됩니다

개념적으로:

```text
sample = [noisy_pbr_latent | normal_latent | position_latent]
```

즉 view는 `N_view` 축으로 유지되고, normal/position은 채널 쪽으로 붙습니다.


2. material-aware self attention이랑 MCAA랑 똑같은거야?

답:

완전히 같은 말로 쓰면 안 됩니다.

- 논문/그림에서 말하는 `MCAA`는 개념 블록
- 코드에서 실제 self-attention 자리를 대체하는 건 `material-aware self attention` 또는 `MDA`에 가깝습니다

현재 코드에서는 plain self-attention 자리에 [materialmvp/modules.py](/home/diglab/workspace/projects/MaterialMVP/materialmvp/modules.py)의 `attn1`이 들어가고, `use_mda=True`일 때 이 `attn1`이 PBR-aware processor를 사용합니다.

즉 코드 기준으로는:

- `MCAA` = 논문 그림의 상위 개념
- `material-aware self attention` = 실제 구현에서 self-attention 자리를 대체하는 블록

이라고 보는 게 안전합니다.


3. albedo랑 MR branch는 어디서 합쳐져? reference, multivew, cross attention에서 다 나뉘어져서 attention돼?

답:

현재 코드는 `albedo UNet`, `MR UNet` 두 개가 따로 있는 구조가 아닙니다.

즉:

- backbone은 공유
- tensor 안에서 `N_pbr` 축으로 `albedo`, `mr`를 함께 들고 감
- 일부 attention processor와 learned token만 PBR-aware하게 처리

합니다.

그래서 "어디서 합쳐진다"기보다 처음부터 shared backbone 안에서 같이 흐른다고 보는 게 맞습니다.

다만 세부적으로 보면:

- `attn1(material-aware self attention)`은 `B x N_pbr x N_view x L x C` 구조를 직접 받음
- `multiview attn`도 `N_pbr`을 유지한 채 view들을 묶어서 attention
- `cross attn`도 `encoder_hidden_states`가 PBR별 token을 가짐
- `reference attn`은 예외적으로 albedo branch 기준으로 계산한 뒤 전체 hidden state에 residual로 다시 퍼짐

즉:

- `albedo`, `mr`는 처음부터 끝까지 완전 분리된 두 네트워크가 아님
- backbone 내부에서 함께 흐르되, 일부 attention 경로는 PBR-aware
- RefAttn은 특히 albedo branch 중심

입니다.

>> 그러면 모든 attention에서 albedo랑 mr은 따로 흘러? 즉, 영향을 안주고받아?

답:

완전히 따로 흘러서 서로 영향을 안 주는 구조는 아닙니다.

정확히는:

- tensor 차원상으로는 `N_pbr` 축이 분리되어 있어서 `albedo`, `mr`를 구분해서 들고 갑니다
- 하지만 backbone UNet은 공유합니다
- 그리고 같은 block 안에서 같은 hidden state 공간에서 같이 처리됩니다

즉:

- `완전 분리 네트워크`는 아님
- `PBR-aware separation`은 있음

입니다.

좀 더 구체적으로:

1. `attn1(material-aware self attn)`은 `B x N_pbr x N_view x L x C`를 직접 받으므로 PBR 구분이 반영됩니다
2. `multiview attn`도 `N_pbr`을 유지한 채 view들을 섞습니다
3. `cross attn`도 PBR별 learned token을 받습니다
4. `reference attn`은 albedo branch 기준으로 계산된 뒤 다시 전체 hidden state에 residual로 더해집니다

그래서 결과적으로:

- `albedo`, `mr`는 따로 token/분기를 가지지만
- shared backbone과 residual add 때문에 간접적인 영향은 주고받습니다

즉 "완전히 독립"도 아니고, "완전히 섞임"도 아닙니다.  
가장 정확한 표현은:

```text
shared backbone 위에서 PBR-aware하게 분리되어 흐르지만, 완전히 차단된 두 branch는 아니다.
```

>>> 그러면 attention으로 정보를 주고받지는 않는거지?

답:

완전히 안 주고받는다고 보면 안 됩니다.

정확히는:

- `albedo`와 `mr`는 `N_pbr` 축으로 구분되어 흐릅니다
- 그래서 "같은 토큰으로 완전히 섞여버리는 것"은 아닙니다
- 하지만 backbone은 공유하고, 일부 branch 출력은 전체 hidden state에 residual로 더해집니다

즉:

- `직접적으로 분리된 attention 경로`는 있음
- `간접적으로 서로 영향`은 있음

입니다.

예를 들어:

- `attn1`은 PBR-aware하게 작동하므로 albedo/mr를 구분해서 봅니다
- 그런데 block 출력은 결국 shared hidden state로 누적됩니다
- `RefAttn`은 albedo branch에서 계산되지만 결과가 다시 전체 hidden state에 더해집니다

그래서 "attention을 통해 전혀 정보를 안 주고받는다"는 건 아닙니다.  
더 정확한 표현은:

```text
PBR별로 구분된 표현을 유지하면서도, shared backbone과 residual add 때문에 간접적인 상호작용은 있다.
```

4. 모든 attention 블록에서 수행돼? down block, mid block, ujp block 전부 다?

답:

네. 현재 구현은 cross-attention이 있는 transformer block 전반에 적용합니다.

[materialmvp/modules.py](/home/diglab/workspace/projects/MaterialMVP/materialmvp/modules.py)의 `init_attention()`을 보면:

- `down_blocks`
- `mid_block`
- `up_blocks`

를 모두 순회하면서 `BasicTransformerBlock`을 `Basic2p5DTransformerBlock`으로 교체합니다.

즉 cross-attention이 달린 transformer block이면:

- down
- mid
- up

전부 확장된 2.5D block으로 바뀝니다.


5. 원래 기본 형태는 self attn -> cross attn -> FFN 인데 이 사이사이에 multi view attn, reference attn이 끼고, self attention이 material-aware self attention으로 치환된 거 맞아? 맞다면, 추가되고 치환된 attention들의 input, output 과 역할 알려줘

답:

네. 코드 기준으로 그렇게 이해하면 맞습니다.

원래 SD block:

```text
self attn -> cross attn -> FFN
```

현재 block:

```text
material-aware self attn
-> + reference attn
-> + multiview attn
-> + cross attn
-> + dino attn
-> FFN
```

각 블록의 입력/출력/역할은 아래와 같습니다.

`1) material-aware self attn (attn1)`

- 입력:
  - `norm_hidden_states`
  - shape 개념상 `B x N_pbr x N_view x L x C`
- 출력:
  - 같은 shape의 hidden state 보정값
- 역할:
  - plain self-attn 대신 albedo/mr 구분을 반영한 self-attn 수행

`2) reference attn (attn_refview)`

- 입력:
  - 현재 block의 albedo branch hidden state
  - reference branch에서 미리 cache된 `condition_embed_dict[layer_name]`
- 출력:
  - 다시 전체 `N_pbr x N_view` 형태로 reshape된 residual
- 역할:
  - reference image의 appearance/material 단서를 현재 생성 흐름에 주입

`3) multiview attn (attn_multiview)`

- 입력:
  - 여러 view를 한 축으로 펼친 hidden state
  - 필요 시 `position_voxel_indices`
- 출력:
  - view 간 정보가 섞인 hidden state residual
- 역할:
  - 앞/옆/뒤 view 사이 일관성 유지

`4) cross attn (attn2)`

- 입력:
  - hidden state
  - `encoder_hidden_states`
  - 여기엔 learned PBR token이 들어감
- 출력:
  - 조건 반영된 hidden state residual
- 역할:
  - PBR type에 맞는 conditional generation

`5) dino attn (attn_dino)`

- 입력:
  - hidden state
  - projected DINO hidden states
- 출력:
  - DINO feature가 반영된 residual
- 역할:
  - reference image의 semantic/appearance prior를 더 강하게 주입

`6) FFN`

- 입력:
  - 위 과정을 거친 hidden state
- 출력:
  - 최종 block output
- 역할:
  - transformer block의 표준 feed-forward refinement

즉 네 이해대로:

- self-attn은 material-aware self-attn으로 치환
- ref/mv/dino attention이 추가
- 대부분 residual add로 누적

입니다.


6. DinoAttn은 뭐야? 역할이 뭐야? reference image에서 DINO hidden state 추출한다는데, DINO hidden state 가 뭐야 이거 안쓰면 DinoAttn도 안써?

답:

네. `DinoAttn`은 DINOv2 feature를 cross-attention 형태로 주입하는 추가 attention branch입니다.

현재 코드에서:

- `self.use_dino=True`면 DINOv2를 로드
- reference image에서 hidden states 추출
- `image_proj_model_dino`로 cross-attn 차원에 맞게 projection
- 각 block에서 `attn_dino` 수행

합니다.

역할은:

- CLIP text token만으로는 부족한 reference image의 semantic feature 보강
- texture/material 관련 시각적 prior 보강

입니다.

그리고 맞습니다.

- `use_dino=False`면 DINO hidden state도 안 만들고
- `attn_dino` 경로도 사실상 사용하지 않습니다


7. 텐서 어떻게 변환되는지 흐름 그림으로 설명해줘(input -> VAE -> UNET(attention 별로) )

답:

코드 기준으로 단순화하면 아래 흐름입니다.

```text
[reference image]
  -> PIL
  -> VAE encode(ref_latents)                  : B x N_ref x C x H x W
  -> DINOv2(hidden states)                   : B x L_dino x C_dino

[normal maps for N views]
  -> PIL list
  -> VAE encode(normal_embeds)               : B x N_view x C x H x W

[position maps for N views]
  -> PIL list
  -> VAE encode(position_embeds)             : B x N_view x C x H x W
  -> voxel index 계산(position_voxel_indices)

[target noisy latent]
  -> B x N_pbr x N_view x C x H x W

----------------------------------------------

UNet 입력 조립:

sample
  = [noisy_latent
     | normal_embeds(repeated over N_pbr)
     | position_embeds(repeated over N_pbr)]

  -> concat on channel
  -> reshape to (B*N_pbr*N_view) x C_total x H x W

encoder_hidden_states
  -> learned_text_clip_albedo / learned_text_clip_mr
  -> B x N_pbr x 77 x 1024
  -> repeat over N_view

----------------------------------------------

각 transformer block:

norm
-> attn1(material-aware self attn)
-> + attn_refview(reference cache)
-> + attn_multiview(view tokens + position index)
-> + attn2(cross attn to learned PBR tokens)
-> + attn_dino(projected DINO hidden states)
-> FFN

----------------------------------------------

출력 latent
  -> reshape to [B, N_pbr, N_view, C, H, W]
  -> VAE decode
  -> first half = albedo views
  -> second half = mr views
```

>> 텐서가 각 transformer 블록 안에서 어떻게 변하는지 더 정확히 설명해줘

답:

하나의 transformer block 안에서 기준 shape를 잡고 보면 이해가 쉽습니다.

입력 hidden state를 먼저:

```text
(B * N_pbr * N_view) x L x C
```

라고 두겠습니다.

여기서:

- `B`: batch
- `N_pbr`: albedo, mr
- `N_view`: view 개수
- `L`: spatial token 수
- `C`: feature dim

입니다.

블록 내부 흐름은 대략 이렇습니다.

`1) norm`

```text
h0: (B*N_pbr*N_view) x L x C
-> norm_hidden_states
```

shape는 그대로입니다.

`2) material-aware self attn (attn1)`

코드에서 먼저:

```text
(B*N_pbr*N_view) x L x C
-> B x N_pbr x N_view x L x C
```

로 다시 묶습니다.

이 상태로 `attn1`에 넣고, 출력도 같은 구조로 받은 뒤 다시:

```text
-> (B*N_pbr*N_view) x L x C
```

로 펴서 residual add 합니다.

즉:

```text
h1 = h0 + Attn1(h0 grouped as B x N_pbr x N_view x L x C)
```

`3) reference attn (attn_refview)`

여기서는 `norm_hidden_states`를:

```text
(B*N_pbr*N_view) x L x C
-> B x N_pbr x (N_view*L) x C
```

로 보고, 그중 `[:, 0, ...]` 즉 albedo branch만 꺼냅니다.

그래서 RefAttn query는 대략:

```text
B x (N_view*L) x C
```

입니다.

reference branch에서 미리 만든 `condition_embed_dict[layer_name]`도:

```text
B x (N_ref*L) x C
```

형태입니다.

RefAttn 결과는 다시:

```text
-> (B*N_pbr*N_view) x L x C
```

로 reshape되어 전체 hidden state에 더해집니다.

즉:

```text
h2 = h1 + RefAttn(albedo_query, ref_condition)
```

`4) multiview attn (attn_multiview)`

여기서는 view들을 한 줄로 펴서:

```text
(B*N_pbr*N_view) x L x C
-> (B*N_pbr) x (N_view*L) x C
```

형태로 만듭니다.

즉 "같은 PBR 안에서 여러 view token들을 한 시퀀스로 연결"한 다음 attention합니다.

출력은 다시:

```text
-> (B*N_pbr*N_view) x L x C
```

로 돌아와 hidden state에 더해집니다.

즉:

```text
h3 = h2 + MVAttn(view-concatenated tokens)
```

>>> 이건 내 개념 이해가 부족한 거 같은데, (B*N_pbr*N_view) x L x C랑 (B*N_pbr) x (N_view*L) x C, B x (N_ref*L) x C의 차이를 모르겠어 어떻게 묶냐에 따라 attention을 할 때 차이가 생기나?

답:

네. **어떻게 reshape해서 묶느냐에 따라 attention이 "누구와 누구를 볼 수 있는지"가 달라집니다.**  
이게 핵심입니다.

각 shape의 의미를 분리해서 보면:

`1) (B*N_pbr*N_view) x L x C`

이건 "한 view씩 독립 sample처럼" 보는 형태입니다.

의미:

- 한 번의 attention이 기본적으로 `한 view 내부의 L개 token`만 봄
- 다른 view token은 같은 attention 안에서 직접 안 봄

즉:

```text
view별로 따로따로 attention
```

에 가깝습니다.

`2) (B*N_pbr) x (N_view*L) x C`

이건 같은 PBR 안의 여러 view token을 한 시퀀스로 이어붙인 형태입니다.

의미:

- 한 sample 안에 `view1의 L개 token + view2의 L개 token + ...`
  가 같이 들어감
- 그래서 attention이 view를 넘어 토큰을 볼 수 있음

즉:

```text
여러 view 사이 attention 가능
```

이 shape가 바로 multiview attention에 적합합니다.

`3) B x (N_ref*L) x C`

이건 reference branch에서 나온 여러 reference image token을 한 시퀀스로 묶은 것입니다.

의미:

- generation 쪽 query가 reference 쪽 token memory를 읽을 수 있음
- 즉 현재 생성 중인 hidden state가 "reference에서 추출된 token bank"를 보는 구조

즉:

```text
generation tokens -> reference tokens로 attention
```

입니다.

그래서 "reshape를 다르게 하면 왜 다르냐"에 대한 답은:

```text
attention은 같은 batch 안의 sequence 차원에 있는 token끼리만 직접 상호작용하기 때문
```

입니다.

예를 들어:

- `(B*N_pbr*N_view) x L x C`
  - 한 view 내부 토큰끼리만 self-attention
- `(B*N_pbr) x (N_view*L) x C`
  - 여러 view 토큰끼리 self-attention
- `B x (N_ref*L) x C`
  - reference token memory를 만들기 위한 구조

입니다.

한 줄 요약:

```text
reshape는 단순 모양 바꾸기가 아니라, attention이 허용하는 상호작용 범위를 바꾸는 작업이다.
```

`5) cross attn (attn2)`

이때 hidden state는 여전히:

```text
(B*N_pbr*N_view) x L x C
```

이고, 여기에 대응하는 `encoder_hidden_states`는 learned PBR token을 view별로 repeat한 것입니다.

대략:

```text
encoder_hidden_states_gen:
(B*N_pbr*N_view) x 77 x 1024
```
>>> 여기서 77 x 1024는 텍스트 임베딩인가?

답:

네. 정확히는 **text encoder가 기대하는 형태를 맞춘 learned text-like embedding**입니다.

현재 코드의:

- `learned_text_clip_albedo`
- `learned_text_clip_mr`
- `learned_text_clip_ref`

는 shape가 `77 x 1024`입니다.

이 의미는:

- `77`: Stable Diffusion/CLIP text token length
- `1024`: text embedding dim

입니다.

즉 자연어 문장을 tokenizer에 넣어 얻은 embedding은 아니지만,  
UNet 입장에서는 거의 **텍스트 임베딩처럼 쓰는 학습 가능한 token sequence**입니다.

그래서 답을 정확히 하면:

- `완전한 자연어 텍스트 임베딩`은 아니고
- `text embedding 자리에 들어가는 learned conditioning token`입니다

라고 하면 가장 정확합니다.

이게 cross-attention의 key/value 역할을 합니다.

출력은 다시 같은 shape:

```text
h4 = h3 + CrossAttn(h3, encoder_hidden_states_gen)
```

`6) dino attn (attn_dino)`

DINO hidden states는 먼저 projection을 거쳐:

```text
B x L_dino' x C
```
>>> 그러면 (B*N_pbr*N_view) x L x C가 프로젝션을 거쳐 B x L_dino' x C가 되는거야? 원래 뭐였는데 프로젝션을 거쳐서 저게 되는건데?

답:

아니요. 그건 다른 텐서입니다.

프로젝션되는 건 `(B*N_pbr*N_view) x L x C` hidden state가 아니라,  
**DINO에서 나온 feature**입니다.

흐름을 정확히 쓰면:

`1) reference image -> DINOv2`

먼저 reference image를 DINOv2에 넣어서 raw feature를 얻습니다.

대략:

```text
B x L_dino x 1536
```

입니다.

여기서:

- `L_dino`: DINO token 개수
- `1536`: DINO feature dim

입니다.

`2) DINO feature -> projection`

그 다음 `image_proj_model_dino`가 이걸:

```text
B x L_dino x 1536
-> B x L_dino' x 1024
```

로 바꿉니다.

즉 projection은:

- DINO feature dim `1536`
- UNet cross-attention dim `1024`

를 맞추기 위한 과정입니다.

`3) 그 다음 repeat`

projection된 DINO token을:

```text
B x L_dino' x 1024
-> (B*N_pbr*N_view) x L_dino' x 1024
```

형태로 repeat해서 각 생성 branch가 볼 수 있게 만듭니다.

그리고 이걸 `attn_dino`의 key/value로 씁니다.

반면 생성 쪽 hidden state는 계속:

```text
(B*N_pbr*N_view) x L x C
```

형태로 유지됩니다.

즉 네가 쓴 두 텐서는 같은 텐서가 아닙니다.

- `(B*N_pbr*N_view) x L x C`
  - 생성 UNet의 hidden state
- `B x L_dino' x C`
  - reference image에서 나온 DINO feature를 projection한 것

입니다.

가 되고, 이후 `N_pbr * N_view`만큼 반복돼:

```text
(B*N_pbr*N_view) x L_dino' x C
```

형태가 됩니다.

그리고 hidden state와 cross-attention해서:

```text
h5 = h4 + DinoAttn(h4, dino_hidden_states_proj)
```

가 됩니다.
>>>>그럼 dinoAttn이랑 reference attn 차이가 뭐야?

답:

둘 다 "reference image에서 온 정보"를 generation hidden state에 넣는다는 점은 같지만,  
**무엇을 source로 쓰는지와 어떤 정보를 넣는지가 다릅니다.**

가장 짧게 구분하면:

- `reference attn`
  - reference image를 **UNet reference branch로 흘려서 얻은 layer-wise feature memory**를 읽음
- `dinoAttn`
  - reference image를 **DINOv2로 인코딩한 semantic feature token**을 읽음

즉 source가 다릅니다.

`1) reference attn`

source:

- `ref_latents`
- `unet_dual` 또는 `unet`
- `condition_embed_dict[layer_name]`

흐름:

1. reference image를 VAE latent로 바꿈
2. reference branch UNet에 넣음
3. 각 layer에서 feature를 `condition_embed_dict[layer_name]`에 저장
4. generation branch의 같은 layer가 그걸 읽음

즉 reference attn은:

```text
"현재 UNet layer와 같은 표현 공간"의 reference feature memory
```

를 읽는 구조입니다.

역할:

- reference image의 appearance/material 단서를 layer-by-layer로 전달
- 현재 생성 중인 representation과 더 직접적으로 맞물리는 conditioning

특징:

- layer-specific
- UNet 내부 feature 기반
- albedo branch 중심으로 사용

`2) dinoAttn`

source:

- reference image
- DINOv2 encoder
- `image_proj_model_dino`

흐름:

1. reference image를 DINOv2로 인코딩
2. high-level token feature를 얻음
3. cross-attn 차원으로 projection
4. 각 block에서 hidden state가 이 token들을 읽음

즉 dinoAttn은:

```text
"UNet 내부 feature"가 아니라
"별도 비전 인코더가 만든 semantic token"
```

을 읽는 구조입니다.

역할:

- semantic prior 제공
- reference 이미지의 고수준 시각 특징 보강
- CLIP/learned token만으로 부족한 visual cue 보완

특징:

- layer-specific cache가 아님
- 외부 vision encoder feature 기반
- semantic / visual prior 쪽 성격이 강함

차이를 표처럼 정리하면:

```text
Reference Attn:
  source = reference branch UNet feature
  granularity = layer-wise cached feature
  role = appearance/material 단서 직접 주입

Dino Attn:
  source = DINOv2 feature token
  granularity = external semantic token
  role = semantic / visual prior 보강
```

조금 더 직관적으로 말하면:

- `reference attn`은 "reference 이미지를 현재 UNet과 비슷한 표현공간으로 변환해서 읽는 것"
- `dinoAttn`은 "reference 이미지를 별도 비전 모델의 의미 feature로 읽는 것"

입니다.

즉 둘은 중복이라기보다 보완 관계에 가깝습니다.

`7) FFN`

마지막으로:

```text
h6 = h5 + FFN(norm(h5))
```

가 block 출력입니다.

한 줄 요약:

```text
(B*N_pbr*N_view) x L x C
<-> 필요할 때만 B x N_pbr x N_view x L x C 또는 (B*N_pbr) x (N_view*L) x C 로 reshape
-> attention
-> 다시 (B*N_pbr*N_view) x L x C 로 복원
-> residual add
```

8. reference image에서 DINO hidden state 추출한다는데, DINO hidden state 가 뭐야

답:

`DINO hidden state`는 DINOv2 transformer가 reference image를 인코딩한 뒤 내부에서 내놓는 token feature입니다.

쉽게 말해:

- RGB 이미지를 바로 쓰는 게 아니라
- DINOv2가 이미지 내용을 고차원 feature token들로 바꾼 것

입니다.

이 token들은 보통:

- 물체 형상
- 영역별 시각 패턴
- semantic 구조

같은 정보를 압축해서 담고 있습니다.

현재 코드의 `Dino_v2.forward()`는 DINOv2의 출력 `[0]`을 받아서:

- `(b*n) x l x c`
  형태를
- `b x (n*l) x c`

형태로 다시 정리합니다.

즉 DINO hidden state는 "reference image를 보고 얻은 토큰 시퀀스 feature"라고 이해하면 됩니다.

>> DINO로 나온 feature의 텐서 차원은 어떻게돼?

답:

코드 기준으로 단계별로 다릅니다.

`1) DINOv2 raw output`

[materialmvp/modules.py](/home/diglab/workspace/projects/MaterialMVP/materialmvp/modules.py)의 `Dino_v2.forward()`에서 DINOv2의 출력 `[0]`을 받습니다.

입력 image가 tensor일 때 중간 shape는 대략:

```text
(B*N_ref) x L_dino x C_dino
```

입니다.

여기서 보통:

- `C_dino = 1536`

입니다. 코드에서도 `ImageProjModel(... clip_embeddings_dim=1536)`로 맞춰 둡니다.

그 다음 `rearrange`로:

```text
(B*N_ref) x L_dino x 1536
-> B x (N_ref * L_dino) x 1536
```

가 됩니다.

추론에서는 보통 reference image 1장이라:

```text
B x L_dino x 1536
```

처럼 보면 됩니다.

`2) projection 후`

그 다음 `image_proj_model_dino`를 통과하면 cross-attention 차원에 맞게 바뀝니다.

이 모델은:

- 입력: `1536`
- 출력: `4 * cross_attention_dim`

구조이고, 기본 cross attention dim은 `1024`입니다.

그래서 projection 후에는 대략:

```text
B x L_dino' x 1024
```

가 됩니다.

여기서 `L_dino'`는 원래 DINO token 수에 `clip_extra_context_tokens=4`가 반영된 길이입니다.

`3) attention 직전`

block 안에서는 이걸 다시 `N_pbr * N_view`만큼 repeat해서:

```text
(B*N_pbr*N_view) x L_dino' x 1024
```

형태로 `attn_dino`에 넣습니다.

9. position map`을 바탕으로 다중 해상도의 voxel index를 만들고, 이를 통해 view 간 비슷한 3D 위치가 서로 attention되도록 유도한다고 했는데 이게 무슨소리야? 정확히 설명해줘

답:

핵심은 "각 view의 픽셀들이 사실 3D 표면의 어떤 위치를 보고 있는지"를 이용해서, 서로 다른 view의 대응점을 찾는다는 뜻입니다.

position map의 각 픽셀은 대략:

```text
이 픽셀이 보고 있는 3D 위치 (x, y, z)
```

를 담고 있습니다.

현재 코드는 이 3D 위치를 바로 쓰지 않고, 여러 해상도의 voxel grid로 양자화합니다.

절차는:

1. position map을 작은 grid block으로 나눔
2. 각 block 안의 평균 3D 위치를 계산
3. 그 위치를 voxel index로 양자화
4. 여러 view에서 voxel index가 비슷한 token들은 "같은 표면 부근"이라고 간주

즉:

- front view의 어떤 패치
- side view의 어떤 패치

가 2D 좌표는 달라도, 3D 위치가 비슷하면 attention에서 서로 더 잘 연결됩니다.

이게 필요한 이유는 단순 multiview attention만 쓰면:

- 앞면의 한 점과 옆면의 전혀 다른 점이 섞일 수 있기 때문입니다

position-aware alignment를 쓰면:

```text
2D로는 멀지만 3D로는 같은 표면 위치
```

를 더 잘 매칭할 수 있습니다.

즉 "비슷한 3D 위치끼리 attention"이란 말은:

- 서로 다른 카메라 view에서도
- 실제 mesh 상 같은 위치를 보는 token들을
- 더 잘 대응시키도록 돕는다는 뜻입니다

>> 복셀 계산은 실제 mesh써서 해?

답:

직접적으로는 "mesh vertex/face를 바로 읽어서" 계산하는 게 아니라,  
**mesh를 렌더링해서 얻은 position map을 기반으로 계산**합니다.

즉 흐름은:

```text
mesh
-> renderer가 각 view의 position map 생성
-> position map의 각 픽셀에 3D 위치값 저장
-> 그 값을 grid/voxel로 양자화
```

입니다.

그래서 근본적으로는 실제 mesh geometry에서 나온 정보가 맞습니다.  
다만 voxel 계산 함수가 직접 mesh vertex를 순회하는 건 아니고,  
이미 렌더된 `position map` 위에서 계산합니다.

즉 답을 정확히 하면:

- `예, 결국 실제 mesh 기반 정보다`
- `하지만 직접 mesh를 쓰는 게 아니라 mesh에서 렌더된 position map을 쓴다`

입니다.

10. Consistency-Regularized Training을 통해 두 reference image를 가깝게 만들었다는데, 그러면 inference할 때에는 한 이미지만 사용하는거지?

답:

네. inference 때는 한 이미지만 씁니다.

학습 때만:

- reference A
- reference B

두 장을 써서 "조명이나 약간의 조건 변화가 있어도 결과 material은 비슷해야 한다"는 제약을 줍니다.

하지만 추론에서는 [textureGenPipeline.py](/home/diglab/workspace/projects/MaterialMVP/textureGenPipeline.py) 기준으로 보통 reference image 1장을 받아서 생성합니다.

즉 consistency-regularized training은:

- 추론 입력을 두 장으로 만들기 위한 것이 아니라
- 한 장만 들어와도 더 안정적으로 material을 뽑게 만들기 위한 학습 전략

입니다.


11. inpaint 할 때, mesh-aware vertex inpaint로 구조적인 빈칸을 먼저 줄이고, 2D inpaint로 남은 작은 hole을 마감한다고 하는데 정확히 어떻게 mesh-aware vertex inpaint가 구조적 빈칸 줄이고, 2D inpaint가 남은 작은 홀 마감하는지 정확한 원리 설명

답:

현재 코드에서 이 부분은 두 단계입니다.

`1) mesh-aware vertex inpaint`

[DifferentiableRenderer/MeshRender.py](/home/diglab/workspace/projects/MaterialMVP/DifferentiableRenderer/MeshRender.py)의 `uv_inpaint()`는 먼저:

```text
meshVerticeInpaint(texture_np, mask, vtx_pos, vtx_uv, pos_idx, uv_idx)
```

를 호출합니다.

이 함수는 C++ 확장이라 내부 세부는 여기서 바로 안 보이지만, 입력을 보면 의도가 명확합니다.

입력:

- 현재 UV texture
- 채워진 영역 mask
- vertex position
- vertex uv
- triangle topology

즉 단순 2D 이미지 보정이 아니라, mesh connectivity를 이용합니다.

의미는:

- UV 공간에서 떨어져 있어도 실제 mesh에서 이웃인 영역을 따라 값을 전파할 수 있고
- triangle 연결성을 따라 빈칸을 줄일 수 있다는 것입니다

그래서 이 단계는 특히:

- seam 양옆이 실제론 같은 surface인 경우
- UV가 끊겨 있어서 2D 이웃만으로는 연결이 안 되는 경우

에 유리합니다.

`2) 2D inpaint`

그 뒤 남은 hole은 OpenCV `cv2.inpaint(..., cv2.INPAINT_NS)`로 메웁니다.

이 단계는 mesh topology는 모르고, UV image 위에서만 동작합니다.

원리는:

- 이미 채워진 주변 픽셀의 색/구조를 따라
- 빈 영역으로 부드럽게 연장

하는 것입니다.

즉 역할 분담은:

- `mesh-aware vertex inpaint`
  - mesh 연결성을 따라 큰 구조적 빈칸부터 줄임
  - seam across propagation에 유리
- `2D inpaint`
  - 남은 작은 hole, 미세한 gap, 가장자리 틈을 시각적으로 마감

입니다.

한 줄 요약:

```text
먼저 mesh topology를 이용해 "surface 기준으로" 채우고,
그 다음 UV image 기준으로 "보이는 작은 구멍"을 마감한다.
```

>> mesh에 만들어낸 텍스처 입히는 방법 처음부터 끝까지 아주 자세히 설명해줘

답:

현재 파이프라인에서 "텍스처를 mesh에 입힌다"는 건 단순히 이미지 파일 하나 저장하는 게 아니라,  
최종적으로:

- UV texture image를 만들고
- 그 texture를 참조하는 OBJ/MTL을 저장하고
- 필요하면 GLB까지 변환

하는 전체 과정을 뜻합니다.

처음부터 끝까지 순서대로 쓰면 이렇습니다.

`1) mesh 준비`

[textureGenPipeline.py](/home/diglab/workspace/projects/MaterialMVP/textureGenPipeline.py) 에서:

- mesh를 로드
- 필요 시 remesh
- 필요 시 UV wrap

합니다.

여기서 중요한 건 bake를 하려면 mesh가:

- vertex position
- face topology
- UV coordinates

를 가져야 한다는 점입니다.

`2) renderer에 mesh 등록`

그 다음:

```python
self.render.load_mesh(mesh=mesh)
```

를 호출합니다.

이 단계에서 렌더러는 내부적으로:

- `vtx_pos`
- `pos_idx`
- `vtx_uv`
- `uv_idx`

를 받아 저장합니다.

그리고 UV texture space 쪽에서 쓸 보조 정보도 만듭니다.  
예를 들어 `self.tex_position`, `self.tex_grid` 같은 것들이 이후 bake 때 사용됩니다.

의미는:

- UV의 각 texel이 mesh 상 어디를 대표하는지
- texture 이미지의 어떤 위치가 어떤 표면에 대응하는지

를 렌더러가 알고 있게 만드는 단계입니다.

`3) multi-view 이미지 생성`

선택된 view들에 대해:

- `albedo` 이미지들
- `mr` 이미지들

을 생성합니다.

이 시점 결과는 아직 mesh에 붙은 게 아니라 그냥 view 이미지들입니다.

예:

```text
view_00_albedo.png
view_01_albedo.png
...
view_00_mr.png
...
```

`4) 각 view 이미지를 UV texture 공간으로 역투영`

그 다음 `bake_from_multiview()`가 각 view마다 `back_project()`를 부릅니다.

여기서 하는 일이 핵심입니다.

각 UV texel에 대해:

1. 그 texel이 대표하는 3D surface 위치를 가져오고
2. 현재 view 카메라로 다시 투영해서
3. image 상 어느 픽셀을 봐야 하는지 계산하고
4. depth / visibility / normal cosine을 검사해서
5. 유효하면 그 RGB 값을 texel에 채웁니다

즉:

```text
UV texel -> 3D point -> current view image pixel -> sampled RGB
```

입니다.

이게 "이미지를 mesh에 입힌다"의 실질적인 수학적 과정입니다.

`5) 각 view에서 얻은 texture 후보 merge`

한 view만으로는 전체 surface가 다 안 보이므로, 각 view에서 얻은 texture 후보들을 합칩니다.

이때:

- 정면에 가까운 surface일수록 큰 weight
- canonical view일수록 더 큰 prior weight 가능
- 이미 거의 다 칠해진 view는 skip

합니다.

최종적으로:

```text
merged_albedo_texture
merged_mr_texture
```

가 만들어집니다.

이 상태는 아직 hole이 있을 수 있습니다.

`6) hole inpaint`

merge 후 비어 있는 UV texel은:

- mesh-aware vertex inpaint
- OpenCV 2D inpaint

로 메웁니다.

즉 최종 texture image를 usable한 상태로 다듬습니다.

`7) renderer에 texture 설정`

그 다음 [textureGenPipeline.py](/home/diglab/workspace/projects/MaterialMVP/textureGenPipeline.py) 에서:

```python
self.render.set_texture(texture, force_set=True)
self.render.set_texture_mr(texture_mr)
```

를 호출합니다.

이 단계는 렌더러 내부 mesh 객체에:

- diffuse texture
- metallic-roughness texture

를 등록하는 단계입니다.

즉 이제 렌더러는:

- 어떤 mesh geometry를 쓸지
- 그 mesh가 어떤 UV를 쓸지
- 그 UV에 어떤 texture image가 연결되는지

를 모두 알고 있습니다.

`8) save_mesh()로 OBJ/MTL/texture 파일 저장`

그 다음:

```python
self.render.save_mesh(output_mesh_path, downsample=True)
```

가 호출됩니다.

이 함수는 내부적으로:

- mesh vertex / face / uv
- texture image
- metallic map
- roughness map

을 꺼내서 저장합니다.

저장 결과는 보통:

- `textured_mesh.obj`
- `textured_mesh.mtl`
- `textured_mesh.jpg`
- `textured_mesh_metallic.jpg`
- `textured_mesh_roughness.jpg`

형태입니다.

중요한 점은 OBJ 파일 안에는 texture 픽셀 데이터가 직접 들어가는 게 아니라:

- vertex
- face
- uv coordinate
- mtl 참조

가 들어가고,  
MTL 파일이 다시 texture image 파일 경로를 가리킨다는 것입니다.

즉 "텍스처를 mesh에 입힌다"는 건 결국:

```text
mesh geometry + uv coordinates + texture image file reference
```

를 함께 저장하는 것입니다.

`9) GLB 변환`

마지막으로 옵션이 켜져 있으면:

```python
convert_obj_to_glb(...)
```

를 호출합니다.

이 단계에서는 Blender를 이용해 OBJ/MTL/texture를 읽고,  
그걸 하나의 GLB asset으로 묶습니다.

즉 결과적으로:

- OBJ + MTL + texture 파일 세트
또는
- GLB 하나

로 최종 산출됩니다.

한 줄로 전체를 요약하면:

```text
multi-view로 생성한 2D albedo/mr 이미지를
mesh의 UV texel에 역투영해서 하나의 texture atlas로 만들고,
그 texture atlas를 UV 좌표를 통해 OBJ/GLB mesh에 연결해 저장한다.
```
