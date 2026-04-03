### 1) Reference Attention (`attn_refview`)

핵심 목적:

- reference image에서 얻은 layer-wise feature memory를
- generation branch에 주입하는 것

#### 1-1. reference memory는 어떻게 만드나

reference image는 먼저:

```text
reference image -> VAE -> ref_latents
```

로 바뀝니다.

그다음 `unet_dual`이 reference branch로 한 번 forward되면서  
각 layer의 `norm_hidden_states`를:

```text
condition_embed_dict[layer_name]
```

에 저장합니다.

이때 저장되는 shape는 대략:

```text
B x (N_ref * L) x C
```

입니다.

즉 reference branch는 일종의 layer-wise memory bank를 만드는 단계입니다.

#### 1-2. generation 쪽 query는 어떻게 만드나

generation branch block 안에서는 `norm_hidden_states`를:

```text
(B*N_pbr*N_view) x L x C
-> B x N_pbr x (N_view*L) x C
```

로 보고, 여기서:

```text
[:, 0, ...]
```

즉 albedo branch만 뽑습니다.

그래서 query는:

```text
Q_ref ~ B x (N_view*L) x C
```

입니다.

#### 1-3. key/value는 어디서 오나

key/value는 reference branch memory:

```text
condition_embed_dict[layer_name]
```

에서 옵니다.

즉:

```text
K_ref, V_ref ~ B x (N_ref*L) x C
```

입니다.

#### 1-4. 누가 누구를 보나

이 attention은:

```text
현재 generation albedo 토큰
-> reference branch에서 저장해 둔 같은 layer의 feature memory
```

를 읽는 구조입니다.

즉 "reference image가 이 layer에서 어떤 feature를 갖고 있었는가"를 generation 쪽이 보는 것입니다.

#### 1-5. 출력은 어디로 가나

RefAttn 출력은 다시:

```text
(B*N_pbr*N_view) x L x C
```

형태로 reshape되어 `hidden_states`에 residual add 됩니다.

중요:

- query는 albedo branch 중심
- output은 전체 hidden state로 되돌아감

그래서 직접적으로는 albedo 중심이지만, 간접적으로는 MR에도 영향이 퍼질 수 있습니다.



---

### 2) Multi-view Attention (`attn_multiview`)

핵심 목적:

- 여러 view가 서로 직접 정보를 주고받게 하는 것

#### 2-1. 입력 shape는 어떻게 바꾸나

현재 hidden state:

```text
(B*N_pbr*N_view) x L x C
```

를:

```text
(B*N_pbr) x (N_view*L) x C
```

로 바꿉니다.

즉 같은 PBR 안에서 여러 view의 토큰들을 한 줄로 이어붙입니다.

#### 2-2. query/key/value는 어디서 오나

multiview attention은 self-attention 성격이라:

```text
Q_mv = projection(multiview_hidden_states)
K_mv = projection(multiview_hidden_states)
V_mv = projection(multiview_hidden_states)
```

입니다.

즉 여러 view를 이어붙인 토큰 시퀀스가 자기 자신을 봅니다.

#### 2-3. position index는 어디에 쓰이나

여기서 `position_voxel_indices`가 함께 들어갑니다.

즉 단순히 모든 view token이 다 섞이는 게 아니라:

- 비슷한 3D 위치를 보는 토큰들이
- 더 잘 대응되도록 정렬 정보가 추가됩니다

즉:

```text
front view의 어떤 토큰
<-> side view의 같은 3D 표면 위치 토큰
```

이 연결되도록 돕습니다.

#### 2-4. 출력은 어디로 가나

출력은 다시:

```text
(B*N_pbr*N_view) x L x C
```

로 reshape되어 hidden state에 residual add 됩니다.

즉 multiview attn은:

```text
view 내부 attention이 아니라
여러 view 사이 직접 attention
```

입니다.

---

### 3) Cross Attention (`attn2`)

핵심 목적:

- 생성 중인 hidden state가 PBR type conditioning token을 읽게 하는 것

#### 3-1. query는 어디서 오나

query는 현재 generation hidden state입니다.

shape:

```text
Q_cross ~ (B*N_pbr*N_view) x L x C
```

#### 3-2. key/value는 어디서 오나

key/value는 learned PBR token입니다.

현재 구현에서는:

- `learned_text_clip_albedo`
- `learned_text_clip_mr`

를 stack해서 `encoder_hidden_states`를 만듭니다.

shape는 대략:

```text
(B*N_pbr*N_view) x 77 x 1024
```

입니다.

즉 이건 text prompt처럼 쓰는 learned token sequence입니다.

#### 3-3. 누가 누구를 보나

즉 cross-attn은:

```text
generation hidden state
-> 해당 PBR type의 learned conditioning tokens
```

를 읽는 구조입니다.

쉽게 말하면:

- albedo branch는 albedo용 learned token을 읽고
- mr branch는 mr용 learned token을 읽습니다

#### 3-4. 출력

출력은 hidden state와 같은 shape로 나와 residual add 됩니다.

---

### 4) DINO Attention (`attn_dino`)

핵심 목적:

- reference image의 semantic / visual prior를 generation hidden state에 주입

#### 4-1. source는 무엇인가

reference image를 DINOv2에 넣어 hidden states를 얻습니다.

이 raw feature는 대략:

```text
B x L_dino x 1536
```

입니다.

그다음 `image_proj_model_dino`를 거쳐:

```text
B x L_dino' x 1024
```

로 projection합니다.

#### 4-2. query/key/value는 어디서 오나

query:

```text
Q_dino ~ (B*N_pbr*N_view) x L x C
```

즉 generation hidden state입니다.

key/value:

projection된 DINO token을 `N_pbr * N_view`만큼 repeat해서:

```text
K_dino, V_dino ~ (B*N_pbr*N_view) x L_dino' x C
```

형태로 만든 것입니다.

#### 4-3. 누가 누구를 보나

즉 이 attention은:

```text
generation hidden state
-> DINO semantic token
```

을 읽는 구조입니다.

reference attn과의 차이는:

- reference attn은 reference UNet branch feature memory를 읽음
- dino attn은 외부 DINO 인코더의 semantic token을 읽음

입니다.

#### 4-4. albedo만인가, MR도인가

DINO attention은 albedo branch만 따로 뽑지 않고  
hidden state 전체에 걸립니다.

즉:

- albedo에도 작용
- mr에도 작용

합니다.

출력은 마찬가지로 hidden state에 residual add 됩니다.

---

## 한 번에 요약

```text
Material-aware self attn:
  같은 PBR, 같은 view 내부 토큰끼리 self-attention

Reference attn:
  generation(albedo 중심) -> reference UNet feature memory

Multi-view attn:
  여러 view 토큰끼리 직접 attention

Cross attn:
  generation -> learned PBR token

DINO attn:
  generation -> DINO semantic token
```

즉 이 다섯 attention은 이름만 다른 게 아니라:

- query가 어디서 오고
- key/value가 어디서 오고
- 어떤 축을 펼쳐서 attention하느냐

가 각각 다릅니다.