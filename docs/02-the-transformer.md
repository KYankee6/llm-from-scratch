# 파트 2: 트랜스포머

이 파트가 워크숍의 핵심입니다. PyTorch로 전체 GPT 모델 아키텍처를 처음부터 작성합니다.

## 큰 그림

GPT는 **자기회귀 언어 모델**입니다. 토큰 시퀀스가 주어지면 다음 토큰을 예측합니다. 이 예측을 루프로 쌓으면 텍스트 생성이 됩니다.

아키텍처는 동일한 **트랜스포머 블록**을 여러 층 쌓은 구조입니다. 각 블록에는 다음 요소가 들어갑니다.

1. **멀티헤드 셀프 어텐션** — 각 토큰이 이전 토큰 전체를 볼 수 있게 합니다
2. **피드포워드 네트워크(MLP)** — 각 위치를 독립적으로 처리합니다
3. **잔차 연결** — 각 하위 층의 출력에 입력을 다시 더합니다
4. **레이어 정규화** — 학습을 안정화합니다

## 작성하기: `model.py`

scratchpad 안에 `model.py`라는 새 파일을 만듭니다. 이 섹션을 읽으며 클래스를 하나씩 추가하세요. 끝까지 작성하면 파일에는 `GPTConfig`, `CausalSelfAttention`, `MLP`, `Block`, `GPT`가 들어갑니다.

### 설정

```python
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 65       # 문자 단위: Shakespeare의 고유 문자 65개
    block_size: int = 256      # 최대 시퀀스 길이(컨텍스트 창)
    n_layer: int = 6           # 트랜스포머 블록 수
    n_head: int = 6            # 어텐션 헤드 수
    n_embd: int = 384          # 임베딩 차원
```

`vocab_size`는 토크나이저에서 옵니다(Shakespeare의 경우 문자 65개). `block_size`는 모델이 한 번에 볼 수 있는 최대 토큰 수입니다. `n_embd`는 모델의 폭입니다. 모든 은닉 상태는 이 크기의 벡터입니다.

### 임베딩

```python
import torch
import torch.nn as nn

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),   # 토큰 임베딩
            wpe = nn.Embedding(config.block_size, config.n_embd),   # 위치 임베딩
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying: 출력 투영이 토큰 임베딩과 가중치를 공유합니다
        self.transformer.wte.weight = self.lm_head.weight
```

두 개의 임베딩 테이블이 있습니다.

- **`wte`**(word token embedding): 각 토큰 ID를 학습되는 벡터로 매핑합니다. 크기: `[65, 384]`
- **`wpe`**(word position embedding): 각 위치(0부터 255)를 학습되는 벡터로 매핑합니다. 크기: `[256, 384]`

**Weight tying**: 토큰 → 임베딩으로 매핑하는 같은 행렬을 출력에서 임베딩 → logits로 매핑할 때도 재사용합니다(전치된 형태). 이렇게 하면 파라미터가 줄고 학습이 좋아집니다. 토큰의 입력 표현과 출력 표현이 일관되도록 강제되기 때문입니다. 여기서는 어휘가 65개라 절약량이 작지만, 큰 어휘에서는 중요하며 표준적인 방식입니다.

### 순전파

```python
    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)

        tok_emb = self.transformer.wte(idx)    # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)    # (T, n_embd)
        x = tok_emb + pos_emb                  # (B, T, n_embd) — 브로드캐스팅으로 위치 정보가 더해집니다

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)               # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss
```

```
토큰 ID (B, T)
    │
    ▼
┌─────────┐     ┌─────────┐
│   wte   │     │   wpe   │
│ [65,384]│     │[256,384]│
└─────────┘     └─────────┘
    │               │
    ▼               ▼
  tok_emb    +   pos_emb      → x (B, T, 384)
                                  │
                                  ▼
                          ┌──────────────┐
                          │ Block × 6    │
                          └──────────────┘
                                  │
                                  ▼
                          ┌──────────────┐
                          │  LayerNorm   │
                          │   lm_head    │  Linear: 384 → 65
                          └──────────────┘
                                  │
                                  ▼
                          logits (B, T, 65)
```

위치 임베딩은 토큰 임베딩에 더해집니다. 이것이 모델이 단어 순서를 알게 되는 방식입니다. 위치 정보가 없다면 "the dog bit the man"과 "the man bit the dog"는 똑같아 보입니다.

### 셀프 어텐션

셀프 어텐션은 각 토큰이 시퀀스 안의 모든 이전 토큰에 주의를 기울일 수 있게 하는 메커니즘입니다.

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)  # Q, K, V 투영
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)       # 출력 투영
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 멀티헤드용 reshape: (B, T, C) → (B, n_head, T, head_dim)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # causal mask가 적용된 attention(각 토큰은 이전 토큰만 볼 수 있음)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
```

```
x (B, T, 384)
    │
    ▼
┌─────────┐
│  c_attn │  하나의 Linear → Q, K, V로 나눔
└─────────┘
    │
    ▼
┌─────────────────────────────┐
│  6개 헤드로 분리            │  각 헤드: (B, T, 64)
│                             │
│  Q @ K^T / sqrt(64)         │  유사도 점수
│  미래 위치 마스킹           │  causal: 뒤가 아니라 앞만 봄
│  softmax → 가중치           │
│  weights @ V                │  가중합
│                             │
│  head1  head2  ...  head6   │
└─────────────────────────────┘
    │
    ▼  모든 헤드 연결
┌─────────┐
│  c_proj │  다시 384차원으로 투영
└─────────┘
    │
    ▼
출력 (B, T, 384)
```

쪼개서 보면 다음과 같습니다.

1. **Q, K, V 투영**: 하나의 선형층이 입력을 Query, Key, Value 세 행렬로 투영합니다. 각각의 크기는 `(B, T, n_embd)`입니다.

2. **멀티헤드 reshape**: 임베딩 차원을 `n_head`개의 별도 헤드로 나눕니다. 각 헤드는 `head_dim = n_embd / n_head = 64`차원입니다. 이렇게 하면 모델이 입력의 여러 측면을 병렬로 주시할 수 있습니다.

3. **Scaled dot-product attention**: 각 query 위치에 대해 모든 key 위치와의 유사도 점수를 계산하고, 미래 위치를 마스킹(causal)한 뒤 softmax를 적용하고, 그 결과로 value에 가중치를 줍니다. 수식은 `softmax(QK^T / sqrt(head_dim)) @ V`입니다.

4. **Causal masking**(`is_causal=True`): 위치 `i`는 위치 `0..i`까지만 볼 수 있습니다. 학습 중 미래 토큰을 보고 "부정행위"하는 것을 막습니다. 이것이 GPT를 **자기회귀** 모델로 만드는 핵심입니다.

5. **출력 투영**: 모든 헤드를 연결한 뒤 다시 `n_embd` 차원으로 투영합니다.

### 왜 멀티헤드인가?

384차원짜리 헤드 하나 대신 64차원 헤드 6개를 쓰면 모델이 서로 다른 관계를 동시에 추적할 수 있습니다. 어떤 헤드는 자음 뒤에 어떤 모음이 오는지 볼 수 있고, 다른 헤드는 줄바꿈 패턴을 볼 수 있으며, 또 다른 헤드는 최근 문맥에 집중할 수 있습니다.

### MLP 블록

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)       # 위로 투영: 384 → 1536
        x = self.gelu(x)       # 비선형성
        return self.c_proj(x)  # 다시 아래로 투영: 1536 → 384
```

```
x (B, T, 384)
    │
    ▼
┌─────────┐
│  c_fc   │  Linear: 384 → 1536 (4배 확장)
└─────────┘
    │
    ▼
┌─────────┐
│  GELU   │  비선형성
└─────────┘
    │
    ▼
┌─────────┐
│  c_proj │  Linear: 1536 → 384 (다시 투영)
└─────────┘
    │
    ▼
출력 (B, T, 384)
```

MLP는 각 위치에 독립적으로 적용됩니다. 표현을 임베딩 차원의 4배로 확장하고, 비선형성(GELU)을 적용한 뒤 다시 줄입니다. 모델이 대부분의 "사고"를 하는 곳이 여기입니다. 어텐션이 정보를 모으면 MLP가 그 정보를 처리합니다.

**왜 ReLU 대신 GELU인가요?** GELU(Gaussian Error Linear Unit)는 ReLU보다 부드럽습니다. 0에서 딱 잘리는 경계가 없어서 그래디언트 흐름에 도움이 됩니다. GPT-2는 속도를 위해 `tanh` 근사를 사용합니다.

### 트랜스포머 블록

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # 잔차 연결이 있는 attention
        x = x + self.mlp(self.ln_2(x))    # 잔차 연결이 있는 MLP
        return x
```

```
x (B, T, 384)
    │
    ├───────────────────┐
    ▼                   │
┌──────────┐            │
│ LayerNorm│            │
└──────────┘            │
    │                   │
    ▼                   │
┌──────────┐            │
│ Self-Attn│            │
└──────────┘            │
    │                   │
    ▼                   │
  + ◄───────────────────┘  잔차 연결
    │
    ├───────────────────┐
    ▼                   │
┌──────────┐            │
│ LayerNorm│            │
└──────────┘            │
    │                   │
    ▼                   │
┌──────────┐            │
│   MLP    │            │
└──────────┘            │
    │                   │
    ▼                   │
  + ◄───────────────────┘  잔차 연결
    │
    ▼
출력 (B, T, 384)
```

두 가지 중요한 설계 선택이 있습니다.

1. **Pre-norm**(attention/MLP 뒤가 아니라 앞에 LayerNorm): 각 하위 층에 들어가는 입력을 정규화해 학습을 안정화합니다. 원래 트랜스포머 논문은 post-norm을 썼지만, 지금은 pre-norm이 표준입니다.

2. **잔차 연결**(`x = x + sublayer(x)`): 입력을 출력에 다시 더합니다. 이렇게 하면 역전파 중 그래디언트가 네트워크를 직접 통과할 수 있어 깊은 네트워크도 학습 가능합니다. 잔차가 없다면 6층 네트워크도 훨씬 학습하기 어렵습니다.

### 파라미터 수

```python
config = GPTConfig()
model = GPT(config)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params / 1e6:.1f}M")  # ~10.8M
```

파라미터는 어디에 있을까요?

- 토큰 임베딩: `65 × 384 = 25K`(문자 단위 어휘라 작음, lm_head와 공유)
- 위치 임베딩: `256 × 384 = 98K`
- 트랜스포머 블록 하나당: `~1.8M`(attention: 4 × 384² = 590K, MLP: 2 × 384 × 1536 = 1.2M, norm은 무시해도 될 정도)
- 6개 블록: `~10.6M`
- 전체: `~10.8M`

거의 모든 파라미터가 임베딩이 아니라 트랜스포머 블록 안에 있다는 점에 주목하세요. GPT-2의 5만 어휘를 쓰면 임베딩 테이블만 50,257 × 384 = 19.3M입니다. 이 모델 전체의 거의 두 배입니다. 그래서 어휘 크기가 중요합니다.

## 핵심 정리

- GPT는 동일한 트랜스포머 블록을 쌓은 구조입니다
- 각 블록: LayerNorm → Self-Attention → Residual → LayerNorm → MLP → Residual
- 셀프 어텐션은 토큰이 모든 이전 토큰을 볼 수 있게 합니다(causal masking은 미래를 보는 것을 막습니다)
- 멀티헤드 어텐션은 여러 어텐션 패턴을 병렬로 실행합니다
- 잔차 연결과 레이어 정규화는 깊은 네트워크를 학습 가능하게 만듭니다
- 입력 임베딩과 출력 투영 사이의 weight tying은 파라미터를 줄입니다

## 다음: [파트 3 — 학습 루프 →](03-training-loop.md)
