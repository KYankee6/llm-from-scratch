# 파트 3: 학습 루프

이제 모델이 있습니다. 다음은 모델에게 언어를 가르칠 차례입니다. 학습 루프는 모델이 실제로 배우는 곳이며, 여기서 내리는 모든 결정은 모델이 수렴할지 아니면 의미 없는 출력으로 발산할지에 영향을 줍니다.

## 학습 목표

GPT는 **다음 토큰 예측**으로 학습합니다. 토큰 `[t0, t1, ..., tn]`이 주어지면 `[t1, t2, ..., tn+1]`을 예측합니다. 손실 함수는 모델이 예측한 확률 분포와 실제 다음 토큰 사이의 cross-entropy입니다.

이것은 자기지도 학습 과제입니다. 라벨이 데이터 자체에서 나옵니다. 모든 텍스트 조각은 한 위치만 밀리면서 동시에 입력이자 타깃이 됩니다.

## 작성하기: `train.py`

scratchpad 안에 `train.py`라는 새 파일을 만듭니다. 이 파일은 파트 2에서 작성한 `model.py`와 파트 4에서 작성할 `generate.py`를 가져옵니다. 지금은 샘플 생성 부분을 건너뛰고, 파트 4를 마친 뒤 돌아와서 추가하면 됩니다.

아래 조각들을 읽는 순서대로 `train.py`에 추가하세요.

### 단계 1: 데이터 로딩(문자 단위)

```python
import torch

def load_data(filepath, block_size, batch_size, device):
    with open(filepath, "r") as f:
        text = f.read()

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    tokens = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    print(f"Dataset: {len(tokens):,} chars, vocab size: {vocab_size}")

    def get_batch(split_tokens):
        ix = torch.randint(len(split_tokens) - block_size - 1, (batch_size,))
        x = torch.stack([split_tokens[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([split_tokens[i + 1:i + block_size + 1] for i in ix]).to(device)
        return x, y

    n = int(0.9 * len(tokens))
    get_train = lambda: get_batch(tokens[:n])
    get_val = lambda: get_batch(tokens[n:])
    return get_train, get_val, vocab_size, stoi, itos
```

각 배치는 다음처럼 만들어집니다.

- `batch_size`개의 무작위 시작 위치를 뽑습니다
- `x`: 위치 `i`부터 `i + block_size`까지의 문자(입력)
- `y`: 위치 `i+1`부터 `i + block_size + 1`까지의 문자(타깃, 한 칸 밀림)

이 함수는 배치 생성기와 함께 `stoi`/`itos` 매핑을 반환합니다. 텍스트 생성 때 이 매핑이 필요합니다.

### 단계 2: 장치 설정

```python
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")     # Apple Silicon GPU
    elif torch.cuda.is_available():
        return torch.device("cuda")    # NVIDIA GPU
    return torch.device("cpu")
```

Apple Silicon이 들어간 MacBook에서는 MPS가 CPU보다 대략 2-3배 빠릅니다.

### 단계 3: 학습률 스케줄

```python
import math

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

두 단계로 구성됩니다.

1. **Warmup**(처음 약 100 step): 학습률을 거의 0에서 `max_lr`까지 올립니다. 옵티마이저가 큰 업데이트를 하기 전에 모멘트 추정값을 보정할 시간을 줍니다.

2. **Cosine decay**(나머지 step): 학습률을 부드럽게 낮춥니다. 초반에는 큰 업데이트로 탐색하고, 후반에는 작은 업데이트로 다듬습니다.

### 단계 4: 옵티마이저

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

Shakespeare 문자 단위 학습에는 `lr=1e-3`와 가벼운 weight decay를 쓰는 기본 `AdamW`가 잘 맞습니다. GPT-2 전체 레시피(분리된 decay 그룹, betas=(0.9, 0.95), weight_decay=0.1)는 대규모 BPE 학습용이라 여기서는 과합니다.

### 단계 5: 전체 학습 루프

```python
import json
from tqdm import tqdm

def train(data_path, max_steps=5000, batch_size=64,
          n_layer=6, n_head=6, n_embd=384, block_size=256):
    device = get_device()
    print(f"Using device: {device}")

    get_train_batch, get_val_batch, vocab_size, stoi, itos = load_data(
        data_path, block_size, batch_size, device
    )

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )
    model = GPT(config).to(device)
    print(f"Model: {n_layer}L/{n_head}H/{n_embd}D, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    max_lr = 1e-3
    min_lr = max_lr * 0.1
    warmup_steps = 100

    loss_log = {"steps": [], "train": [], "val": []}

    pbar = tqdm(range(max_steps), desc="Training")
    for step in pbar:
        # --- 검증 손실 ---
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(20):
                    x, y = get_val_batch()
                    _, loss = model(x, y)
                    val_losses.append(loss.item())
                val_loss = sum(val_losses) / len(val_losses)
                tqdm.write(f"Step {step:5d} | val loss: {val_loss:.4f}")
            model.train()

        # --- 학습률 업데이트 ---
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # --- 학습 step ---
        x, y = get_train_batch()
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        # --- 손실 기록 ---
        loss_log["steps"].append(step)
        loss_log["train"].append(loss.item())
        if step % 100 == 0:
            loss_log["val"].append(val_loss)

        # --- 샘플 생성 ---
        if step > 0 and step % 100 == 0:
            model.eval()
            sample = generate(model, "To be or not", stoi, itos,
                            max_new_tokens=100, temperature=0.8)
            tqdm.write(f"\n--- Step {step} sample ---\n{sample}\n---\n")
            model.train()

        # --- 체크포인트 저장 ---
        if step > 0 and step % 1000 == 0:
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "config": config,
                "stoi": stoi,
                "itos": itos,
            }, f"checkpoint_{step}.pt")

    # --- 최종 체크포인트와 손실 로그 저장 ---
    torch.save({
        "step": max_steps,
        "model_state_dict": model.state_dict(),
        "config": config,
        "stoi": stoi,
        "itos": itos,
    }, "checkpoint_final.pt")

    with open("loss_log.json", "w") as f:
        json.dump(loss_log, f)

    return model, stoi, itos
```

### 단계 6: 진입점

터미널에서 실행할 수 있도록 `train.py` 맨 아래에 이것을 추가합니다.

```python
if __name__ == "__main__":
    train("../data/shakespeare.txt")
```

기본 설정(6L/6H/384D, 5000 step)으로 포함된 Shakespeare 데이터셋을 학습합니다. 다른 인자를 넘겨 모델을 바꿀 수 있습니다.

```python
if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "../data/shakespeare.txt"
    train(data_path)
```

### 각 부분이 하는 일

**검증 손실**: 100 step마다 따로 남겨 둔 데이터에서 평가합니다. train loss는 내려가는데 val loss가 올라가면 과적합입니다.

**그래디언트 클리핑**(`clip_grad_norm_`): 전체 그래디언트 크기를 1.0으로 제한합니다. 가끔 발생하는 큰 그래디언트가 가중치를 망가뜨리는 것을 막습니다.

**샘플 생성**: 100 step마다 텍스트를 생성해서 모델이 배우는 과정을 볼 수 있습니다. 무작위 문자 → 무작위 단어 → Shakespeare풍 텍스트로 바뀌는 모습을 보게 됩니다.

**체크포인트**: 모델 상태를 주기적으로 저장합니다. 체크포인트에는 `stoi`/`itos`가 포함되어 있어서 원본 데이터 없이도 저장된 모델에서 텍스트를 생성할 수 있습니다. 학습이 끝나면 최종 체크포인트가 `checkpoint_final.pt`로 저장됩니다.

**손실 로그**: 학습 손실과 검증 손실은 `loss_log.json`에 저장됩니다. 학습 후 손실 곡선을 그릴 때 사용합니다(파트 5 참고).

## 손실 숫자의 의미(문자 단위, vocab=65)

- **~4.2**: 무작위(학습 전). `ln(65) ≈ 4.17`
- **~3.3**: 문자 빈도를 배움(어떤 글자가 흔한지)
- **~2.5**: 흔한 bigram을 배움("th", "he", "in")
- **~1.5-2.0**: 알아볼 수 있는 단어와 Shakespeare풍 구조를 생성
- **~1.0-1.2**: 좋은 품질, 등장인물 이름과 줄바꿈이 있는 운문 생성
- **<1.0**: 학습 데이터를 외우고 있을 가능성이 큼

## 모델이 배우는 모습 보기

실제 학습 실행은 다음처럼 보입니다(Shakespeare에서 6L/6H/384D, batch_size=64, M3 Pro). 프롬프트는 항상 "To be or not"입니다.

**200단계**(val loss: ~3.5) — 단어가 없는 무작위 문자:
```
To be or notis p ce mei odorethleedetire'ilethed ye m arkesothir fnon b tigb'i.
```

**800단계**(val loss: ~1.8) — 단어가 만들어지고 등장인물 이름이 나타남:
```
To be or not men, and my lord.

ROMEO:
Thou sir, do content the he, stray, there ir;
```

**1000단계**(val loss: 1.64) — 어느 정도 일관된 구절과 Shakespeare식 구조:
```
To be or nothing are good men,
The profent of little, our actory.

CORIOLANUS:
Is it now of your many death?
```

**2400단계**(val loss: ~1.60) — 품질이 가장 좋음. 그럴듯한 Shakespeare:
```
To be or not to be some of you shall know
That everlature by Romeo: what news,
Which you had knock'd my part to speak
```

**3500단계**(val loss: 2.34) — 과적합. 여전히 유창하지만 창의성이 줄어듦:
```
To be or nothing, take me but most profane,
That offer them not amish. If I defeath
Is not a puggival and self,
```

## 과적합: train loss vs val loss

1000만 파라미터와 Shakespeare 약 100만 문자만 있으면 모델은 **과적합**됩니다. 일반 패턴을 배우기보다 학습 데이터를 외웁니다. 다음처럼 분명히 보입니다.

```
Step   500 | val loss: 2.14   ← 빠르게 떨어짐, 구조를 배우는 중
Step  1000 | val loss: 1.64   ← 계속 개선됨
Step  1500 | val loss: 1.57   ← 가장 좋은 구간
Step  2000 | val loss: 1.59   ← 정체되기 시작
Step  2500 | val loss: 1.71   ← val loss 상승, 과적합
Step  3000 | val loss: 1.98   ← 더 나빠짐
Step  3500 | val loss: 2.34   ← 완전히 암기 중(train loss는 0.54)
```

**최고의 모델**은 step 5000이 아니라 step 1500-2000 근처입니다(val loss ~1.57). 그 이후의 모든 step은 새로운 텍스트 생성 능력을 *나쁘게* 만듭니다. 학습 데이터를 더 잘 읊게 될 뿐입니다.

### 과적합의 원인

모델은 **1000만 파라미터**로 **약 100만 문자**를 학습합니다. 파라미터 대 데이터 비율이 10:1입니다. 모델은 학습 세트의 모든 문자를 외울 만큼 충분한 용량을 갖고 있습니다. 해결책은 항상 같습니다. **더 많은 데이터** 또는 **더 작은 모델**입니다.

### 어떻게 대응할까

이 워크숍에서는 과적합이 예상되는 현상이며 괜찮습니다. 중요한 개념을 보여주기 때문입니다. 실제로는 다음을 선택합니다.

1. **더 많은 데이터 사용** — TinyStories(4억 7600만 토큰)는 1000만 모델이 훨씬 오래 학습하게 해줍니다
2. **더 작은 모델 사용** — 2L/2H/128D 모델(약 0.5M 파라미터)은 Shakespeare에서 훨씬 느리게 과적합됩니다
3. **Dropout 추가** — 학습 중 활성값을 무작위로 0으로 만들어 정규화 역할을 하게 합니다
4. **일찍 멈추기** — 가장 낮은 val loss를 가진 체크포인트를 저장하고 사용합니다

## MacBook에서의 일반적인 학습(문자 단위 Shakespeare)

| 모델 | 파라미터 | Batch Size | Steps | 시간(M3 Pro) | 최고 Val Loss | 과적합 시점 |
|------|----------|------------|-------|--------------|---------------|-------------|
| 6L/6H/384D | ~10M | 64 | 5,000 | ~45분 | ~1.7(step 2500) | ~step 1500 |
| 4L/4H/256D | ~4M | 64 | 5,000 | ~20분 | ~1.6(step 3000) | ~step 2000 |
| 2L/2H/128D | ~0.5M | 64 | 5,000 | ~5분 | ~1.8(step 5000) | 거의 없음 |

6L 모델은 가장 빨리 좋은 샘플을 만들지만 가장 빨리 과적합됩니다. 2L 모델은 빠르게 학습되고 과적합이 거의 없지만 출력 품질은 낮습니다. 이것이 근본적인 트레이드오프입니다. **모델 용량 vs 데이터 크기**.

6L/6H/384D 설정으로 시작하세요. M3 Pro에서 batch_size=64를 쓰면 약 1.9 it/s가 나옵니다.

## 핵심 정리

- 목표는 cross-entropy 손실을 사용한 다음 문자 예측입니다
- 문자 단위 토큰화는 작은 데이터셋에 가장 잘 맞습니다. BPE 어휘는 너무 희소합니다
- lr=1e-3과 cosine decay를 쓰는 AdamW는 좋은 출발점입니다
- 1.0의 그래디언트 클리핑은 학습 불안정을 막습니다
- 학습 중 샘플을 생성하세요. 진행 상황을 보는 가장 좋은 방법입니다
- **train loss와 val loss의 차이를 보세요**. val loss가 오르기 시작하면 과적합입니다
- 최고의 모델은 train loss가 가장 낮은 모델이 아니라 val loss가 가장 낮은 모델입니다

## 다음: [파트 4 — 텍스트 생성 →](04-text-generation.md)
