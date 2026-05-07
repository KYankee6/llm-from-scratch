# 파트 4: 텍스트 생성

모델을 학습했습니다. 이제 글을 쓰게 해봅시다. GPT의 텍스트 생성은 **자기회귀** 방식입니다. 토큰을 하나씩 생성하고, 그것을 입력 뒤에 붙이고, 다시 반복합니다.

scratchpad 안에 `generate.py`라는 새 파일을 만듭니다. 작성이 끝나면 `train.py`로 돌아가 맨 위에 `from generate import generate`를 추가하세요. 그러면 파트 3에서 건너뛴 학습 중 샘플 생성이 활성화됩니다.

## 순진한 접근: 탐욕 디코딩(Greedy Decoding)

항상 가장 확률이 높은 다음 토큰을 고릅니다.

```python
def generate_greedy(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)
    return idx
```

이 방식은 결정적입니다. 같은 프롬프트는 항상 같은 출력을 만듭니다. 가장 높은 확률의 이어짐이 스스로를 강화하기 때문에 반복적이고 지루한 결과가 나오기 쉽습니다.

## Temperature(온도)

softmax를 적용하기 전에 logits를 스케일링합니다. temperature가 높을수록 더 무작위적이고, 낮을수록 더 결정적입니다.

```python
logits = logits / temperature
```

수식으로 보면 softmax는 `exp(logit_i) / sum(exp(logit_j))`를 계산합니다. 모든 logits를 temperature로 나누면 분포가 바뀝니다.

- **T = 1.0**: 일반적인 확률
- **T → 0**: greedy(argmax)에 가까워짐
- **T > 1.0**: 분포를 평평하게 만들어 드문 토큰에도 더 많은 기회를 줌
- **T = 0.7-0.9**: 일관성 있으면서도 다양한 텍스트를 만들기 좋은 보통의 sweet spot

## Top-k 샘플링

가장 확률이 높은 k개의 토큰만 고려합니다. 나머지는 모두 `-inf`로 설정합니다.

```python
if top_k > 0:
    values, _ = torch.topk(logits, top_k)
    logits[logits < values[:, -1:]] = float("-inf")
```

이렇게 하면 모델이 극도로 가능성이 낮은 토큰을 샘플링하지 못하게 됩니다. 문자 단위 모델(vocab=65)에서는 `top_k=40`이 적당합니다. 대부분의 문자는 여전히 고려하지만 아주 가능성이 낮은 문자는 제외합니다.

## 전체 generate 함수

```python
@torch.no_grad()
def generate(model, prompt, stoi, itos, max_new_tokens=200, temperature=0.8, top_k=40):
    device = next(model.parameters()).device
    tokens = [stoi[c] for c in prompt if c in stoi]
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, -1:]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)

    return "".join([itos[i] for i in idx[0].tolist()])
```

각 토큰을 생성하는 파이프라인은 다음과 같습니다.

1. 현재 시퀀스를 모델에 넣어 다음 위치의 logits를 얻습니다
2. temperature scaling을 적용합니다
3. top-k로 필터링합니다(가능성이 아주 낮은 토큰 제거)
4. softmax로 확률로 바꿉니다
5. `multinomial`로 분포에서 샘플링합니다
6. 샘플링한 토큰을 붙이고 반복합니다

`@torch.no_grad()`는 그래디언트 계산을 끕니다. 추론에는 필요 없고, 끄면 메모리를 절약할 수 있습니다.

이 함수는 학습 데이터에서 만든 `stoi`/`itos` 매핑을 받습니다. 이 매핑은 문자가 토큰 ID와 어떻게 오가는지 정의합니다.

### 명령줄 인터페이스

터미널에서 실행할 수 있도록 `generate.py` 맨 아래에 이것을 추가합니다.

```python
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="학습된 GPT 체크포인트에서 텍스트를 생성합니다")
    parser.add_argument("checkpoint", help="체크포인트 파일 경로(예: checkpoint_final.pt)")
    parser.add_argument("--prompt", default="To be or not", help="생성을 시작할 텍스트")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="생성할 토큰 수")
    parser.add_argument("--temperature", type=float, default=0.8, help="샘플링 temperature(낮을수록 더 결정적)")
    parser.add_argument("--top_k", type=int, default=40, help="가장 가능성 높은 top-k 토큰에서만 샘플링")
    parser.add_argument("--seed", type=int, default=None, help="재현성을 위한 난수 seed")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    config = checkpoint["config"]
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    output = generate(model, args.prompt, stoi, itos,
                      max_new_tokens=args.max_new_tokens,
                      temperature=args.temperature,
                      top_k=args.top_k)
    print(output)
```

## Seed(시드)를 사용한 재현성

생성에는 무작위 샘플링(`torch.multinomial`)이 들어가므로 같은 프롬프트도 실행할 때마다 다른 출력을 만듭니다. 재현 가능한 결과를 얻으려면 생성 전에 seed를 설정합니다.

```python
torch.manual_seed(42)
print(generate(model, "To be or not", stoi, itos, temperature=0.8))
# seed=42에서는 매번 같은 출력
```

명령줄에서는 다음처럼 실행합니다.

```bash
python generate.py checkpoint_final.pt --prompt "To be or not" --seed 42
```

## 다양한 설정 실험하기

```python
checkpoint = torch.load("checkpoint_final.pt", weights_only=False)
config = checkpoint["config"]
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]

model = GPT(config)
model.load_state_dict(checkpoint["model_state_dict"])

# 결정적이고 반복적임
print(generate(model, "To be or not to be", stoi, itos, temperature=0.1))

# 균형 잡힘
print(generate(model, "To be or not to be", stoi, itos, temperature=0.8))

# 창의적이지만 일관성이 떨어질 수 있음
print(generate(model, "To be or not to be", stoi, itos, temperature=1.5))
```

## 기대할 수 있는 결과

Shakespeare에서 학습한 실제 샘플은 다음과 같습니다(6L/6H/384D).

### 200단계(val loss ~3.5) — 무작위 문자
```
To be or notis p ce mei odorethleedetire'ilethed ye m arkesothir fnon b tigb'i.
```

### 1000단계(val loss 1.64) — 단어와 구조가 나타나기 시작
```
To be or nothing are good men,
The profent of little, our actory.

CORIOLANUS:
Is it now of your many death?
```

### 2400단계(val loss ~1.60) — 가장 좋은 품질, 그럴듯한 Shakespeare
```
To be or not to be some of you shall know
That everlature by Romeo: what news,
Which you had knock'd my part to speak
```

참고: 가장 좋은 출력은 대략 step 1500-2500 사이에 나옵니다. 그 이후에는 모델이 과적합되어 암기한 학습 데이터를 다시 내뱉기 시작합니다(자세한 내용은 파트 3 참고).

## 핵심 정리

- 자기회귀 생성: 토큰 하나를 예측하고, 붙이고, 반복합니다
- Greedy decoding은 결정적이고 반복적입니다
- Temperature는 무작위성을 조절합니다(보통 0.7-0.9가 좋습니다)
- Top-k는 극도로 가능성이 낮은 토큰을 제거합니다
- 문자 단위 모델에서는 학습 중 샘플을 생성해 모델이 배우는 과정을 확인하세요

## 다음: [파트 5 — 모두 연결하기 →](05-putting-it-together.md)
