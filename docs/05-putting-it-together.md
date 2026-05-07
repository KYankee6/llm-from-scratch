# 파트 5: 모두 연결하기

이제 모든 조각을 연결하고, 실제 데이터로 학습하고, 대회를 준비할 시간입니다.

## 프로젝트 구조

지금쯤이면 다음 파일이 있어야 합니다.

```
scratchpad/
├── model.py           # GPT 아키텍처(파트 2)
├── train.py           # 토큰화 + 데이터 로딩 + 학습 루프(파트 1 & 3)
└── generate.py        # 텍스트 생성(파트 4)
```

Shakespeare 데이터셋은 저장소의 `data/shakespeare.txt`에 포함되어 있습니다. 따로 다운로드할 필요가 없습니다.

### Google Colab

로컬 환경 대신 Colab을 사용한다면 다음 순서로 진행합니다.

1. [colab.research.google.com](https://colab.research.google.com/)에서 새 노트북을 엽니다
2. **Runtime → Change runtime type → GPU (T4)** 로 이동합니다
3. 첫 번째 셀에서 의존성을 설치합니다.
   ```python
   !pip install -q torch numpy tqdm tiktoken
   ```
4. Shakespeare를 다운로드합니다.
   ```python
   import urllib.request, os
   os.makedirs("data", exist_ok=True)
   urllib.request.urlretrieve(
       "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
       "data/shakespeare.txt"
   )
   ```
5. 노트북 셀에 모델, 학습 루프, 생성 코드를 작성합니다(문서에서 붙여넣거나 직접 작성)
6. 데이터 경로는 `"../data/shakespeare.txt"`가 아니라 `"data/shakespeare.txt"`를 사용합니다

바로 실행할 수 있는 노트북도 저장소의 `colab.ipynb`에 포함되어 있습니다.

## 단계 1: 학습

```bash
cd scratchpad
python train.py
```

기본 설정은 Shakespeare에서 6L/6H/384D 모델(약 1000만 파라미터)을 batch_size=64로 5000 step 학습합니다. M3 Pro에서는 약 45분이 걸립니다. 실행 중 다음을 보게 됩니다.

- 100 step마다 val loss와 생성 샘플
- 1000 step마다 체크포인트
- 마지막에 최종 체크포인트와 손실 로그

## 단계 2: 생성

```bash
python generate.py checkpoint_final.pt
```

체크포인트를 불러와 세 가지 프롬프트에서 텍스트를 생성합니다. 어떤 체크포인트 파일이든 인자로 넘길 수 있습니다.

## 단계 3: 실험

기본 파이프라인이 동작하면 대회 전에 직관을 쌓기 위해 다음을 실험해 보세요.

### 모델 크기 vs 품질

같은 데이터로 세 모델을 학습하고 출력 품질을 비교합니다.

| 설정 | 파라미터 | n_layer | n_head | n_embd | 예상 손실 |
|------|----------|---------|--------|--------|-----------|
| Tiny | ~0.5M | 2 | 2 | 128 | ~2.0 |
| Small | ~4M | 4 | 4 | 256 | ~1.5 |
| Medium | ~10M | 6 | 6 | 384 | ~1.2 |

`train.py`의 `train()` 호출을 수정합니다.

```python
# tiny — 빠름, 아이디어 테스트에 좋음
model, stoi, itos = train(data_path, n_layer=2, n_head=2, n_embd=128)

# medium — 기본값, 좋은 기준선
model, stoi, itos = train(data_path, n_layer=6, n_head=6, n_embd=384)

# large — 정당화하려면 더 많은 데이터가 필요함
model, stoi, itos = train(data_path, n_layer=12, n_head=12, n_embd=768)
```

### 컨텍스트 길이

`block_size=128`과 `block_size=512`로 학습해 보세요. 긴 컨텍스트는 모델이 전체 연과 압운 구조를 잡게 해주지만 메모리를 더 많이 씁니다. 필요하면 batch_size를 줄이세요.

### 학습률

`3e-4`(보수적), `1e-3`(기본값), `3e-3`(공격적)을 시도해 보세요. 알맞은 학습률은 모델 크기와 데이터에 따라 달라집니다.

## 학습 모니터링

학습 루프는 `loss_log.json`을 저장합니다. 어떤 도구로든 그릴 수 있습니다.

```python
# pip install matplotlib
import json, matplotlib.pyplot as plt

with open("loss_log.json") as f:
    log = json.load(f)

plt.figure(figsize=(10, 6))
plt.plot(log["steps"], log["train"], alpha=0.3, label="train")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_curve.png")
plt.show()
```

### 무엇을 봐야 하나

- **학습 손실이 내려가지 않음**: 학습률이 너무 낮거나 버그가 있음
- **학습 손실은 내려가는데 검증 손실이 올라감**: 과적합, 더 많은 데이터나 더 작은 모델 필요
- **Loss spike**: 학습률을 낮추거나 그래디언트 클리핑을 확인
- **Loss plateau**: 모델이 배울 수 있는 만큼 배움. 더 많은 데이터나 더 큰 모델 필요

## 이제: 대회

모델을 학습했고, 과적합을 봤고, 설정 실험도 해봤습니다. 이제 배운 것을 적용할 차례입니다.

### [파트 6 — 대회: 최고의 AI 시인 →](06-competition.md)

목표는 좋은 시 데이터셋을 찾고, 가능한 최고의 모델을 학습하고, 모델이 생성한 최고의 시를 제출하는 것입니다. 데이터, 모델 크기, 토크나이저, 학습 전략 등 무엇이든 바꿀 수 있습니다. 규칙은 두 가지뿐입니다. 노트북에서 처음부터 학습해야 하며, 시는 모델에서 나온 것이어야 합니다.

## 더 읽을거리

- [Karpathy의 microgpt](http://karpathy.github.io/2026/02/12/microgpt/) — 순수 Python 200줄로 구현한 전체 GPT
- [build-nanogpt 영상 강의](https://github.com/karpathy/build-nanogpt) — 빈 파일에서 GPT-2를 만드는 4시간짜리 영상
- [nanochat](https://github.com/karpathy/nanochat) — 전체 ChatGPT 클론 학습 파이프라인
- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) — 원래의 트랜스포머 논문
- [GPT-2 논문(2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — 비지도 학습자로서의 언어 모델
- [TinyStories 논문](https://arxiv.org/abs/2305.07759) — 선별된 데이터로 학습한 작은 모델
- [Chinchilla (2022)](https://arxiv.org/abs/2203.15556) — 데이터와 파라미터의 최적 스케일링
