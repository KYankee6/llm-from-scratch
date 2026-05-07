# LLM을 처음부터 직접 학습하기

GPT 학습 파이프라인의 모든 부분을 직접 작성하면서 각 구성 요소가 무엇을 하고 왜 필요한지 이해하는 실습형 워크숍입니다.

Andrej Karpathy의 [nanoGPT](https://github.com/karpathy/nanoGPT)는 제가 LLM과 트랜스포머를 처음 제대로 접한 계기였습니다. 동작하는 언어 모델을 PyTorch 몇백 줄로 만들 수 있다는 사실은 AI를 바라보는 방식을 완전히 바꾸었고, 이 분야를 더 깊이 파고들게 했습니다.

이 워크숍은 다른 사람도 같은 경험을 할 수 있게 하려는 시도입니다. nanoGPT는 GPT-2(1억 2400만 파라미터) 재현을 목표로 하며 폭넓은 내용을 다룹니다. 이 프로젝트는 핵심만 남겨 약 1000만 파라미터 모델로 줄였고, 노트북에서 1시간 안에 학습되도록 구성했습니다. 한 번의 워크숍 세션 안에서 끝낼 수 있게 설계했습니다.

## 만들게 될 것

MacBook에서 처음부터 학습한 GPT 모델입니다. Shakespeare풍 텍스트를 생성할 수 있습니다. 직접 작성할 내용은 다음과 같습니다.

- **토크나이저** — 텍스트를 모델이 처리할 수 있는 숫자로 바꾸기
- **모델 아키텍처** — 트랜스포머: 임베딩, 어텐션, 피드포워드 층
- **학습 루프** — 순전파, 손실, 역전파, 옵티마이저, 학습률 스케줄링
- **텍스트 생성** — 학습한 모델에서 샘플링하기

## 사전 준비

- 아무 노트북이나 데스크톱(Mac, Linux, Windows)
- Python 3.12 이상
- Python 코드를 읽을 수 있는 정도의 익숙함(머신러닝 경험은 필요하지 않습니다)

학습은 Apple Silicon GPU(MPS), NVIDIA GPU(CUDA), CPU 중 가능한 장치를 자동으로 사용합니다. [Google Colab](https://colab.research.google.com/)에서도 동작합니다. 파일을 업로드한 뒤 `!python train.py`로 실행하면 됩니다.

## 시작하기

### 로컬(권장)

[uv](https://docs.astral.sh/uv/)가 없다면 먼저 설치합니다.

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

그다음 프로젝트를 준비합니다.

```bash
uv sync
mkdir scratchpad && cd scratchpad
```

### Google Colab

로컬 환경이 없다면 저장소를 Colab에 업로드하고 의존성을 설치합니다.

```python
!pip install torch numpy tqdm tiktoken
```

`data/shakespeare.txt`를 Colab 파일에 업로드한 뒤, 노트북 셀에 코드를 작성하거나 `.py` 파일을 업로드해서 `!python train.py`로 실행합니다.

---

문서는 순서대로 따라가세요. 각 파트는 파이프라인의 한 조각을 작성하게 하며, 각 구성 요소가 무엇을 하고 왜 필요한지 설명합니다. 끝까지 마치면 직접 작성한 `model.py`, `train.py`, `generate.py`가 생깁니다.

| 파트 | 작성할 것 | 개념 |
|------|-----------|------|
| [파트 1: 토큰화](docs/01-tokenization.md) | 문자 단위 토크나이저 | 문자 인코딩, 어휘 크기, 작은 데이터에서 BPE가 실패하는 이유 |
| [파트 2: 트랜스포머](docs/02-the-transformer.md) | 전체 GPT 모델 아키텍처 | 임베딩, 셀프 어텐션, 레이어 정규화, MLP 블록 |
| [파트 3: 학습 루프](docs/03-training-loop.md) | 완전한 학습 파이프라인 | 손실 함수, AdamW, 그래디언트 클리핑, 학습률 스케줄링 |
| [파트 4: 텍스트 생성](docs/04-text-generation.md) | 추론과 샘플링 | temperature, top-k, 자기회귀 디코딩 |
| [파트 5: 모두 연결하기](docs/05-putting-it-together.md) | 실제 데이터로 학습하고 실험하기 | 손실 곡선, 스케일링 실험, 다음 단계 |
| [파트 6: 대회](docs/06-competition.md) | 최고의 AI 시인 학습하기 | 데이터셋 찾기, 규모 키우기, 최고의 시 제출하기 |

## 아키텍처: GPT 한눈에 보기

```
입력 텍스트
    │
    ▼
┌─────────────────┐
│   토크나이저    │  "hello" → [20, 43, 50, 50, 53]  (문자 단위)
└────────┬────────┘
         ▼
┌─────────────────┐
│  토큰 임베딩 +  │  토큰 ID → 벡터(n_embd 차원)
│  위치 임베딩    │  + 위치 정보
└────────┬────────┘
         ▼
┌─────────────────┐
│  트랜스포머     │  × n_layer
│  블록:          │
│  ┌────────────┐ │
│  │ LayerNorm  │ │
│  │ Self-Attn  │ │  n_head개의 병렬 어텐션 헤드
│  │ + Residual │ │
│  ├────────────┤ │
│  │ LayerNorm  │ │
│  │ MLP (FFN)  │ │  4배 확장, GELU, 다시 투영
│  │ + Residual │ │
│  └────────────┘ │
└────────┬────────┘
         ▼
┌─────────────────┐
│   LayerNorm     │
│   Linear → logits│  vocab_size개 출력(다음 토큰 확률)
└─────────────────┘
```

## 이 워크숍의 모델 설정

| 설정 | 파라미터 | n_layer | n_head | n_embd | 학습 시간(M3 Pro) |
|------|----------|---------|--------|--------|-------------------|
| Tiny | ~0.5M | 2 | 2 | 128 | ~5분 |
| Small | ~4M | 4 | 4 | 256 | ~20분 |
| **Medium(기본값)** | **~10M** | **6** | **6** | **384** | **~45분** |

모든 설정은 문자 단위 토큰화(vocab_size=65)와 block_size=256을 사용합니다.

## 토큰화: 문자 vs BPE

이 워크숍은 Shakespeare에 **문자 단위** 토큰화를 사용합니다. BPE 토큰화(GPT-2의 5만 어휘)는 작은 데이터셋에서 잘 동작하지 않습니다. 대부분의 토큰 bigram이 너무 드물어서 모델이 패턴을 배우기 어렵기 때문입니다.

| 토크나이저 | 어휘 크기 | 필요한 데이터셋 크기 |
|------------|-----------|----------------------|
| **문자 단위** | ~65 | 작음(Shakespeare, ~1MB) |
| **BPE(tiktoken)** | 50,257 | 큼(TinyStories 이상, 100MB+) |

파트 5에서는 더 큰 데이터셋을 위해 BPE로 전환하는 방법을 다룹니다.

## 주요 참고 자료

- [nanoGPT](https://github.com/karpathy/nanoGPT) — 이 워크숍의 기반이 된 프로젝트. PyTorch 약 300줄로 구현한 최소 GPT 학습 코드
- [build-nanogpt 영상 강의](https://github.com/karpathy/build-nanogpt) — 빈 파일에서 GPT-2를 만드는 4시간짜리 영상
- [Karpathy의 microgpt](http://karpathy.github.io/2026/02/12/microgpt/) — 의존성 없이 순수 Python 200줄로 구현한 전체 GPT
- [nanochat](https://github.com/karpathy/nanochat) — 전체 ChatGPT 클론 학습 파이프라인
- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) — 원래의 트랜스포머 논문
- [GPT-2 논문(2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — 비지도 학습자로서의 언어 모델
- [TinyStories 논문](https://arxiv.org/abs/2305.07759) — 선별된 데이터로 학습한 작은 모델이 기대 이상으로 잘하는 이유
