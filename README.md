# A6-1 : AI의 속 엔진(두뇌)을 내 손으로 직접 만들어보기

> 분야: 딥러닝 · 학습시간: 80시간 · 필수
>
> 과제 명세 원문: [docs/A6-1.pdf](docs/A6-1.pdf)

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [학습 목표](#학습-목표)
- [최종 결과물](#최종-결과물)
- [프로젝트 구조](#프로젝트-구조)
- [개발 환경과 제약 사항](#개발-환경과-제약-사항)
- [실행 방법](#실행-방법)
- [할 일](#할-일)
- [보너스 과제](#보너스-과제)
- [결과 예시 (명세 참고)](#결과-예시-명세-참고)
- [결과 요약](#결과-요약)
- [명세에서 확인할 점](#명세에서-확인할-점)
- [참고 자료](#참고-자료)

## 프로젝트 개요

이 미션은 PyTorch 같은 프레임워크 없이 **NumPy만으로 작은 딥러닝 프레임워크**를 직접 만든다. 핵심은 **자동 미분(AutoGrad)** 엔진, 즉 계산을 해 나가면서 그 과정을 기억해 두었다가 역전파(Backpropagation, 결과에서 거꾸로 거슬러 가며 각 값의 기울기를 구하는 방법)로 기울기를 자동으로 계산하는 장치다. 딥러닝 학습이 실패하는 원인은 대부분 눈에 보이지 않는 수식 오류나 초기 설정의 작은 차이에서 오기 때문에, 만든 엔진이 맞는지 **기울기 검증(Gradient Checking)** 으로 증명하고, **가중치 초기화(Initialization)** 방식이 학습에 주는 영향을 실험으로 확인한다. 이 과정을 통해 프레임워크 내부 동작을 이해하고 학습 실패의 원인을 진단하는 힘을 기른다.

**핵심 과제** (명세의 학습 흐름 「구현 → 검증 → 실험」)

1. 구현: Tensor와 AutoGrad 엔진, 레이어, 옵티마이저를 만든다.
2. 검증: Gradient Checking으로 역전파 로직이 맞는지(무결성) 진단한다.
3. 실험: 초기화 실험으로 학습 실패 원인을 재현하고 분석한다.

## 학습 목표

과제를 마치면 아래를 스스로 설명할 수 있어야 한다. (명세 "과제 목표")

- 계산 그래프(Computational Graph, 계산 순서를 노드와 화살표로 나타낸 그림)의 동적 생성과 순전파/역전파 원리
- 수치 미분(Numerical Differentiation)으로 역전파 로직의 무결성을 검증하는 원리
- 가중치 초기화가 기울기 소실(Vanishing Gradient)과 폭주(Exploding Gradient)에 미치는 영향
- 연쇄 법칙(Chain Rule)이 코드 레벨에서 어떻게 구현되는지
- 텐서의 브로드캐스팅(Broadcasting, 모양이 다른 배열끼리 자동으로 늘려 계산하는 규칙)이 기울기 계산에 미치는 영향

## 최종 결과물

GitHub 레포지토리 URL로 제출한다.

| 결과물 | 경로 | 포함할 내용 | 상태 |
|---|---|---|---|
| 미니 딥러닝 프레임워크 소스 코드 | `core/`, `nn/`, `optim/` | tensor, autograd, layer, optimizer 모듈 | ⬜ |
| 검증 리포트 (Verification Report) | `gradient_check.py`, 실행 로그 | 수치 미분과 역전파 결과의 오차를 검증한 스크립트 실행 로그 | ⬜ |
| 실험 리포트 (Experiment Report) | `figures/`, `README.md` | 초기화 방법(Zero vs Random vs He)에 따른 Loss 감소 비교 그래프 및 분석 | ⬜ |
| 학습 검증 스크립트 | `train_xor.py`, `train_mnist.py` | XOR 문제 해결 및 MNIST 분류 수행 코드 | ⬜ |
| README.md | `README.md` | 프레임워크 구조 설명, 설치 및 실행 방법, 모듈별 사용 예시 (아래 필수 섹션 포함) | ⬜ |

**README.md 필수 섹션** (명세 템플릿)

- 프로젝트 개요: 프레임워크 목적과 구조 설명
- 설치 방법: 의존성 설치 명령어 (`pip install -r requirements.txt`)
- 실행 방법: 각 스크립트 실행 커맨드 및 순서
- 모듈 구조: core/nn/optim 간 의존 관계 설명
- 사용 예시: Tensor 생성 → 연산 → backward 호출 예시 코드
- 실험 결과 요약: Gradient Check 통과 여부, 초기화 실험 결론

**미션 완료 자가 체크리스트** (명세)

- [ ] `gradient_check.py` 실행 시 모든 레이어의 오차가 1e-7 이하인가?
- [ ] Zero Init으로 학습 실패를 재현할 수 있는가?
- [ ] He Init으로 XOR 문제가 정상 동작하는가? (Loss < 0.1 또는 정확도 95% 이상)
- [ ] 초기화 비교 그래프가 `figures/` 폴더에 있는가?
- [ ] README.md에 설치 및 실행 방법이 명시되어 있는가?
- [ ] MNIST 분류 시 1 Epoch에서 정확도 80% 이상인가?

## 프로젝트 구조

명세의 결과물 기준으로 만들 구조다. 아직 만들지 않은 파일은 (예정)으로 표시한다. 명세가 정한 이름은 `core/nn/optim`, `gradient_check.py`, `figures/`, `requirements.txt`이고, 나머지 파일 이름은 구현하면서 정한다.

```
codyssey_A6-1/
├── core/
│   ├── tensor.py               # (예정) Tensor 클래스 (data, grad, _backward)
│   └── autograd.py             # (예정) backward, 연쇄 법칙 전파
├── nn/
│   ├── layers.py               # (예정) Linear, ReLU, Sigmoid, Softmax
│   └── init.py                 # (예정) Zero, Random, He, Xavier 초기화
├── optim/
│   └── optimizers.py           # (예정) SGD, Adam
├── gradient_check.py           # (예정) 수치 기울기 vs 역전파 기울기 검증
├── train_xor.py                # (예정) XOR 학습 검증
├── train_mnist.py              # (예정) MNIST 분류 검증 (데이터 로더 직접 구현)
├── experiment_init.py          # (예정) 초기화 비교 실험
├── figures/                    # (예정) 초기화 비교 Loss 그래프
├── requirements.txt            # (예정) 의존성 목록
├── docs/
│   └── A6-1.pdf                # 과제 명세
└── README.md
```

## 개발 환경과 제약 사항

**개발 환경**

- Python 3.10 이상 권장
- 구현: NumPy

**금지 사항**

- PyTorch, TensorFlow 등 딥러닝 전용 라이브러리는 로직 참고용으로만 확인하고, 실제 구현에는 NumPy만 사용한다.
- 데이터 로더는 `torch.utils.data` 등 자동 로더를 쓰지 않고 직접 구현한다.

**데이터셋 조건**

- 외부 데이터셋(MNIST 등)을 쓸 때는 바이너리(UBYTE) 또는 CSV를 직접 파싱해 NumPy로 로드한다. 다운로드에는 requests/urllib/gzip 사용을 허용한다.
- 명세의 데이터셋 가이드는 학습 로직을 빠르게 확인하는 용도로 Scikit-learn의 `load_digits()`(MNIST 소형판), `make_moons()`(비선형 분류용)를 권장하며, 이때도 데이터를 NumPy 배열로 변환해 사용하는 과정을 반드시 거치도록 한다.

## 실행 방법

> 아직 구현 전이다. 환경 준비와 실행 명령은 구현하면서 채운다.

명세는 설치 명령으로 `pip install -r requirements.txt`를 예로 든다.

```bash
pip install -r requirements.txt
```

실행 순서는 명세의 흐름(구현 → 검증 → 실험)에 맞춰 `gradient_check.py` → 초기화 실험 → XOR/MNIST 학습 검증 순으로 채운다.

## 할 일

명세의 기능 요구 사항이다. 끝낸 항목은 `[x]`로 바꾼다.

### 1. 텐서 및 자동 미분 엔진

Tensor는 숫자 배열(data)과 그 기울기(grad), 그리고 "나를 만든 연산의 기울기를 어떻게 계산하는가"(_backward)를 함께 들고 다니는 상자다. backward를 부르면 연쇄 법칙(바깥 함수의 미분 × 안쪽 함수의 미분)으로 기울기가 거꾸로 전달된다.

- [ ] Tensor 클래스가 데이터(`data`), 기울기(`grad`), 생성 연산 정보(`_backward`)를 속성으로 가진다.
- [ ] `backward` 메서드가 연결된 노드들에 연쇄 법칙(Chain Rule)으로 기울기를 전파한다.
- [ ] 기본 사칙연산(+, -, *, /)의 순전파와 역전파 로직을 구현한다.
- [ ] 행렬 곱(MatMul)의 순전파와 역전파 로직을 구현한다.

### 2. 무결성 검증 시스템 (Gradient Checking)

수식으로 구한 기울기(Analytic Gradient)가 맞는지, x를 아주 조금 움직여 함숫값이 얼마나 변하는지로 구한 기울기(Numerical Gradient)와 비교해 확인한다.

- [ ] 해석적 기울기(Analytic Gradient)와 수치 기울기(Numerical Gradient)의 차이를 계산하는 함수를 구현한다.
- [ ] 수치 미분 공식은 grad ≈ [f(x+ε) - f(x-ε)] / (2ε), ε = 1e-5를 사용한다.
- [ ] 두 기울기의 상대 오차(Relative Difference)가 1e-7 이하임을 증명하는 테스트 코드를 포함한다(모든 레이어 대상).
- [ ] 검증 스크립트(`gradient_check.py`) 실행 로그를 검증 리포트로 남긴다.

### 3. 레이어 및 초기화 전략

레이어는 신경망을 이루는 블록이고, 초기화는 학습을 시작하기 전 가중치에 넣어 둘 처음 값을 정하는 방법이다. 처음 값을 어떻게 정하느냐에 따라 학습이 아예 안 되거나 빨리 될 수 있다.

- [ ] Linear 레이어를 모듈화하여 구현한다.
- [ ] ReLU 레이어를 모듈화하여 구현한다.
- [ ] Sigmoid 레이어를 모듈화하여 구현한다.
- [ ] Softmax 레이어를 모듈화하여 구현한다.
- [ ] Zero Initialization(모든 가중치를 0으로 초기화)을 옵션으로 제공한다.
- [ ] Random Initialization(표준정규분포에서 샘플링)을 옵션으로 제공한다.
- [ ] He Initialization(N(0, √(2/n_in))에서 샘플링, ReLU용)을 옵션으로 제공한다.
- [ ] Xavier Initialization(N(0, √(2/(n_in+n_out)))에서 샘플링, Sigmoid/Tanh용)을 옵션으로 제공한다. Xavier는 구현 필수이고, 초기화 비교 실험에 넣는 것은 선택이다.

### 4. 학습 및 최적화

옵티마이저는 구한 기울기로 가중치를 얼마나, 어느 방향으로 고칠지 정하는 규칙이다. 만든 프레임워크로 실제 문제를 풀어 정상 동작을 확인한다.

- [ ] SGD(Stochastic Gradient Descent) 옵티마이저를 구현한다.
- [ ] Adam 옵티마이저를 구현한다.
- [ ] XOR 문제를 해결한다: He Init 기준 100 Epoch 내 Loss < 0.1 또는 정확도 95% 이상.
- [ ] MNIST 분류를 수행한다: 1 Epoch 정확도 80% 이상(소형 실험 기준, 성공 판정 표에서는 "선택").
- [ ] MNIST 데이터 로더를 자동 로더 없이 직접 구현한다.
- [ ] Zero/Random/He 3종으로 초기화 비교 실험을 수행한다(필수 비교 실험).
- [ ] Zero Init 실패를 재현한다: 50 Epoch 후 Loss 감소 없음(대칭성 문제 확인).
- [ ] 초기화 방식에 따라 학습 초기의 Loss 감소 속도가 달라지는 것을 그래프로 시각화하고 `figures/`에 저장한다.
- [ ] 초기화 실험 결과를 분석해 실험 리포트로 정리한다.

**성공 판정 최소 기준** (명세)

| 검증 항목 | 필수 기준 | 비고 |
|---|---|---|
| Gradient Check | 모든 레이어 오차 < 1e-7 | 상대 오차 기준 |
| XOR 문제 (필수) | 100 Epoch 내 Loss < 0.1 또는 정확도 95% 이상 | He Init 기준 |
| MNIST 분류 (선택) | 1 Epoch 정확도 80% 이상 | 소형 실험 기준 |
| Zero Init 실패 재현 | 50 Epoch 후 Loss 감소 없음 | 대칭성 문제 확인 |

### 5. README 작성

- [ ] 프로젝트 개요, 설치 방법, 실행 방법, 모듈 구조, 사용 예시, 실험 결과 요약 섹션을 작성한다([최종 결과물](#최종-결과물)의 필수 섹션 참고).
- [ ] `requirements.txt`를 만들고 설치 명령(`pip install -r requirements.txt`)을 적는다.

## 보너스 과제

### 보너스 1. Dropout 및 Regularization

정규화(Regularization)는 모델이 학습 데이터만 외우는 과적합을 막는 기법이다.

- [ ] 학습 시와 테스트 시의 동작이 다른 Dropout 레이어(Inverted Dropout 방식)를 구현한다.
- [ ] L2 Regularization(Weight Decay)을 옵티마이저에 추가한다.
- [ ] 과적합 억제 효과를 확인한다.

### 보너스 2. 다양한 활성화 함수 구현

- [ ] Tanh 활성화 함수와 역전파 로직을 구현한다.
- [ ] LeakyReLU 활성화 함수와 역전파 로직을 구현한다.
- [ ] ELU 등 추가 활성화 함수와 역전파 로직을 구현한다.

## 결과 예시 (명세 참고)

명세는 "정답이 아니라 참고 예시"이며 실제 문구와 구조는 달라도 된다고 했다. 예시는 Gradient Check 로그, 초기화 실험 결과 표, Loss 그래프 형식을 보여 준다.

**검증 로그 예시**

```
[Pass] Layer Linear_1 Gradient Check - Analytic: 0.0342, Numerical: 0.0342, Diff: 2.3e-08
[Pass] Layer Linear_2 Gradient Check - Analytic: -0.0156, Numerical: -0.0156, Diff: 1.8e-09
All gradient checks passed! Maximum difference: 2.3e-08 < 1e-07 ✓
```

**초기화 실험 결과 예시**

| Initialization | Epoch 5 Loss | Epoch 50 Loss | Converged? |
|---|---|---|---|
| Zero Init | 0.6931 | 0.6931 | No (Stuck) |
| Random Init | 0.4521 | 0.0892 | Yes (Slow) |
| He Init | 0.2134 | 0.0123 | Yes (Fast) |

- 실험 결론 예시: Zero Init은 대칭성 문제로 모든 뉴런이 같은 값을 학습해 Loss가 줄지 않고, Random Init은 분산이 커서 초반이 불안정하며, He Init은 빠르게 수렴했다는 식으로 정리한다.
- Loss 감소 그래프 예시: X축 Epoch (0-100), Y축 Loss (0-1.0), 세 가지 초기화 방식의 Loss 곡선을 한 그래프에 겹쳐 비교한다.

## 결과 요약

구현 후 채운다.

| 항목 | 명세 기준 | 결과 | 통과 |
|---|---|---|---|
| Gradient Check (모든 레이어) | 상대 오차 1e-7 이하 | | ⬜ |
| 수치 미분 ε | 1e-5 | | ⬜ |
| XOR (He Init) | 100 Epoch 내 Loss < 0.1 또는 정확도 95% 이상 | | ⬜ |
| MNIST 분류 | 1 Epoch 정확도 80% 이상 | | ⬜ |
| Zero Init 실패 재현 | 50 Epoch 후 Loss 감소 없음 | | ⬜ |
| 초기화 비교 실험 | Zero/Random/He 3종, 그래프 `figures/`에 저장 | | ⬜ |

## 명세에서 확인할 점

- **MNIST가 필수인지 선택인지**: 성공 판정 표에는 "MNIST 분류 (선택)"이라고 되어 있지만, 기능 요구 사항은 "XOR 문제 해결 및 MNIST 분류를 수행하여"라고 하고, 최종 결과물과 자가 체크리스트("MNIST 분류 시 1 Epoch에서 정확도 80% 이상인가?")에도 들어 있다.
- **오차 기준의 경계**: 자가 체크리스트와 요구 사항은 "1e-7 이하", 성공 판정 표는 "오차 < 1e-7"(미만)이다.
- **NumPy만 사용과 다른 라이브러리**: 제약 사항은 "실제 구현에는 NumPy만 사용한다"인데, 데이터셋 가이드는 Scikit-learn의 `load_digits()`, `make_moons()`를 권장하고, 초기화 비교 그래프를 그릴 시각화 라이브러리(예: Matplotlib)의 허용 여부는 적혀 있지 않다.
- **모듈 이름**: 최종 결과물은 "tensor, autograd, layer, optimizer 모듈", README 템플릿은 "core/nn/optim 간 의존 관계"로 적혀 있어 어느 이름으로 폴더를 나눌지 정해야 한다.
- **초기화 수식 표기**: He "N(0, √(2/n_in))", Xavier "N(0, √(2/(n_in+n_out)))"에서 두 번째 값이 표준편차인지 분산인지 적혀 있지 않다. 제곱근 형태로 보아 표준편차로 읽힌다.

## 참고 자료

- 과제 명세: [docs/A6-1.pdf](docs/A6-1.pdf)
- 명세 데이터셋 가이드: Scikit-learn `load_digits()`, `make_moons()`, Torchvision/Keras의 MNIST, CIFAR-10, Fashion-MNIST
