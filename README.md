# ML/DL 실습 강의 자료

머신러닝부터 딥러닝·LLM까지 4일 과정의 실습 노트북 모음입니다. 각 노트북은 한글 설명과 함께 바로 실행 가능하도록 구성되어 있습니다.

## 강의 일정 & 실습 내용

### 📅 1일차

**오전 — 인공지능 개요 및 현황**
- 환경 준비 및 파이썬 기초

| 실습 | 노트북 |
|---|---|
| 파이썬 기초 | [Python_basic.ipynb](Python_basic.ipynb) |
| NumPy 기초 | [numpy_example.ipynb](numpy_example.ipynb) |
| Pandas 기초 | [pandas_example.ipynb](pandas_example.ipynb) |
| 패키지 설치 안내 | [requirements_txt.ipynb](requirements_txt.ipynb) |

**오후 — 머신러닝: 지도학습**

| 실습 | 노트북 |
|---|---|
| 선형 회귀 | [linear_regression.ipynb](linear_regression.ipynb) |
| KNN (붓꽃 분류) | [IRIS_KNN.ipynb](IRIS_KNN.ipynb) |
| 결정 트리 | [Decision_Tree.ipynb](Decision_Tree.ipynb) |
| SVM | [SVM.ipynb](SVM.ipynb) · [SVM_Kernel_example.ipynb](SVM_Kernel_example.ipynb) · [SVM_rbf_example.ipynb](SVM_rbf_example.ipynb) |
| AdaBoost (앙상블) — 와인 품종 분류 | [AdaBoost_Example.ipynb](AdaBoost_Example.ipynb) |

### 📅 2일차

**오전 — 머신러닝: 비지도학습**

| 실습 | 노트북 |
|---|---|
| 비지도학습 개요 | [Unsupervised_ipynb.ipynb](Unsupervised_ipynb.ipynb) |
| 군집화 (Clustering) | [clustering_examples.ipynb](clustering_examples.ipynb) |
| 차원 축소 | [dimensionality_reduction.ipynb](dimensionality_reduction.ipynb) |
| SVD (특이값 분해) | [SVD.ipynb](SVD.ipynb) |

**오후 — 머신러닝: 강화학습**

| 실습 | 노트북 |
|---|---|
| Q-learning — CartPole | [RL_Gym_CartPole_Qlearning.ipynb](RL_Gym_CartPole_Qlearning.ipynb) |
| Q-learning — FrozenLake | [RL_Gym_FrozenLake_v1_Qlearning_ipynb.ipynb](RL_Gym_FrozenLake_v1_Qlearning_ipynb.ipynb) |
| Q-learning — 택시 게임 (+ 정책 지도 시각화) | [RL_Gym_Taxi_Qlearning.ipynb](RL_Gym_Taxi_Qlearning.ipynb) |

### 📅 3일차

**오전 — 머신러닝 실습**
- 1~2일차 지도/비지도/강화학습 노트북 종합 실습

**오후 — 딥러닝 개요**

| 실습 | 노트북 |
|---|---|
| 자동 미분 (Autograd) | [autograd_example.ipynb](autograd_example.ipynb) |
| PyTorch 기초 | [torch_example.ipynb](torch_example.ipynb) |
| 단층 퍼셉트론 (SLP) — MNIST | [mnist_slp.ipynb](mnist_slp.ipynb) |
| 다층 퍼셉트론 (MLP) — MNIST | [mnist_mlp.ipynb](mnist_mlp.ipynb) |

### 📅 4일차

**오전 — 딥러닝 모델**

| 실습 | 노트북 |
|---|---|
| CNN (합성곱 신경망) — MNIST | [mnist_cnn.ipynb](mnist_cnn.ipynb) |
| LSTM (순환 신경망) — 노이즈 사인파 시계열 예측 | [LSTM_Example.ipynb](LSTM_Example.ipynb) |
| Seq2Seq | [seq2seq_example.ipynb](seq2seq_example.ipynb) |

**오후 — LLM 개요 및 딥러닝 모델 실습**

| 실습 | 노트북 |
|---|---|
| 토큰화 (Tokenization) | [tokenization.ipynb](tokenization.ipynb) |
| 워드 임베딩 (Word2Vec) | [word2vec_embedding.ipynb](word2vec_embedding.ipynb) |

## 실행 환경

- Python 3.11 (가상환경 권장)
- 주요 패키지: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `torch`, `gymnasium`
- 한글 그래프 폰트: Ubuntu 기준 `NanumGothic` (`sudo apt install fonts-nanum`)

```bash
# 패키지 설치
pip install numpy pandas matplotlib scikit-learn torch gymnasium

# Jupyter 실행
jupyter lab
```

> 노트북의 matplotlib 그래프에서 한글이 깨질 경우, 상단 import 셀의 폰트 설정(`plt.rcParams['font.family'] = 'NanumGothic'`)을 확인하세요.

## 참고 자료

- Style-based Generator (StyleGAN) 데모 영상: https://www.youtube.com/watch?v=kSLJriaOumA
