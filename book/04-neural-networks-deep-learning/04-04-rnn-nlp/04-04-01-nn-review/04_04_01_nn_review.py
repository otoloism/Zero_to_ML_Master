# -*- coding: utf-8 -*-
"""
04-04-01 신경망 복습

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-04장 순환 신경망과 자연어 처리/04-04-01 신경망 복습.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-04-01-A 벡터와 행렬 — 데이터를 숫자로 담는 그릇
import numpy as np  # NumPy: 숫자 배열을 다루는 핵심 라이브러리

# 벡터 = 1차원 배열 (숫자를 한 줄로 나열)
a = np.array([1, 2, 3])
print("벡터 a:", a)
print("  모양(shape):", a.shape)   # (3,) → 원소 3개짜리 벡터

# 행렬 = 2차원 배열 (숫자를 행×열로 배치)
W = np.array([[1, 4],
              [2, 5],
              [3, 6]])
print("행렬 W:", W)
print("  모양(shape):", W.shape)   # (3, 2) → 3행 2열

# 행렬 곱: (3,) @ (3,2) = (2,)  ← 가운데 3이 맞아야 곱셈 가능!
y = np.dot(a, W)                   # a @ W 와 동일
print("행렬 곱 결과:", y)
print("  모양(shape):", y.shape)   # (2,) → 2개짜리 벡터로 변환됨


# %% [Block 2] 04-04-01-B 순전파 — 입력에서 출력까지의 여행
import numpy as np

class Affine:
    """선형 변환 계층: y = xW + b
    비유: '저울에 달아 배합하는 장치' — 입력에 가중치를 곱하고 편향을 더함"""

    def __init__(self, W, b):
        self.W = W                    # 가중치 행렬 (입력차원 × 출력차원)
        self.b = b                    # 편향 벡터 (출력차원,)
        self.x = None                # 역전파용 입력 캐싱
        self.dW = None               # W의 기울기 (역전파 결과)
        self.db = None               # b의 기울기 (역전파 결과)

    def forward(self, x):
        """순전파: 입력 x를 선형 변환"""
        self.x = x                    # 역전파 때 쓸 입력을 저장
        return np.dot(x, self.W) + self.b  # y = xW + b

    def backward(self, dout):
        """역전파: 출력 쪽 기울기 dout을 받아 입력 쪽 기울기 dx를 반환"""
        dx = np.dot(dout, self.W.T)  # 입력에 대한 기울기 — 04-01-04 공식
        self.dW = np.dot(self.x.T, dout)  # 가중치에 대한 기울기
        self.db = np.sum(dout, axis=0)     # 편향에 대한 기울기
        return dx

# 실제 사용 예시
np.random.seed(42)
x = np.array([[1.0, 0.5]])           # 입력: 1×2 (데이터 1개, 특징 2개)
W = np.random.randn(2, 3) * 0.01    # 가중치: 2×3 (입력2 → 출력3)
b = np.zeros(3)                      # 편향: (3,) 0으로 초기화

affine = Affine(W, b)
y = affine.forward(x)
print("입력 x의 모양:", x.shape)      # (1, 2)
print("가중치 W의 모양:", W.shape)    # (2, 3)
print("출력 y의 모양:", y.shape)      # (1, 3)
print("출력 y:", y.round(4))


# %% [Block 3] 04-04-01-C 손실 함수 — "얼마나 틀렸는지" 채점하기
import numpy as np

def cross_entropy_error(y, t):
    """교차 엔트로피 손실
    비유: '정답 확률에 벌점을 매기는 채점기'
    y: 신경망의 예측 확률 (소프트맥스 출력)
    t: 정답 레이블 (원-핫 벡터 또는 인덱스)"""
    delta = 1e-7                      # log(0) 방지용 아주 작은 값
    return -np.sum(t * np.log(y + delta))

# 예시 1: 정답이 "고양이"(2번)이고, 잘 맞춘 경우
y_good = np.array([0.01, 0.01, 0.98])  # "고양이" 확률 98%
t      = np.array([0,    0,    1])       # 정답: 고양이 (원-핫)
print("잘 맞춘 경우 손실:", round(cross_entropy_error(y_good, t), 4))

# 예시 2: 많이 틀린 경우
y_bad  = np.array([0.8,  0.1,  0.1])   # "강아지" 확률 80%
print("많이 틀린 경우 손실:", round(cross_entropy_error(y_bad, t), 4))


# %% [Block 4] 04-04-01-D 역전파 — 오답 노트로 실력 올리기
class ReLU:
    """ReLU 활성화 함수
    비유: '양수만 통과시키는 문지기' — 음수는 0으로, 양수는 그대로 통과"""

    def __init__(self):
        self.mask = None              # 어떤 원소가 0 이하였는지 기록

    def forward(self, x):
        self.mask = (x <= 0)         # 0 이하인 위치를 True로 표시
        out = x.copy()
        out[self.mask] = 0           # True인 위치를 0으로 바꿈
        return out

    def backward(self, dout):
        dout[self.mask] = 0          # 순전파에서 차단된 곳은 역전파도 차단
        return dout

# 테스트
relu = ReLU()
x = np.array([[-1.0, 2.0, -0.5, 3.0]])
print("입력:", x)
print("ReLU 출력:", relu.forward(x))   # 음수 → 0, 양수 → 그대로


# %% [Block 5] 04-04-01-E 미니배치 학습 — 매일 조금씩 공부하기 — 학습 루프의 5단계 패턴 (모든 신경망 학습의 기본!)
# === 학습 루프의 5단계 패턴 (모든 신경망 학습의 기본!) ===
# 아래는 개념 설명용 의사코드(pseudo-code)입니다

for epoch in range(max_epoch):          # 전체 데이터를 여러 번 반복
    for batch in mini_batches:
        y_pred = model(x_batch)         # ① 예측 (순전파)
        loss = cross_entropy(y_pred, t) # ② 채점 (손실 계산)
        model.cleargrads()               # ③ 이전 기울기 초기화
        loss.backward()                   # ④ 역전파 (기울기 계산)
        optimizer.update()               # ⑤ 가중치 갱신 (학습률만큼 이동)


# %% [Block 6] 04-04-01-F 04-04장의 도구 — TwoLayerNet과 Trainer
class TwoLayerNet:
    """2층 신경망: 입력→Affine→ReLU→Affine→Softmax
    비유: '2단계 공장 라인' — 1차 가공(Affine+ReLU) → 2차 가공(Affine) → 출하(Softmax)"""

    def __init__(self, input_size, hidden_size, output_size):
        # 가중치 초기화 (작은 난수로 시작)
        W1 = np.random.randn(input_size, hidden_size) * 0.01
        b1 = np.zeros(hidden_size)
        W2 = np.random.randn(hidden_size, output_size) * 0.01
        b2 = np.zeros(output_size)

        # 계층 조립 (04-01-04의 계층들을 순서대로 배치)
        self.layers = [
            Affine(W1, b1),      # 1층: 선형 변환
            ReLU(),               # 활성화: 음수 차단
            Affine(W2, b2),      # 2층: 선형 변환
        ]

    def predict(self, x):
        """순전파: 계층을 차례로 통과"""
        for layer in self.layers:
            x = layer.forward(x)    # 각 계층의 출력이 다음 계층의 입력
        return x
