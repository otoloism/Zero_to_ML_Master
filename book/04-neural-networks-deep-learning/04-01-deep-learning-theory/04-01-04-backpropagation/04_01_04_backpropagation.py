# -*- coding: utf-8 -*-
"""
04-01-04 오차역전파법 — 계산 그래프로 이해하기

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-01장 딥러닝 이론과 구현/04-01-04 오차역전파법 — 계산 그래프로 이해하기.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-01-04-C 덧셈 & 곱셈 계층 — "역전파의 가장 기본 부품"
class MulLayer:
    """곱셈 계층: 순전파에서 곱하고, 역전파에서 입력을 교차.
    비유: '환율 교환소' — 내 것과 상대 것을 서로 바꿔치기."""

    def forward(self, x, y):
        self.x = x                       # 역전파용으로 저장 (캐싱)
        self.y = y                       # 역전파용으로 저장
        return x * y                     # 순전파: 곱하기

    def backward(self, dout):
        # 핵심: 입력을 서로 바꿔서 곱하기!
        dx = dout * self.y               # x의 기울기 = dout × y
        dy = dout * self.x               # y의 기울기 = dout × x
        return dx, dy


# %% [Block 2] 04-01-04-C 덧셈 & 곱셈 계층 — "역전파의 가장 기본 부품"
class AddLayer:
    """덧셈 계층: 순전파에서 더하고, 역전파에서 그대로 통과.
    비유: '택배 합치기' — 두 상자를 합쳐도, 각자 무게는 변하지 않음."""

    def forward(self, x, y):
        return x + y                     # 순전파: 더하기

    def backward(self, dout):
        # 덧셈의 역전파는 초간단: 그대로 양쪽에 전달!
        dx = dout                        # x의 기울기 = dout (그대로)
        dy = dout                        # y의 기울기 = dout (그대로)
        return dx, dy


# %% [Block 3] 04-01-04-C 덧셈 & 곱셈 계층 — "역전파의 가장 기본 부품" — 사과 100원 × 2개 = 200원, 소비세 1.1배 → 총 220원
# 사과 100원 × 2개 = 200원, 소비세 1.1배 → 총 220원
mul_apple = MulLayer()
mul_tax = MulLayer()

# ── 순전파 ──
apple_price = 100
apple_num = 2
tax = 1.1

apple_total = mul_apple.forward(apple_price, apple_num)  # 200
price = mul_tax.forward(apple_total, tax)               # 220

# ── 역전파 ──
dprice = 1                                              # 출발점: 1
dapple_total, dtax = mul_tax.backward(dprice)
dapple_price, dapple_num = mul_apple.backward(dapple_total)

print(f"총 가격: {price}")
print(f"사과 가격 기울기: {dapple_price}")   # 2.2 (1원↑ → 2.2원↑)
print(f"사과 개수 기울기: {dapple_num}")     # 110 (1개↑ → 110원↑)
print(f"소비세 기울기: {dtax}")               # 200


# %% [Block 4] 04-01-04-D ReLU & Sigmoid 계층 — "활성화 함수도 역전파 가능!"
import numpy as np

class Relu:
    """ReLU 계층: 양수는 통과, 음수는 차단.
    비유: '한쪽으로만 열리는 문' — 양수만 들여보냄."""

    def forward(self, x):
        self.mask = (x <= 0)          # 음수 위치를 기록 (마스크)
        out = x.copy()                  # 원본 보존을 위해 복사
        out[self.mask] = 0              # 음수 위치를 0으로 차단
        return out

    def backward(self, dout):
        dout[self.mask] = 0             # 순전파 때 0이었던 곳은 기울기도 0!
        dx = dout                       # 양수였던 곳은 기울기 그대로 통과
        return dx

# ── 테스트 ──
relu = Relu()
x = np.array([[-1.0, 2.0], [3.0, -0.5]])
out = relu.forward(x)
print("순전파:", out)

dout = np.array([[1.0, 1.0], [1.0, 1.0]])
dx = relu.backward(dout)
print("역전파:", dx)


# %% [Block 5] 04-01-04-E Affine 계층 — "신경망의 핵심 엔진"
class Affine:
    """Affine 계층: 행렬곱 + 편향 = 신경망 한 층의 핵심.
    비유: '여러 다리를 건너는 신호' — W가 다리, b가 출발 보너스."""

    def __init__(self, W, b):
        self.W = W                       # 가중치 행렬
        self.b = b                       # 편향 벡터
        self.x = None                    # 역전파용 캐시
        self.dW = None                   # W의 기울기 (학습에 사용)
        self.db = None                   # b의 기울기

    def forward(self, x):
        self.x = x                       # 입력 저장 (역전파에서 dW 계산에 필요)
        out = np.dot(x, self.W) + self.b # Y = X·W + b (04-01-02 순전파!)
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)      # dX = dY · W^T (입력의 기울기)
        self.dW = np.dot(self.x.T, dout) # dW = X^T · dY (가중치의 기울기)
        self.db = np.sum(dout, axis=0)   # dB = sum(dY) (편향의 기울기)
        return dx

# ── 테스트 ──
np.random.seed(0)
W = np.random.randn(2, 3)              # 입력 2개 → 출력 3개
b = np.zeros(3)
affine = Affine(W, b)

x = np.array([[1.0, 0.5]])             # 배치 1개, 입력 2개
out = affine.forward(x)
print(f"순전파 출력: {np.round(out, 3)}")
print(f"출력 shape: {out.shape}")       # (1, 3)

dout = np.array([[1.0, 0.5, 0.2]])
dx = affine.backward(dout)
print(f"역전파 dx: {np.round(dx, 3)}")
print(f"dW shape: {affine.dW.shape}")  # (2, 3) — W와 같은 shape!


# %% [Block 6] 04-01-04-F Softmax-with-Loss — "역전파의 결과는 놀랍도록 간단"
def softmax(a):
    """04-01-02에서 구현한 소프트맥스 (오버플로 방지 포함)."""
    c = np.max(a, axis=1, keepdims=True)
    exp_a = np.exp(a - c)
    return exp_a / np.sum(exp_a, axis=1, keepdims=True)

def cross_entropy_error(y, t):
    """04-01-03에서 구현한 교차 엔트로피."""
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

class SoftmaxWithLoss:
    """소프트맥스 + 교차 엔트로피를 합친 출력층.
    비유: '성적표' — 예측 점수와 만점의 차이를 그대로 피드백."""

    def forward(self, x, t):
        self.t = t                       # 정답 저장
        self.y = softmax(x)              # 소프트맥스로 확률 변환
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size  # 핵심! y−t가 전부!
        return dx

# ── 테스트 ──
loss_layer = SoftmaxWithLoss()
x = np.array([[0.3, 2.9, 4.0]])           # 신경망 출력 (logit)
t = np.array([[0, 0, 1]])                  # 정답: 2번 클래스

loss = loss_layer.forward(x, t)
dx = loss_layer.backward()

print(f"손실: {loss:.4f}")
print(f"예측 확률 y: {np.round(loss_layer.y, 3)}")
print(f"역전파 dx:  {np.round(dx, 3)}")
print(f"  → dx = y - t = {np.round(loss_layer.y - t, 3)}")
