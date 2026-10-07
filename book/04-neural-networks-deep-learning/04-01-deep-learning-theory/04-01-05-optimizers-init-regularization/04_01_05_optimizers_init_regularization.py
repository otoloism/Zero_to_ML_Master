# -*- coding: utf-8 -*-
"""
04-01-05 학습 관련 기술들 — 옵티마이저·초기화·정규화

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-01장 딥러닝 이론과 구현/04-01-05 학습 관련 기술들 — 옵티마이저·초기화·정규화.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-01-05-A SGD의 한계 — "지그재그로 내려가는 비효율"
class SGD:
    """확률적 경사 하강법.
    비유: '눈 감고 경사만 느껴서 내려가기' — 단순하지만 비효율적일 수 있음."""

    def __init__(self, lr=0.01):
        self.lr = lr                     # 학습률 (한 걸음 크기)

    def update(self, params, grads):
        """모든 파라미터를 기울기 반대 방향으로 이동."""
        for key in params:
            params[key] -= self.lr * grads[key]  # W ← W − lr × dW


# %% [Block 2] 04-01-05-B Momentum & AdaGrad — "관성 / 학습률 자동 조절"
import numpy as np

class Momentum:
    """모멘텀: 관성을 이용한 경사 하강법.
    비유: '볼링공 굴리기' — 한번 방향이 잡히면 쉽게 안 멈춤."""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum         # α: 속도 유지 비율 (0.9 = 90% 유지)
        self.v = None                    # 속도 (처음엔 없음)

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key in params:
                self.v[key] = np.zeros_like(params[key])  # 속도 초기화

        for key in params:
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            # ↑ 이전 속도의 90% 유지 + 새 기울기 반영
            params[key] += self.v[key]    # 속도 방향으로 이동


# %% [Block 3] 04-01-05-B Momentum & AdaGrad — "관성 / 학습률 자동 조절"
class AdaGrad:
    """AdaGrad: 학습률을 자동 조절.
    비유: '많이 달린 선수는 천천히, 덜 달린 선수는 빠르게'"""

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None                    # 기울기 누적 제곱합

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key in params:
                self.h[key] = np.zeros_like(params[key])

        for key in params:
            self.h[key] += grads[key] ** 2        # 기울기² 누적
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
            # ↑ √h가 크면(많이 움직였으면) → 실질 학습률 작아짐


# %% [Block 4] 04-01-05-C Adam — "실전 기본값. 일단 Adam 쓰세요"
class Adam:
    """Adam: Momentum + AdaGrad 통합.
    비유: '내비게이션 + 자동변속기' — 방향도, 속도도 자동 조절."""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1               # Momentum 계수 (1차 모멘트)
        self.beta2 = beta2               # AdaGrad 계수 (2차 모멘트)
        self.m = None                    # 1차 모멘트 (기울기의 평균)
        self.v = None                    # 2차 모멘트 (기울기²의 평균)
        self.t = 0                        # 타임스텝

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key in params:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

        self.t += 1
        for key in params:
            # 1차 모멘트: 기울기의 지수 이동 평균 (방향)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            # 2차 모멘트: 기울기²의 지수 이동 평균 (크기)
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            # 편향 보정 (초반 추정 부정확성 보정)
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # 파라미터 갱신
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-7)


# %% [Block 5] 04-01-05-D 가중치 초기화 — "올바른 출발선에 서기"
import numpy as np

n_in = 784                              # 입력 뉴런 수 (MNIST)
n_out = 50                              # 출력 뉴런 수

# ❌ 나쁜 초기화: 너무 큰 값
W_bad = np.random.randn(n_in, n_out) * 1.0
print(f"나쁜 초기화 — 표준편차: {W_bad.std():.4f}")  # ~1.0 (너무 큼!)

# ✅ Xavier 초기화: 시그모이드/tanh에 적합
# 표준편차 = 1 / √(입력 뉴런 수)
W_xavier = np.random.randn(n_in, n_out) / np.sqrt(n_in)
print(f"Xavier  — 표준편차: {W_xavier.std():.4f}")  # ~0.036

# ✅ He 초기화: ReLU에 적합 (현대 표준!)
# 표준편차 = √(2 / 입력 뉴런 수)
W_he = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
print(f"He      — 표준편차: {W_he.std():.4f}")      # ~0.051


# %% [Block 6] 04-01-05-E 배치 정규화(BN) — "층마다 물 마시기"
import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-7):
    """배치 정규화 순전파.
    비유: '매 쉬는시간마다 컨디션을 리셋' — 분포가 흐트러지지 않게."""
    mu = np.mean(x, axis=0)             # ① 배치 평균
    var = np.var(x, axis=0)              # ② 배치 분산
    x_hat = (x - mu) / np.sqrt(var + eps)  # ③ 정규화 (평균0, 분산1)
    out = gamma * x_hat + beta           # ④ 스케일·시프트 (학습 가능!)
    return out

# ── 테스트: BN 적용 전후 비교 ──
np.random.seed(0)
x = np.random.randn(4, 3) * 5 + 10    # 평균~10, 표준편차~5 (불안정!)
gamma = np.ones(3)                     # 스케일 (처음엔 1)
beta = np.zeros(3)                     # 시프트 (처음엔 0)

out = batch_norm_forward(x, gamma, beta)

print("[BN 적용 전]")
print(f"  평균: {np.round(x.mean(axis=0), 2)}")
print(f"  표준편차: {np.round(x.std(axis=0), 2)}")
print("[BN 적용 후]")
print(f"  평균: {np.round(out.mean(axis=0), 2)}")
print(f"  표준편차: {np.round(out.std(axis=0), 2)}")


# %% [Block 7] 04-01-05-F 오버피팅 & 드롭아웃 — "연습만 잘하면 안 된다!"
class Dropout:
    """드롭아웃: 학습 시 뉴런을 랜덤으로 비활성화.
    비유: '조별 과제에서 매번 다른 팀원 빠지기'
    → 모든 팀원이 독립적으로 일할 수 있게 훈련!"""

    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio  # 50%의 뉴런을 끔
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            # 학습 시: 랜덤 마스크 생성 → 일부 뉴런 차단
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask             # 꺼진 뉴런은 0
        else:
            # 추론 시: 모든 뉴런 사용, 대신 비율만큼 스케일 다운
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask              # 순전파 때 꺼진 곳 → 역전파도 차단

# ── 테스트 ──
np.random.seed(0)
x = np.ones((1, 10))                    # 뉴런 10개, 모두 1
dp = Dropout(dropout_ratio=0.3)        # 30% 끄기
out = dp.forward(x, train_flg=True)
print(f"입력:   {x.astype(int)}")
print(f"출력:   {out.astype(int)}")        # 일부가 0으로 꺼짐!
print(f"마스크: {dp.mask.astype(int)}")
