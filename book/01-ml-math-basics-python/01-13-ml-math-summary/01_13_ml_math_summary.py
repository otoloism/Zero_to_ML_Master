# -*- coding: utf-8 -*-
"""
01-13🤖 ML 수학 연결 총정리 — 모든 것을 이어서!

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 01권  수포자들을 위한 머신러닝 수학 완전기초와 파이썬 맛보기/01-13🤖 ML 수학 연결 총정리 — 모든 것을 이어서!.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 01-13-G 7단계 — 전체 코드: 미니 신경망으로 XOR 풀기
import numpy as np

class MiniNeuralNet:
    """2층 신경망 — 순전파부터 역전파까지 전부 손으로 구현합니다.

    비유:
    작은 공장입니다. 원재료(x)가 두 대의 기계(W1, W2)를 거쳐
    완제품(확률)이 되고, 불량이 나오면 각 기계에 책임을 물어
    설정을 조금씩 고칩니다."""

    def __init__(self, n_in, n_hidden, n_out, seed=0):
        rng = np.random.default_rng(seed)      # 시드 고정 → 결과 재현
        self.W1 = rng.normal(0, 0.5, (n_in, n_hidden))   # 🎲 정규분포 초기화(01-12)
        self.b1 = np.zeros(n_hidden)             # 🧱 편향은 0으로 시작
        self.W2 = rng.normal(0, 0.5, (n_hidden, n_out))
        self.b2 = np.zeros(n_out)

    def relu(self, x):
        """음수는 0으로, 양수는 그대로. 🔍 검수대 역할"""
        return np.maximum(0, x)

    def softmax(self, x):
        """로짓을 확률로. 최댓값 빼기는 오버플로 방지(01-12)"""
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def forward_pass(self, x):
        """① 순전파 — 역전파에서 쓸 중간값을 반드시 저장한다"""
        self.x = x
        self.z1 = x @ self.W1 + self.b1     # 🧱 선형변환 (01-04 행렬곱)
        self.h1 = self.relu(self.z1)         # 🔨 비선형 (01-08)
        self.z2 = self.h1 @ self.W2 + self.b2
        self.y_hat = self.softmax(self.z2)   # 🎲 확률로 (01-12)
        return self.y_hat

    def compute_loss(self, t):
        """② 손실 — 교차 엔트로피. 1e-15는 log(0) 방지용"""
        return -np.sum(t * np.log(self.y_hat + 1e-15))

    def backward_and_update(self, t, lr=0.3):
        """③ 역전파 + ④ 업데이트

        비유:
        사고 조사관이 블랙박스를 되감으며 각 부품의 책임을 매기고,
        책임만큼 설정을 조정합니다."""
        gz2 = self.y_hat - t                     # ⚡ 출발 신호 = 예측 − 정답
        gW2 = np.outer(self.h1, gz2)           # ⚡ W2의 책임 (외적)
        gb2 = gz2

        gz1 = (self.W2 @ gz2) * (self.z1 > 0)   # ⚡ 아래로 전달 × ReLU 문지기
        gW1 = np.outer(self.x, gz1)            # ⚡ W1의 책임
        gb1 = gz1

        self.W2 -= lr * gW2                      # ④ 경사하강: 기울기 반대로
        self.b2 -= lr * gb2
        self.W1 -= lr * gW1
        self.b1 -= lr * gb1

# ===== XOR 데이터 (정답은 원-핫 인코딩) =====
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
T = np.array([[1,0], [0,1], [0,1], [1,0]], dtype=float)

net = MiniNeuralNet(2, 8, 2)          # 입력 2 → 은닉 8 → 출력 2

for epoch in range(1, 301):                # ⑤ 300 에폭 반복
    total = 0.0
    for x_i, t_i in zip(X, T):
        net.forward_pass(x_i)
        total += net.compute_loss(t_i)
        net.backward_and_update(t_i)
    if epoch % 50 == 0 or epoch == 1:
        print(f"epoch {epoch:>3} | 평균 손실 {total/len(X):.4f}")

print()
for x_i, t_i in zip(X, T):
    p = net.forward_pass(x_i)
    print(f"입력 {x_i} → 예측 {np.argmax(p)} (확신 {p.max():.3f})  정답 {np.argmax(t_i)}")


# %% [Block 2] 01-13-J 10단계 — NumPy에서 PyTorch로: 60줄이 15줄이 된다
import torch
import torch.nn as nn

model = nn.Sequential(            # 🔨 층을 쌓는다 = 합성함수
    nn.Linear(2, 8),               # 🧱 xW1 + b1 (우리가 짠 그 줄)
    nn.ReLU(),                     # 🔨 np.maximum(0, x)
    nn.Linear(8, 2)                # 🧱 h1W2 + b2
)
loss_fn = nn.CrossEntropyLoss()   # 🎲 softmax + 교차엔트로피가 한 몸
optimizer = torch.optim.SGD(model.parameters(), lr=0.3)

X = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
T = torch.tensor([0, 1, 1, 0])   # 원-핫 대신 정답 인덱스만

for epoch in range(300):
    y_hat = model(X)              # ① 순전파  ← forward_pass()
    loss = loss_fn(y_hat, T)     # ② 손실     ← compute_loss()
    optimizer.zero_grad()       # ⚠️ 이전 그래디언트 청소 (필수!)
    loss.backward()             # ③ 역전파   ← 우리가 짠 5줄이 이 한 줄
    optimizer.step()            # ④ 업데이트 ← W -= lr * gW


# %% [Block 3] [추가·저장소 전용] 실습 8 풀이 — 죽은 뉴런 세기 (책 본문에는 문장만 실림, 10단계에서 X·T 가 torch 텐서로 바뀌었으므로 NumPy 용 X·T 를 다시 정의)
# ── 10단계(Block 2)에서 X, T 를 torch 텐서로 덮어썼으므로 7단계 NumPy 용 X, T 를 다시 정의합니다 ──
import numpy as np
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
T = np.array([[1,0], [0,1], [0,1], [1,0]], dtype=float)

# 실습 8 — 죽은 뉴런 세기 (7단계 MiniNeuralNet, X, T가 정의된 뒤 실행)
def train_xor(lr, seed=0, epochs=300):
    net = MiniNeuralNet(2, 8, 2, seed=seed)
    for epoch in range(epochs):
        total = 0.0
        for x_i, t_i in zip(X, T):
            net.forward_pass(x_i)
            total += net.compute_loss(t_i)
            net.backward_and_update(t_i, lr=lr)
    return net, total / len(X)

for lr in [0.3, 3.0]:
    net, loss = train_xor(lr)
    Z1 = X @ net.W1 + net.b1                    # (4, 8): 네 입력 × 은닉 뉴런 8개
    dead = np.all(Z1 <= 0, axis=0)              # 네 입력 모두에서 0 이하인 뉴런 = 죽은 뉴런
    print(f"lr={lr} | 최종 평균 손실 {loss:.4f} | 죽은 뉴런 {dead.sum()}/8 → {np.where(dead)[0].tolist()}")


# %% [Block 4] [추가·저장소 전용] 🐞 디버깅 팁 — 죽은 뉴런 수 한 줄 (본문 인라인 코드의 실행 예시)
# 바로 위 실습 8 루프의 마지막 net(lr=3.0)에서 가중치를 꺼내 한 줄 공식을 그대로 확인합니다
W1, b1 = net.W1, net.b1
# 죽은 뉴런 수 = 네 입력 모두에서 z1 ≤ 0 인 은닉 뉴런 개수 (NumPy 7단계 기준)
dead_count = np.all(X @ W1 + b1 <= 0, axis=0).sum()      # 클래스 안이라면 net.W1, net.b1
print("죽은 뉴런 수:", dead_count, "/ 8")


# %% [Block 5] 01-13 연습문제 Q14 — ReLU vs LeakyReLU vs He 초기화 (NumPy, lr 0.5 / 3.0)
import numpy as np
# ⚠️ 7단계의 MiniNeuralNet 클래스와 X, T를 먼저 실행해 두세요.

class MiniNeuralNet2(MiniNeuralNet):
    """7단계 신경망에 스위치 두 개만 추가: act(활성화 함수), init(초기화 방식)"""
    def __init__(self, n_in, n_hidden, n_out, seed=0, act="relu", init="normal", alpha=0.01):
        super().__init__(n_in, n_hidden, n_out, seed)
        self.act, self.alpha = act, alpha
        if init == "he":                                  # He 초기화: 표준편차 = √(2 / 입력 개수)
            rng = np.random.default_rng(seed)
            self.W1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden))
            self.W2 = rng.normal(0, np.sqrt(2 / n_hidden), (n_hidden, n_out))

    def relu(self, x):                                    # forward_pass가 이 함수를 부릅니다
        if self.act == "leaky":
            return np.where(x > 0, x, self.alpha * x)     # 음수도 0.01배만큼은 통과
        return np.maximum(0, x)

    def backward_and_update(self, t, lr=0.3):
        gz2 = self.y_hat - t
        gW2 = np.outer(self.h1, gz2)
        # TODO ① 문지기(활성화 미분): ReLU는 0 또는 1, LeakyReLU는 alpha 또는 1
        d = (self.z1 > 0) if self.act == "relu" else np.where(self.z1 > 0, 1.0, self.alpha)
        gz1 = (self.W2 @ gz2) * d
        gW1 = np.outer(self.x, gz1)
        self.W2 -= lr * gW2;  self.b2 -= lr * gz2
        self.W1 -= lr * gW1;  self.b1 -= lr * gz1

def count_dead(net, X):
    """네 입력 모두에서 z1 ≤ 0 인 은닉 뉴런 수 = 한 번도 '일'하지 않은 뉴런"""
    Z1 = X @ net.W1 + net.b1                              # (4, 8)
    return int(np.all(Z1 <= 0, axis=0).sum())

def train(lr, act, init, seed=0, epochs=300):
    net = MiniNeuralNet2(2, 8, 2, seed=seed, act=act, init=init)
    for epoch in range(epochs):
        total = 0.0
        for x_i, t_i in zip(X, T):
            net.forward_pass(x_i)
            total += net.compute_loss(t_i)
            net.backward_and_update(t_i, lr=lr)
    return total / len(X), count_dead(net, X)

for lr in [0.5, 3.0]:
    for act, init in [("relu", "normal"), ("leaky", "normal"), ("relu", "he"), ("leaky", "he")]:
        results = [train(lr, act, init, seed=s) for s in range(10)]
        # TODO ② 시드 10개 중 '성공(손실 < 0.05)' 횟수와 평균 죽은 뉴런 수를 출력해 보세요
        ok = sum(loss < 0.05 for loss, dead in results)
        avg_dead = np.mean([dead for loss, dead in results])
        print(f"lr={lr:<4} {act:>5} + {init:<6} | 성공 {ok}/10 | 평균 죽은 뉴런 {avg_dead:.1f}/8")


# %% [Block 6] 01-13 연습문제 Q14 — PyTorch 버전
import torch
import torch.nn as nn

X = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
T = torch.tensor([0, 1, 1, 0])

def make_model(act="relu", init="default", seed=0):
    torch.manual_seed(seed)
    act_layer = nn.LeakyReLU(0.01) if act == "leaky" else nn.ReLU()   # 🔨 활성화 교체는 한 줄
    model = nn.Sequential(nn.Linear(2, 8), act_layer, nn.Linear(8, 2))
    if init == "he":                                                  # 🎲 He(Kaiming) 초기화
        nn.init.kaiming_normal_(model[0].weight, nonlinearity="relu")
        nn.init.zeros_(model[0].bias)
    return model

def train(lr, act, init, seed=0, epochs=300):
    model = make_model(act, init, seed)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        loss = loss_fn(model(X), T)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        dead = (model[0](X) <= 0).all(dim=0).sum().item()   # 네 입력 모두에서 꺼진 뉴런 수
    return loss.item(), dead

for lr in [0.5, 3.0]:
    for act, init in [("relu", "default"), ("leaky", "default"), ("relu", "he"), ("leaky", "he")]:
        results = [train(lr, act, init, seed=s) for s in range(10)]
        ok = sum(l < 0.05 for l, d in results)
        print(f"lr={lr:<4} {act:>5} + {init:<7} | 성공 {ok}/10 | 평균 죽은 뉴런 {sum(d for l, d in results)/10:.1f}/8")


# %% [Block 7] [추가·저장소 전용] 🔁 PyTorch에서는 처방도 한 줄 — LeakyReLU · He(Kaiming) 초기화 (10단계 모델에 적용하는 예시)
import torch.nn as nn

# ② ReLU → LeakyReLU (미분은 오토그라드가 처리)
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.LeakyReLU(0.01),      # 기존: nn.ReLU()
    nn.Linear(8, 2)
)

# ③ He(Kaiming) 초기화 — 첫 번째 Linear 층에 적용
nn.init.kaiming_normal_(model[0].weight, nonlinearity="relu")
nn.init.zeros_(model[0].bias)
