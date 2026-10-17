# -*- coding: utf-8 -*-
"""
Zero to 머신러닝 딥러닝 Master — Medium 시리즈 #26 · 1부
ML 수학 총정리 — 순전파·손실·역전파·경사하강법 5단계를 손계산과 NumPy XOR 신경망으로 (1부)

원문(책): 01-13 「ML 수학 연결 총정리 — 모든 것을 이어서! (1부)」 https://wikidocs.net/439749
Medium : TODO (게시 후 추가)
책     : https://wikidocs.net/book/21464
홈페이지: https://mldict.net
블로그 : https://wikidocs.net/blog/@mldict/

실행: pip install -r requirements.txt && python 26_1_xor_mini_neural_net.py
※ 아래 코드는 책 본문의 코드 블록을 순서대로 옮긴 것입니다. [추가]/[수정] 표시가 있는 부분만 실행을 위해 보탰습니다.
"""

#======================================================================
# [1] 01-13-G 7단계 — 미니 신경망으로 XOR 풀기 (NumPy)
#     MiniNeuralNet 클래스(forward_pass · compute_loss · backward_and_update)로 XOR 4문제를 300 에폭 학습하고, 50 에폭마다 평균 손실과 최종 예측을 출력합니다. 시드 0 고정이라 책과 같은 숫자가 나옵니다.
#     ⚠️ 책의 「🧪 직접 확인해보기 — 학습률을 3.0으로 바꾸면?」은 결과만 있어, 같은 신경망을 lr=3.0으로 다시 학습시키는 코드를 추가했습니다(표시된 부분).
#======================================================================
print("\n" + "=" * 60)
print("[1] 01-13-G 7단계 — 미니 신경망으로 XOR 풀기 (NumPy)")
print("=" * 60)
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

# [추가 코드 — 책 본문에는 없음] 🧪 직접 확인해보기 — 학습률을 3.0으로 바꾸면? (책: 손실이 1.5861에서 멈춤)
net_big = MiniNeuralNet(2, 8, 2)      # 같은 시드(0)로 다시 시작
for epoch in range(1, 301):
    total = 0.0
    for x_i, t_i in zip(X, T):
        net_big.forward_pass(x_i)
        total += net_big.compute_loss(t_i)
        net_big.backward_and_update(t_i, lr=3.0)   # 보폭만 0.3 → 3.0
    if epoch in (1, 50, 100, 300):
        print(f"epoch {epoch:>3} | 평균 손실 {total/len(X):.4f}")
