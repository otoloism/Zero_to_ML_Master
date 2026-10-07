# -*- coding: utf-8 -*-
"""
Zero to 머신러닝 딥러닝 Master — Medium 시리즈 #08 · 2부
SVD 추천 시스템부터 미니 신경망까지 — 수포자 수학 6개 장을 하나로 (2부)

원문(책): 00-06 「SVD·PCA — 300페이지 책을 3페이지로 (2부)」 https://wikidocs.net/439842
Medium : TODO (게시 후 추가)
책     : https://wikidocs.net/book/21464
홈페이지: https://mldict.net
블로그 : https://wikidocs.net/blog/@mldict/

실행: pip install -r requirements.txt && python 08_2_svd_recsys_mini_nn.py
※ 아래 코드는 책 본문의 코드 블록을 순서대로 옮긴 것입니다(내용 변경 없음).
"""

#======================================================================
# [1] 00-06-D · 4단계 — SVD 추천 시스템
#     0을 그대로 둔 순진한 SVD와, 사용자별 평균으로 채우고 중심화한 개선 버전의 예측·RMSE를 비교합니다.
#======================================================================
print("\n" + "=" * 60)
print("[1] 00-06-D · 4단계 — SVD 추천 시스템")
print("=" * 60)
import numpy as np
np.set_printoptions(suppress=True)

# 사용자 5명 × 영화 5편, 0은 '아직 평가하지 않음'
R = np.array([
    [5, 4, 0, 1, 2],   # 사용자0: 영화0,1을 좋아함
    [4, 5, 1, 0, 3],   # 사용자1: 비슷한 취향
    [1, 2, 5, 4, 0],   # 사용자2: 영화2,3을 좋아함
    [2, 0, 4, 5, 1],
    [0, 1, 3, 5, 4]
], dtype=float)

mask = R > 0          # 관측된 평점의 위치 (00-05의 불리언 마스크! 🔗)
print(f"관측 평점: {mask.sum()}개 / 전체 {R.size}개")

def svd_predict(matrix, k):
    """
    SVD 랭크-k 근사로 평점을 예측합니다.

    비유:
    퍼즐의 큰 패턴 k개만 파악해서 빈칸을 추론하는 것과 같습니다.
    """
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]   # 2단계와 동일!

# ══════ 버전 A) 순진한 방법 — 0을 그대로 두고 SVD ══════
U, S, Vt = np.linalg.svd(R, full_matrices=False)
print("\n[A] 특잇값:", S.round(3))

energy = np.cumsum(S**2) / np.sum(S**2)
print("[A] k별 예측값과 관측 오차")
for k in range(1, 6):
    P = svd_predict(R, k)
    # 관측된 칸에서만 오차 측정 (00-03의 잔차 개념 🔗)
    rmse = np.sqrt(np.mean((P[mask] - R[mask])**2))
    print(f"   k={k}: 에너지 {energy[k-1]:.3f} | [0,2] 예측 {P[0,2]:6.2f}"
          f" | 관측 RMSE {rmse:.4f}")

# ══════ 버전 B) 개선 — 사용자별 평균으로 채우고 중심화 ══════
# 아이디어: "평가 안 함"을 0(=최악)이 아니라 '그 사람의 평균'으로 보자
user_mean = np.array([R[i][mask[i]].mean() for i in range(len(R))])
print("\n[B] 사용자별 평균 평점:", user_mean.round(2))

R_filled = R.copy()
for i in range(len(R)):
    R_filled[i][~mask[i]] = user_mean[i]      # ~는 부정(NOT) 연산자

# 3단계 PCA처럼 '중심화' — 사용자별 후함/짬을 제거 (00-05의 발상 🔗)
R_centered = R_filled - user_mean[:, None]   # [:, None]로 열벡터 만들어 브로드캐스트

for k in [1, 2, 3]:
    P = svd_predict(R_centered, k) + user_mean[:, None]   # 평균 복원!
    rmse = np.sqrt(np.mean((P[mask] - R[mask])**2))
    print(f"[B] k={k}: [0,2] 예측 {P[0,2]:.2f} | 관측 RMSE {rmse:.4f}")

# ══════ 최종 추천 결과 (개선 버전, k=2) ══════
P_final = svd_predict(R_centered, 2) + user_mean[:, None]
print("\n[B] 미평가 칸의 예측 평점 (k=2)")
for i, j in zip(*np.where(~mask)):          # 0이었던 위치만 순회
    print(f"   사용자{i} → 영화{j}: {P_final[i, j]:.2f}점")


#======================================================================
# [2] 00-06-E · 5단계 — 미니 신경망으로 XOR 학습
#     ReLU·softmax·교차 엔트로피·역전파로 XOR을 학습하고, gW2 를 수치미분과 대조합니다 (seed 고정).
#======================================================================
print("\n" + "=" * 60)
print("[2] 00-06-E · 5단계 — 미니 신경망으로 XOR 학습")
print("=" * 60)
import numpy as np

class MiniNeuralNet:
    """
    00-01 ~ 00-05의 내용만으로 만든 2층 신경망입니다.

    비유:
    원료(입력)를 받아 공정(행렬 곱셈)을 거쳐 제품(예측)을 만들고,
    품질 검사(손실)에서 불량이 나오면 그 결과를 라인 앞쪽으로
    되돌려 설정을 조금씩 고치는(역전파) 작은 생산 라인입니다.
    """

    def __init__(self, n_input, n_hidden, n_output, seed=0):
        rng = np.random.default_rng(seed)
        # 00-02: 가중치는 행렬, 00-05: 작은 정규분포에서 초기화
        # ⚠️ 스케일이 크면 초반부터 기울기가 불안정해집니다
        self.W1 = rng.standard_normal((n_input, n_hidden)) * 0.5
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.standard_normal((n_hidden, n_output)) * 0.5
        self.b2 = np.zeros(n_output)

    def softmax(self, x):
        """00-05: 최댓값 빼기로 오버플로 방지 (결과는 불변임을 증명했음)"""
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def forward(self, x):
        """순전파: 입력 → 예측 확률"""
        self.x = x
        self.z1 = x @ self.W1 + self.b1        # 00-02 행렬곱
        self.h1 = np.maximum(0, self.z1)         # 00-04 ReLU
        self.z2 = self.h1 @ self.W2 + self.b2  # 00-02 행렬곱
        self.y_hat = self.softmax(self.z2)     # 00-05 확률화
        return self.y_hat

    def loss(self, y):
        """00-05: 교차 엔트로피 L = -Σ y log(ŷ)"""
        return -np.sum(y * np.log(self.y_hat + 1e-15))

    def backward(self, y):
        """
        역전파: 00-04의 연쇄법칙으로 기울기 계산.
        기울기 변수는 지시서 규칙대로 g 접두사를 씁니다.
        """
        # 🎁 00-05에서 유도한 선물: softmax + CE의 기울기 = ŷ - y
        gz2 = self.y_hat - y

        gW2 = np.outer(self.h1, gz2)     # 00-01 외적 → (은닉, 출력) 행렬
        gb2 = gz2

        # 00-04 연쇄법칙: 뒤층 기울기를 W2로 되돌리고 ReLU 문지기 통과
        gz1 = (self.W2 @ gz2) * (self.z1 > 0)

        gW1 = np.outer(self.x, gz1)
        gb1 = gz1
        return gW1, gb1, gW2, gb2

    def update(self, grads, lr=0.3):
        """00-04 경사하강법: W ← W - η·∂L/∂W"""
        gW1, gb1, gW2, gb2 = grads
        self.W1 -= lr * gW1
        self.b1 -= lr * gb1
        self.W2 -= lr * gW2
        self.b2 -= lr * gb2

# ══════ XOR 데이터 (직선 하나로는 못 나누는 문제!) ══════
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=float)  # 원-핫

net = MiniNeuralNet(n_input=2, n_hidden=8, n_output=2, seed=0)

print("=== 학습 시작 (은닉 8개) ===")
for epoch in range(300):
    total = 0.0
    for x, y in zip(X, Y):
        net.forward(x)                 # ① 순전파
        total += net.loss(y)           # ② 손실
        net.update(net.backward(y), lr=0.3)   # ③ 역전파+갱신
    if epoch % 50 == 0 or epoch == 299:
        print(f"  {epoch:3d} epoch: 평균 손실 = {total/4:.4f}")

print("\n=== 최종 예측 ===")
for x, y in zip(X, Y):
    p = net.forward(x)
    mark = "✅" if np.argmax(p) == np.argmax(y) else "❌"
    print(f"  입력{x} → 예측{np.argmax(p)} (정답{np.argmax(y)}) 확률{p.round(3)} {mark}")

# ══════ 🔬 00-04의 수치미분으로 역전파 검증 ══════
print("\n=== 역전파 기울기 검증 (수치미분과 대조) ===")
check = MiniNeuralNet(2, 4, 2, seed=1)
x, y = X[1], Y[1]
check.forward(x)
gW2_formula = check.backward(y)[2]        # 공식으로 구한 gW2

h = 1e-5
gW2_numeric = np.zeros_like(check.W2)
for i in range(check.W2.shape[0]):
    for j in range(check.W2.shape[1]):
        tmp = check.W2[i, j]
        check.W2[i, j] = tmp + h; check.forward(x); l_plus = check.loss(y)
        check.W2[i, j] = tmp - h; check.forward(x); l_minus = check.loss(y)
        gW2_numeric[i, j] = (l_plus - l_minus) / (2 * h)   # 00-04 중심차분
        check.W2[i, j] = tmp        # ★ 원상복구 필수!

print("  공식 gW2 첫 행:", gW2_formula[0].round(6))
print("  수치 gW2 첫 행:", gW2_numeric[0].round(6))
print(f"  최대 오차: {np.abs(gW2_formula - gW2_numeric).max():.2e}")
