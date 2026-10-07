# -*- coding: utf-8 -*-
"""
00-06SVD·PCA— 300페이지 책을 3페이지로

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 00권  비전공자 처음부터 — 수포자 훑어보기/00-06SVD·PCA— 300페이지 책을 3페이지로.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 📍 00-06-A · 1단계 — 고유벡터와 고유값: 방향이 변하지 않는 특별한 화살표
import numpy as np

A = np.array([[4., 1.],
              [2., 3.]])

# ══════ ① 라이브러리로 고유값·고유벡터 구하기 ══════
# eig()는 (고유값 배열, 고유벡터 행렬)을 반환합니다.
# ⚠️ 고유벡터 행렬은 '각 열'이 하나의 고유벡터입니다 (행이 아님!)
eigenvalues, eigenvectors = np.linalg.eig(A)

print("고유값:", eigenvalues)
print("고유벡터(열 단위 저장):\n", eigenvectors.round(6))

# ══════ ② 정의 Av = λv 가 성립하는지 검증 ══════
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]        # i번째 '열' = i번째 고유벡터
    lam = eigenvalues[i]
    left = A @ v                  # Av  (00-02 행렬×벡터)
    right = lam * v                # λv  (00-01 스칼라배)
    ok = np.allclose(left, right) # 부동소수점이라 '거의 같은지'로 비교
    print(f"  λ={lam:.1f}: Av={left.round(4)}, λv={right.round(4)} → {ok}")

# ══════ ③ 특성방정식 det(A - λI) = 0 을 직접 확인 ══════
# 손으로 푼 λ=5, 2 를 넣으면 행렬식이 0이 되어야 합니다
I = np.eye(2)                      # 2×2 단위행렬 (00-02)
for lam in [5.0, 2.0, 3.0]:  # 3은 고유값이 아님 → 0이 안 나와야 정상
    det = np.linalg.det(A - lam * I)
    tag = "고유값 ✅" if abs(det) < 1e-9 else "고유값 아님 ❌"
    print(f"  det(A - {lam}I) = {det:+.6f}  → {tag}")

# ══════ ④ 고유벡터는 '방향'만 같으면 된다 ══════
v_hand = np.array([1., 1.])           # 손으로 구한 것
v_lib = eigenvectors[:, 0]           # 라이브러리 결과 (길이 1로 정규화됨)
print(f"\n손계산 (1,1)을 00-01의 정규화 적용: {(v_hand/np.linalg.norm(v_hand)).round(6)}")
print(f"라이브러리 결과              : {v_lib.round(6)}")


# %% [Block 2] 📍 00-06-B · 2단계 — SVD: 모든 행렬을 분해하는 만능 도구
import numpy as np
np.set_printoptions(suppress=True)   # 1e-16 같은 지수 표기 끄기

# 각 행이 '앞 행 + 1'인 규칙적인 행렬 → 사실상 패턴이 2개뿐
A = np.array([[1., 2., 3., 4.],
              [2., 3., 4., 5.],
              [3., 4., 5., 6.],
              [4., 5., 6., 7.]])

# ══════ ① SVD 분해 ══════
# ⚠️ S는 '1차원 배열'로 반환됩니다 (대각행렬 아님!)
U, S, Vt = np.linalg.svd(A)

print("특잇값 S :", S.round(4))
print("행렬 랭크 :", np.linalg.matrix_rank(A), "← 0이 아닌 특잇값의 개수")

# ══════ ② 에너지 비율로 k 정하기 ══════
# energy(k) = (상위 k개 σ²의 합) / (전체 σ²의 합)
energy = np.cumsum(S**2) / np.sum(S**2)   # cumsum: 누적 합
print("\n--- k별 에너지 비율과 오차 ---")
for k in [1, 2, 3]:
    # ★핵심 한 줄★ 수식 A_k = U[:,:k] Σ_k Vt[:k,:] 그대로
    A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

    # 00-01의 노름을 행렬로 확장한 '프로베니우스 노름'으로 오차 측정
    err = np.linalg.norm(A - A_k)            # 기본값이 'fro'

    # 이론값 √(σ_{k+1}² + ...) 과 비교 (에카르트-영 정리)
    theory = np.sqrt(np.sum(S[k:]**2))
    print(f"  k={k}: 에너지 {energy[k-1]:.4f} | 실제오차 {err:.6f}"
          f" | 이론오차 {theory:.6f}")

# ══════ ③ k=1과 k=2 근사 행렬 비교 ══════
A_1 = U[:, :1] @ np.diag(S[:1]) @ Vt[:1, :]
A_2 = U[:, :2] @ np.diag(S[:2]) @ Vt[:2, :]
print("\nk=1 근사 (패턴 1개만):\n", A_1.round(2))
print("\nk=2 근사 (패턴 2개):\n", A_2.round(2))
print("\n원본과 k=2가 같은가?", np.allclose(A, A_2))

# ══════ ④ '층의 합' 형태로도 같은지 확인 ══════
# A_k = Σ σ_i · u_i v_iᵀ  (외적의 가중합)
A_layers = sum(S[i] * np.outer(U[:, i], Vt[i, :]) for i in range(2))
print("층의 합 방식과 슬라이싱 방식이 같은가?", np.allclose(A_2, A_layers))


# %% [Block 3] 📍 00-06-C · 3단계 — PCA: 데이터의 가장 중요한 방향 찾기
import numpy as np
np.random.seed(42)

# ══════ 데이터 생성: 키와 상관된 몸무게 (00-05 정규분포 활용) ══════
n = 100
height = np.random.normal(170, 8, n)                    # N(170, 8²)
weight = 0.8 * height - 60 + np.random.normal(0, 3, n)  # 키에 노이즈 추가
X = np.column_stack([height, weight])                    # (100, 2)

# ══════ 경로 A) 공분산 행렬 → 고유값 분해 ══════
# ① 중심화 — ⚠️ 절대 빠뜨리면 안 되는 단계!
Xc = X - X.mean(axis=0)

# ② 공분산 행렬 C = XcᵀXc/(n-1)  ← 00-03의 AᵀA와 같은 형태
C = np.cov(Xc.T)                     # ⚠️ 전치 필수! (변수가 행이 되어야 함)
print("공분산 행렬:\n", C.round(4))

# ③ 고유값 분해 — 대칭행렬이므로 eigh (1단계에서 배운 규칙!)
eigenvalues, eigenvectors = np.linalg.eigh(C)

# ④ eigh는 '오름차순'으로 주므로 내림차순으로 뒤집기
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]      # 열 단위로 재정렬

ratio = eigenvalues / eigenvalues.sum()      # 설명 분산 비율
print(f"\n고유값 : {eigenvalues.round(4)}")
print(f"설명비율: PC1 {ratio[0]:.1%} / PC2 {ratio[1]:.1%}")
print(f"PC1 방향: {eigenvectors[:, 0].round(4)}")

# ══════ 경로 B) SVD로 바로 (수식 ⑤ 검증!) ══════
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
lambda_from_svd = S**2 / (n - 1)          # λ = σ²/(n-1)

print(f"\n[검증] 공분산 경로 고유값: {eigenvalues.round(4)}")
print(f"[검증] SVD 경로  σ²/(n-1): {lambda_from_svd.round(4)}")
print(f"[검증] 두 방법이 일치하는가? {np.allclose(eigenvalues, lambda_from_svd)}")

# ══════ 차원 축소: 2D → 1D 투영 후 복원 ══════
k = 1
V_k = eigenvectors[:, :k]            # PC1만 사용 (2×1)
Z = Xc @ V_k                         # 투영: (100,2)@(2,1) → (100,1)
X_restored = Z @ V_k.T + X.mean(axis=0)   # 복원 (평균 다시 더하기!)

recon_error = np.linalg.norm(X - X_restored)
theory_error = np.sqrt((n - 1) * eigenvalues[1])   # 버린 고유값으로 예측
print(f"\n2D→1D 압축 후 복원 오차: {recon_error:.4f}")
print(f"이론 예측 √((n-1)·λ₂)  : {theory_error:.4f}")


# %% [Block 4] 📍 00-06-D · 4단계 — 추천 시스템: SVD의 실전 활용
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


# %% [Block 5] 📍 00-06-E · 5단계 — 종합: 00-01부터 00-06까지 하나로 잇기
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
