# -*- coding: utf-8 -*-
"""
03-04-06 신경망 비용 함수와 역전파 — (Cost Function & Back-Propagation) — 사고 현장에서 블랙박스를 역재생하며 책임을 추적하는 탐정

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-04장 - 정규화와 신경망 기초/03-04-06 신경망 비용 함수와 역전파.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 구현 원리
import numpy as np

np.random.seed(1)                          # 난수 고정 → 재현 가능

def sigmoid(z):
    """활성화 함수 : 무한한 값을 0~1로 눌러 주는 품질 게이트."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def add_bias(A):
    """맨 앞에 1로 채운 열(bias unit)을 붙인다."""
    return np.hstack([np.ones((A.shape[0], 1)), A])

# ==================================================================
# 1. 순전파 — 중간값을 '전부' 남긴다 (역전파에서 쓸 블랙박스 녹화)
# ==================================================================
def forward_pass(X, Theta1, Theta2):
    """
    입력을 신경망에 통과시키고 모든 중간 기록을 반환한다.

    비유:
    블랙박스가 주행 전체를 녹화하는 것.
    사고(오차)가 나면 이 녹화본을 거꾸로 돌려 원인을 찾는다.
    """
    a1 = add_bias(X)                   # (m,2) → (m,3)
    z2 = a1 @ Theta1.T                   # (m,3)@(3,2) = (m,2)
    a2 = add_bias(sigmoid(z2))        # 거르고 → bias 붙이기 → (m,3)
    z3 = a2 @ Theta2.T                   # (m,3)@(3,1) = (m,1)
    a3 = sigmoid(z3)                   # 출력층 (m,1)
    return a1, z2, a2, z3, a3

# ==================================================================
# 2. 비용 함수 (2단원 공식, K=1인 경우)
# ==================================================================
def compute_cost(X, y, Theta1, Theta2, lam=0.0):
    """
    신경망 비용 = 데이터 항 + 정규화 항

    비유:
      데이터 항  = 학생 전원의 시험 채점 (이중 합)
      정규화 항  = 학교 전체 시설 점검   (삼중 합)
    둘은 완전히 다른 것을 센다.
    """
    m = len(y)
    *_, a3 = forward_pass(X, Theta1, Theta2)
    # *_ 는 "앞의 반환값들은 관심 없으니 버린다"는 파이썬 문법(언패킹)

    # ⚠ log(0) = -무한대 방지. 학습이 잘 될수록 터지는 역설적 버그
    eps = 1e-15
    a3 = np.clip(a3, eps, 1 - eps)

    # 데이터 항 (K=1이므로 안쪽 합은 생략됨)
    data_term = -np.mean(y * np.log(a3) + (1 - y) * np.log(1 - a3))

    # 정규화 항 : bias 열(0번 열)은 제외 → [:, 1:]
    reg_term = (lam / (2 * m)) * (np.sum(Theta1[:, 1:] ** 2)
                                  + np.sum(Theta2[:, 1:] ** 2))

    return data_term + reg_term

# ==================================================================
# 3. 역전파 — 이 절의 심장
# ==================================================================
def back_propagation(X, y, Theta1, Theta2, lam=0.0):
    """
    출력에서 시작해 거꾸로 책임(δ)을 배분하고 기울기를 구한다.

    비유:
    불량품에서 출발해 공정을 역순으로 훑는 탐정.
      "완제품이 틀렸다" → "직전 공정이 얼마나 기여했나"
      → "그 앞 공정은..." 거꾸로 한 번만 훑으면 끝.
    """
    m = len(y)
    a1, z2, a2, z3, a3 = forward_pass(X, Theta1, Theta2)

    # --- ① 출력층의 책임 : 그냥 '예측 - 정답' ---
    delta3 = a3 - y                       # (m,1)
    # 로그 손실 미분과 시그모이드 미분이 약분되어 이렇게 깔끔해진다 ⭐

    # --- ② 은닉층의 책임 : 뒤에서 받아 와 앞으로 배분 ---
    #     (Θ⁽²⁾)ᵀδ⁽³⁾ 부분 — 단, bias 행은 빼야 한다 (bias엔 앞선 유닛이 없음)
    back_signal = delta3 @ Theta2[:, 1:]   # (m,1)@(1,2) = (m,2)
    # Theta2[:, 1:] 로 bias 열을 제외 ⭐ 최다 실수 지점

    #     .* a .* (1-a) 부분 — '개선 가능성'을 곱한다
    a2_no_bias = a2[:, 1:]                 # bias 제외한 활성값 (m,2)
    delta2 = back_signal * a2_no_bias * (1 - a2_no_bias)
    # * 는 element-wise 곱 (자리별 곱). @ (행렬곱)이 아님에 주의! ⚠

    # --- ③ 기울기 : "전선에 흐른 신호 × 도착지의 책임" ---
    grad1 = (delta2.T @ a1) / m           # (2,m)@(m,3) = (2,3) — Theta1과 같은 모양
    grad2 = (delta3.T @ a2) / m           # (1,m)@(m,3) = (1,3) — Theta2와 같은 모양

    # --- ④ 정규화 항 추가 (bias 열은 제외) ---
    reg1 = (lam / m) * Theta1
    reg1[:, 0] = 0                       # ⭐ 첫 열(bias)만 0으로 되돌리기
    reg2 = (lam / m) * Theta2
    reg2[:, 0] = 0

    return grad1 + reg1, grad2 + reg2

# ==================================================================
# 4. 데이터 : XNOR (03-04-05와 동일)
# ==================================================================
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[1], [0], [0], [1]], dtype=float)   # (4,1) 세로 벡터

# 가중치를 '랜덤으로' 초기화 — 이번엔 손으로 정하지 않는다!
# ⚠ 0으로 초기화하면 모든 유닛이 똑같이 학습되어 의미가 없다 (대칭성 문제)
Theta1 = np.random.randn(2, 3)          # (2,3) : 은닉 2개 × (입력 2 + bias)
Theta2 = np.random.randn(1, 3)          # (1,3) : 출력 1개 × (은닉 2 + bias)

# ==================================================================
# 5. ⭐ 기울기 검산 (gradient checking) — 절대 건너뛰지 말 것
# ==================================================================
def numerical_gradient(X, y, Theta1, Theta2, target="Theta1", step=1e-6):
    """
    정의대로 아주 조금 흔들어 보며 기울기를 구한다.

    비유:
    공식으로 푼 답을 손으로 한 번 더 검산하는 것.
    느리지만 확실하다. (실전 학습에는 쓰지 않고 '검증용'으로만)
    """
    T = Theta1 if target == "Theta1" else Theta2
    num = np.zeros_like(T)

    # ndindex = 다차원 배열의 모든 (행,열) 좌표를 순서대로 만들어 주는 함수
    for idx in np.ndindex(T.shape):
        Tp, Tm = T.copy(), T.copy()      # copy() 필수! 안 하면 원본이 바뀐다
        Tp[idx] += step
        Tm[idx] -= step
        if target == "Theta1":
            cp = compute_cost(X, y, Tp, Theta2)
            cm = compute_cost(X, y, Tm, Theta2)
        else:
            cp = compute_cost(X, y, Theta1, Tp)
            cm = compute_cost(X, y, Theta1, Tm)
        num[idx] = (cp - cm) / (2 * step)   # 중앙차분

    return num

g1, g2 = back_propagation(X, y, Theta1, Theta2)
n1 = numerical_gradient(X, y, Theta1, Theta2, "Theta1")

print("[기울기 검산]")
print("  역전파 :", np.round(g1[0], 8))
print("  수치미분:", np.round(n1[0], 8))
max_err = np.abs(g1 - n1).max()
print(f"  최대 오차: {max_err:.2e} →",
      "✅ 통과" if max_err < 1e-7 else "❌ 유도 재확인")

# ==================================================================
# 6. 학습 — 이제 기계가 스스로 XNOR을 배운다
# ==================================================================
alpha = 3.0                                # 학습률 (데이터가 4개뿐이라 크게 잡아도 안전)

for i in range(1, 20001):
    g1, g2 = back_propagation(X, y, Theta1, Theta2)  # ① 기울기 전부 계산
    Theta1 -= alpha * g1                                # ② 동시 갱신
    Theta2 -= alpha * g2

    if i in (1, 1000, 20000):
        print(f"  iter {i:>5} | J = {compute_cost(X, y, Theta1, Theta2):.6f}")

*_, a3 = forward_pass(X, Theta1, Theta2)
print("\n최종 출력 :", np.round(a3.ravel(), 4))
print("정답      :", y.ravel().astype(int))
print("예측      :", (a3.ravel() >= 0.5).astype(int))


# %% [Block 2] 📝 해설
import numpy as np

def back_propagation_general(X, y, thetas, lam=0.0):
    """
    층 개수가 몇 개든 역전파를 수행한다.

    비유:
    공정이 3개든 100개든, 불량품에서 출발해
    거꾸로 한 번만 훑으면 모든 공정의 책임이 나온다.
    """
    m = len(y)
    L = len(thetas) + 1                     # 전체 층 개수

    # --- ① 순전파 : 모든 활성값 저장 ---
    activations = [add_bias(X)]              # a⁽¹⁾ (bias 포함)
    for i, T in enumerate(thetas):
        z = activations[-1] @ T.T            # [-1] = 리스트의 마지막 원소
        a = sigmoid(z)
        if i < len(thetas) - 1:            # 마지막 층이 아니면 bias 추가
            a = add_bias(a)
        activations.append(a)

    # --- ② 출력층 δ ---
    delta = activations[-1] - y               # δ⁽ᴸ⁾ = a⁽ᴸ⁾ - y

    # --- ③ 거꾸로 훑으며 δ와 기울기 계산 ---
    grads = [None] * len(thetas)            # 결과를 담을 자리 미리 준비

    # reversed(range(n)) = n-1, n-2, ..., 0 (역순 반복)
    for l in reversed(range(len(thetas))):
        a_prev = activations[l]                # 이 가중치의 '입력' 쪽 활성값

        # 기울기 = (도착지 책임)ᵀ × (출발지 신호)
        g = (delta.T @ a_prev) / m

        # 정규화 추가 (bias 열 제외)
        if lam > 0:
            reg = (lam / m) * thetas[l]
            reg[:, 0] = 0
            g = g + reg
        grads[l] = g

        # 다음(앞쪽) 층의 δ 계산 — 첫 가중치까지 왔으면 멈춘다
        if l > 0:
            back = delta @ thetas[l][:, 1:]   # bias 열 제외 ⭐
            a_no_bias = activations[l][:, 1:]
            delta = back * a_no_bias * (1 - a_no_bias)   # * 는 자리별 곱

    return grads
