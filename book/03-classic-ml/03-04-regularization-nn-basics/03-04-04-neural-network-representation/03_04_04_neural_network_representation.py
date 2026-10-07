# -*- coding: utf-8 -*-
"""
03-04-04 신경망의 등장과 모델 표현 — (Neural Networks & Model Representation) — 특성 공학을 사람이 아니라 기계에게 맡기다

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-04장 - 정규화와 신경망 기초/03-04-04 신경망의 등장과 모델 표현.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 구현 원리
import numpy as np

np.random.seed(42)                         # 난수 고정 → 매번 같은 결과

def sigmoid(z):
    """
    활성화 함수. 무한한 점수를 0~1로 눌러 준다.

    비유:
    공장의 '품질 게이트'.
    아무리 큰 값이 들어와도 0~1 규격으로 맞춰서 내보낸다.
    """
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    # clip으로 범위를 잘라 exp 폭발(overflow)을 막는다

def add_bias(A):
    """
    행렬 맨 앞에 1로 채운 열(bias unit)을 붙인다.

    비유:
    택배 상자에 '기본 포장재'를 하나씩 넣어 주는 것.
    내용물이 뭐든 이건 항상 들어간다.
    """
    ones = np.ones((A.shape[0], 1))              # (샘플수, 1) 크기의 1 배열
    return np.hstack([ones, A])                   # hstack = 옆으로 붙이기

# ==================================================================
# 1. 신경망 구조 정하기 : 3 → 3 → 1
#    s1=3(입력), s2=3(은닉), s3=1(출력)
# ==================================================================
s1, s2, s3 = 3, 3, 1

# Θ⁽ʲ⁾ 의 크기는 s_{j+1} × (s_j + 1)
#   행(세로) = 받을 사람 수, 열(가로) = 보낼 사람 수 + 1(bias)
Theta1 = np.random.randn(s2, s1 + 1)          # (3, 4)
Theta2 = np.random.randn(s3, s2 + 1)          # (1, 4)

print("Theta1 shape :", Theta1.shape, "= s2 × (s1+1) =", (s2, s1 + 1))
print("Theta2 shape :", Theta2.shape, "= s3 × (s2+1) =", (s3, s2 + 1))

# ==================================================================
# 2. 순전파 — 섞고(Θ 곱하기) → 거르고(g 씌우기)의 반복
# ==================================================================
def forward_pass(X, Theta1, Theta2):
    """
    입력 X를 받아 신경망을 통과시킨다.

    비유:
    공장의 컨베이어 벨트.
      원재료 → [1공정: 섞기] → [검수] → [2공정: 섞기] → [검수] → 완제품

    반환값을 전부 돌려주는 이유:
    다음 절의 역전파에서 '중간 기록'이 전부 필요하기 때문이다.
    (블랙박스가 주행 과정을 다 녹화해 두는 것과 같다)
    """
    # --- Layer 1 (입력층) ---
    a1 = add_bias(X)                    # (m, 3) → (m, 4)

    # --- Layer 1 → 2 ---
    z2 = a1 @ Theta1.T                    # (m,4) @ (4,3) = (m,3)
    # .T(전치)를 쓰는 이유: Theta1은 (3,4)라 그대로 곱하면 shape가 안 맞는다
    # 수식 z=Θa 는 '세로 벡터' 기준, 코드는 '가로로 눕힌 샘플' 기준이라 전치가 필요

    a2 = add_bias(sigmoid(z2))         # 거르고 → bias 붙이기 → (m, 4)
    # ⚠ 여기서 add_bias를 빼먹는 것이 최다 실수!

    # --- Layer 2 → 3 ---
    z3 = a2 @ Theta2.T                    # (m,4) @ (4,1) = (m,1)
    a3 = sigmoid(z3)                    # 출력층은 bias를 붙이지 않는다

    return a1, z2, a2, z3, a3            # 튜플로 한꺼번에 반환

# ==================================================================
# 3. 샘플 4개를 한꺼번에 통과시키기 (반복문 없이!)
# ==================================================================
X = np.array([
    [0.5, -1.0,  2.0],
    [1.5,  0.3, -0.7],
    [-0.8,  1.2,  0.4],
    [2.0,  2.0,  2.0],
])

a1, z2, a2, z3, a3 = forward_pass(X, Theta1, Theta2)

print("\n[shape 추적]")
print("  X  :", X.shape,  "→ a1 :", a1.shape, "(bias 열 추가됨)")
print("  z2 :", z2.shape, "→ a2 :", a2.shape, "(bias 열 추가됨)")
print("  z3 :", z3.shape, "→ a3 :", a3.shape, "(출력, bias 없음)")

print("\n은닉층 활성값 a⁽²⁾ (bias 제외) :")
print(np.round(a2[:, 1:], 4))          # [:, 1:] = 모든 행, 1번 열부터 (bias 제외)
print("\n최종 출력 hΘ(x) :", np.round(a3.ravel(), 4))
# ravel() = (4,1)을 (4,)로 납작하게 펴기 → 보기 좋게

# ==================================================================
# 4. 검증 : 뉴런 하나를 '수동으로' 계산해 일치하는지 확인
#    (3단원의 "뉴런 = 로지스틱 회귀"를 코드로 증명)
# ==================================================================
manual = sigmoid(np.dot(Theta1[0], a1[0]))   # 첫 샘플, 첫 은닉 유닛
# dot = 내적 (같은 자리끼리 곱해서 전부 더하기) = θᵀx 그 자체
print(f"\n수동 계산 a₁⁽²⁾ = {manual:.6f}")
print(f"벡터화 결과      = {a2[0, 1]:.6f}")
print("→ 일치 :", np.isclose(manual, a2[0, 1]))
# isclose = 부동소수점 오차를 감안해 '거의 같은가'를 판정


# %% [Block 2] 📝 해설
import numpy as np

def forward_any_depth(X, thetas):
    """
    층 개수에 상관없이 순전파를 수행한다.

    비유:
    공정이 3개든 100개든 컨베이어 벨트를 계속 이어 붙이면 된다.
    '섞고 → 거르고'를 층 수만큼 반복할 뿐.

    thetas : [Theta1, Theta2, ...] 가중치 행렬들의 리스트
    """
    activations = []                         # 각 층의 활성값 기록 (역전파용)
    zs = []                                  # 각 층의 z값 기록

    a = add_bias(X)                       # 첫 층: 입력 + bias
    activations.append(a)

    # enumerate = 인덱스와 값을 함께 꺼내 주는 함수
    for i, Theta in enumerate(thetas):
        z = a @ Theta.T                      # 섞기
        zs.append(z)

        a_new = sigmoid(z)                  # 거르기

        is_last = (i == len(thetas) - 1)     # 마지막 층인가?
        if not is_last:
            a_new = add_bias(a_new)          # ⭐ 마지막 층에는 bias를 붙이지 않는다

        activations.append(a_new)
        a = a_new                            # 다음 층의 입력으로 넘긴다

    return a, activations, zs               # 최종 출력, 중간 기록들
