# -*- coding: utf-8 -*-
"""
03-04-05 신경망으로 논리 게이트 만들기 — (Examples and Intuitions — AND to XNOR) — 직선 하나로 못 자르면, 접어서 두 번 자른다

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-04장 - 정규화와 신경망 기초/03-04-05 신경망으로 논리 게이트 만들기.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 구현 원리
import numpy as np

def sigmoid(z):
    """
    활성화 함수.

    비유:
    아무리 큰 값이 들어와도 0~1 규격으로 맞춰 내보내는 '품질 게이트'.
    z=10이면 0.99995, z=-30이면 사실상 0이 된다.
    """
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def add_bias(A):
    """맨 앞에 1로 채운 열(bias unit)을 붙인다."""
    return np.hstack([np.ones((A.shape[0], 1)), A])
    # hstack = 옆으로 붙이기, shape[0] = 행 개수(샘플 수)

# ==================================================================
# 1. 입력 : 가능한 네 가지 조합을 한 번에
# ==================================================================
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=float)                        # dtype=float : 정수로 두면 계산 중 오차 발생

# ==================================================================
# 2. 단일 뉴런 게이트들 (1·2단원)
#    숫자 세 개가 [bias, x1의 가중치, x2의 가중치]
# ==================================================================
GATES = {
    "AND": np.array([-30.,  20.,  20.]),   # 기준선 -30 → 둘 다 필요
    "NOR": np.array([ 10., -20., -20.]),   # 서명이 반대표 → 아무도 없어야
    "OR" : np.array([-10.,  20.,  20.]),   # 기준선 -10 → 하나면 충분
}
# { } 는 딕셔너리 : '이름 → 값' 형태로 묶어 두는 자료구조

Xb = add_bias(X)                          # (4,2) → (4,3)

print("[단일 뉴런 게이트]")
print("  x1 x2 | AND  NOR   OR")
outputs = {}
for name, theta in GATES.items():         # items() = 키와 값을 함께 꺼내기
    outputs[name] = sigmoid(Xb @ theta)   # @ 행렬곱 → (4,3)@(3,) = (4,)

for i in range(4):
    a, n, o = outputs["AND"][i], outputs["NOR"][i], outputs["OR"][i]
    print(f"   {int(X[i,0])}  {int(X[i,1])} | {a:4.1f} {n:4.1f} {o:4.1f}")
    # {값:4.1f} = 4칸 폭, 소수 1자리 → 표처럼 정렬

# ==================================================================
# 3. XNOR 신경망 (4단원) — 은닉층 2개 + 출력층 1개
# ==================================================================
Theta1 = np.vstack([GATES["AND"], GATES["NOR"]])   # 세로로 쌓기 → (2,3)
Theta2 = GATES["OR"].reshape(1, -1)                # (3,) → (1,3) 2차원으로
# reshape(1,-1) 의 -1 = "나머지는 알아서 계산해"

print("\nTheta1 shape :", Theta1.shape, "= s2 × (s1+1) = 2 × 3")
print("Theta2 shape :", Theta2.shape, "= s3 × (s2+1) = 1 × 3")

def xnor_network(X, Theta1, Theta2):
    """
    2층 신경망으로 XNOR을 계산한다.

    비유:
    직선 한 번으로 못 자르는 종이를,
    은닉층에서 '한 번 접고'(a2 공간으로 변환)
    출력층에서 '직선 한 번'(OR)으로 자른다.
    """
    a1 = add_bias(X)                    # (4,2) → (4,3)
    z2 = a1 @ Theta1.T                    # (4,3)@(3,2) = (4,2)
    # .T(전치)가 필요한 이유: 수식은 세로벡터 기준, 코드는 가로 샘플 기준

    a2 = add_bias(sigmoid(z2))         # 거르고 → bias 붙이기 → (4,3)
    # ⚠ add_bias를 빼먹으면 shape 오류가 난다 (최다 실수)

    z3 = a2 @ Theta2.T                    # (4,3)@(3,1) = (4,1)
    a3 = sigmoid(z3)                    # 출력층은 bias를 붙이지 않는다
    return a2[:, 1:], a3.ravel()        # 은닉 활성값(bias 제외), 최종 출력
    # ravel() = (4,1)을 (4,)로 납작하게 펴기

hidden, out = xnor_network(X, Theta1, Theta2)

# ==================================================================
# 4. 진리표 자동 검증
# ==================================================================
expected = np.array([1, 0, 0, 1])          # XNOR의 정답
predicted = (out >= 0.5).astype(int)      # True→1, False→0

print("\n[XNOR 신경망]")
print("  x1 x2 | a1(AND) a2(NOR) |  출력  | 예측 | 정답")
for i in range(4):
    print(f"   {int(X[i,0])}  {int(X[i,1])} |"
          f"  {hidden[i,0]:5.2f}   {hidden[i,1]:5.2f} |"
          f" {out[i]:6.4f} |  {predicted[i]}   |  {expected[i]}")

print("\n검증 결과 :",
      "✅ 완벽히 일치" if np.array_equal(predicted, expected) else "❌ 불일치")
# array_equal = 두 배열이 완전히 같은지 판정
# 'A if 조건 else B' = 한 줄 if문 (조건부 표현식)


# %% [Block 2] 📝 해설
import numpy as np

def print_truth_table(name, theta):
    """
    단일 뉴런 게이트의 진리표를 출력한다.

    비유:
    새로 만든 결재 규정을 네 가지 상황에 다 넣어 보고
    '이 규정이 의도대로 동작하는가' 확인하는 리허설.
    """
    theta = np.asarray(theta, dtype=float)
    combos = [(0, 0), (0, 1), (1, 0), (1, 1)]

    print(f"\n[{name}]  θ = {theta}")
    print("  x1 x2 |     z    |   g(z)   | 결과")
    print("  " + "-" * 38)

    for x1, x2 in combos:
        # x0=1 을 앞에 붙여 내적 계산 (= θᵀx)
        z = float(np.dot(theta, [1, x1, x2]))
        h = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        print(f"   {x1}  {x2} | {z:+8.1f} | {h:8.5f} |  {int(h >= 0.5)}")
        # {z:+8.1f} = 부호 항상 표시, 8칸 폭, 소수 1자리

# 사용 예
print_truth_table("AND", [-30, 20, 20])
print_truth_table("NOR", [10, -20, -20])
