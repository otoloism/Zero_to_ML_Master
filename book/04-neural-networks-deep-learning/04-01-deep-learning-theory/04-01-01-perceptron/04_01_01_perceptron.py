# -*- coding: utf-8 -*-
"""
04-01-01 퍼셉트론— 신경망의 가장 단순한 조상

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-01장 딥러닝 이론과 구현/04-01-01 퍼셉트론— 신경망의 가장 단순한 조상.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-01-01-B AND 게이트 — "둘 다 1이어야 1"
import numpy as np

def AND(x1, x2):
    """AND 게이트: 둘 다 1이어야 1.
    비유: '두 사람이 동시에 OK해야 통과하는 보안문'"""
    x = np.array([x1, x2])            # 입력을 배열로 변환
    w = np.array([0.5, 0.5])           # 가중치: 두 입력을 동등하게
    b = -0.7                            # 편향: 둘 다 1이어야 넘는 높은 문턱
    tmp = np.sum(w * x) + b            # 02권의 내적: w·x + b
    return 1 if tmp > 0 else 0          # 계단함수: 양수→1, 그외→0

# ── 모든 입력 조합 테스트 ──
print("[AND 게이트]")
for x1, x2 in [(0,0), (0,1), (1,0), (1,1)]:
    print(f"  AND({x1}, {x2}) = {AND(x1, x2)}")


# %% [Block 2] 04-01-01-C NAND & OR — "가중치만 바꾸면 다른 게이트"
def NAND(x1, x2):
    """NAND: AND의 반대. 둘 다 1일 때만 0.
    비유: '둘 다 동시에 누르면 막히는 비상문'"""
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])        # AND의 부호를 뒤집음!
    b = 0.7                              # 편향도 반대
    return 1 if np.sum(w * x) + b > 0 else 0

def OR(x1, x2):
    """OR: 하나만 1이어도 1.
    비유: '한 사람만 OK해도 열리는 자동문'"""
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2                            # AND보다 문턱이 낮음!
    return 1 if np.sum(w * x) + b > 0 else 0

# ── 3개 게이트 비교 ──
print("x1 x2 | AND  NAND  OR")
print("───────┼────────────────")
for x1, x2 in [(0,0), (0,1), (1,0), (1,1)]:
    print(f" {x1}  {x2}  |  {AND(x1,x2)}     {NAND(x1,x2)}    {OR(x1,x2)}")


# %% [Block 3] 04-01-01-E 2층 퍼셉트론으로 XOR 해결 — "층을 쌓으면 된다!"
def XOR(x1, x2):
    """XOR = 2층 퍼셉트론: NAND + OR → AND.
    비유: '종이를 한 번 접고(1층) 가위로 자르기(2층)'"""
    s1 = NAND(x1, x2)                  # 1층-① NAND
    s2 = OR(x1, x2)                    # 1층-② OR
    y = AND(s1, s2)                    # 2층: AND → 최종 출력
    return y

# ── 검증: 모든 입력 + 중간 과정까지 출력 ──
print("x1 x2 | NAND  OR  → AND = XOR")
print("───────┼─────────────────────")
for x1, x2 in [(0,0), (0,1), (1,0), (1,1)]:
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    print(f" {x1}  {x2}  |   {s1}    {s2}       {y}     {y}")
