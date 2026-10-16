# -*- coding: utf-8 -*-
"""
Zero to 머신러닝 딥러닝 Master — Medium 시리즈 #23 · 2부
경사하강법의 함정 — 학습률·국소 최솟값·신경망 학습이 같은 원리인 이유 (2부)

원문(책): 01-10 「편미분·그래디언트·경사하강법 (2부)」 https://wikidocs.net/439754
Medium : TODO (게시 후 추가)
책     : https://wikidocs.net/book/21464
홈페이지: https://mldict.net
블로그 : https://wikidocs.net/blog/@mldict/

실행: pip install -r requirements.txt && python 23_2_learning_rate_local_minima.py
※ 아래 코드는 책 본문의 코드 블록을 순서대로 옮긴 것입니다. [추가]/[수정] 표시가 있는 부분만 실행을 위해 보탰습니다.
"""

#======================================================================
# [참고] 01-10-H 8단계 — PyTorch 학습 루프의 세 줄
#     설명용 예시 코드(데이터 미정의) — 실행하지 않으므로 주석으로만 남깁니다.
#======================================================================
# optimizer.zero_grad()   # 지난 걸음의 기울기 기록을 지운다 (grad는 누적되므로!)
# loss.backward()       # ⬅️ 역전파: 100만 개 편미분을 한 번에 계산 → ∇L 완성
# optimizer.step()        # 👣 w ← w - η∇L : 5단계에서 직접 만든 그 한 줄!


#======================================================================
# [1] 01-10-F 6단계 — 학습률 5종 비교 실험
#     같은 시작점 (4, 3)에서 학습률만 바꿔 30걸음 후 위치와 손실을 비교합니다.
#     ⚠️ 이 코드는 1부(5단계)의 compute_loss·compute_gradient 를 사용하므로, 그 정의를 앞에 추가했습니다(표시된 부분).
#======================================================================
print("\n" + "=" * 60)
print("[1] 01-10-F 6단계 — 학습률 5종 비교 실험")
print("=" * 60)
# [추가 코드 — 책 본문에는 없음] 1부(01-10-E 5단계)에서 정의한 지형 함수와 그래디언트
import numpy as np

def compute_loss(position):
    """지형의 높이(손실값)를 계산합니다."""
    x, y = position
    return x ** 2 + 2 * y ** 2

def compute_gradient(position):
    """∇f = (∂f/∂x, ∂f/∂y) = (2x, 4y)"""
    x, y = position
    return np.array([2 * x, 4 * y])

for lr in [0.01, 0.1, 0.25, 0.5, 0.9]:
    position = np.array([4.0, 3.0])
    for step in range(30):
        position = position - lr * compute_gradient(position)
    print(f"lr={lr}: x={position[0]:.6g}, y={position[1]:.6g}, "
          f"f={compute_loss(position):.6g}")


#======================================================================
# [2] 01-10-G 7단계 — 시작점에 따라 갈리는 운명 (지역 최솟값)
#     같은 알고리즘·같은 학습률인데 시작점만 다르게 해서 전역 최솟값과 지역 최솟값에 각각 도착하는 것을 확인합니다.
#======================================================================
print("\n" + "=" * 60)
print("[2] 01-10-G 7단계 — 시작점에 따라 갈리는 운명 (지역 최솟값)")
print("=" * 60)
def f(x):
    """웅덩이가 두 개인 지형 (왼쪽이 진짜 바닥)"""
    return x**4 - 4*x**2 + 0.5*x

def df(x):
    """도함수: 4x³ - 8x + 0.5"""
    return 4*x**3 - 8*x + 0.5

for start in [-2.0, -0.5, 0.5, 2.0]:      # 시작점만 다르게!
    x = start
    for _ in range(200):
        x = x - 0.01 * df(x)                 # 완전히 동일한 알고리즘
    print(f"start={start:>5} -> x={x:.4f}, f={f(x):.4f}")


#======================================================================
# [3] 연습문제 — f(x, y) = a·x² + b·y² 의 그래디언트
#     계수 a, b 를 받는 일반화된 compute_gradient 입니다.
#======================================================================
print("\n" + "=" * 60)
print("[3] 연습문제 — f(x, y) = a·x² + b·y² 의 그래디언트")
print("=" * 60)
def compute_gradient(position, a, b):
    """f(x,y) = a·x² + b·y² 의 그래디언트"""
    x, y = position
    return np.array([2 * a * x, 2 * b * y])


#======================================================================
# [4] 연습문제 — 수치 그래디언트
#     각 방향으로 ±h 만큼 밀어 보며 기울기를 재는 '실측' 방식입니다.
#     ⚠️ 연습문제 해설 코드는 함수 정의만 있어, 실습 1(점 (2, 1))로 공식과 수치 그래디언트를 비교하는 코드를 추가했습니다(표시된 부분).
#======================================================================
print("\n" + "=" * 60)
print("[4] 연습문제 — 수치 그래디언트")
print("=" * 60)
def numerical_gradient(f, position, h=1e-5):
    """각 방향으로 살짝 밀어보며 기울기를 재는 '실측' 방식"""
    grad = np.zeros_like(position)
    for i in range(len(position)):
        p_plus, p_minus = position.copy(), position.copy()
        p_plus[i] += h
        p_minus[i] -= h
        grad[i] = (f(p_plus) - f(p_minus)) / (2 * h)
    return grad

# [추가 코드 — 책 본문에는 없음] 실습 1 — f(x, y) = 3x² + y² 의 점 (2, 1)에서 공식 vs 수치 그래디언트
f_xy = lambda p: 3 * p[0] ** 2 + p[1] ** 2
pos = np.array([2.0, 1.0])
print("공식 그래디언트 (6x, 2y):", compute_gradient(pos, 3, 1))
print("수치 그래디언트        :", numerical_gradient(f_xy, pos))
