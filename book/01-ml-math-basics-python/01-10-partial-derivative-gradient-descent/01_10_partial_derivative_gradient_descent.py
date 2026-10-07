# -*- coding: utf-8 -*-
"""
01-10🧭편미분· 그래디언트 ·경사하강법

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 01권  수포자들을 위한 머신러닝 수학 완전기초와 파이썬 맛보기/01-10🧭편미분· 그래디언트 ·경사하강법.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 01-10-C 3단계 — 🧩 벽 ③: 미분과 벡터가 한꺼번에 나온다
import numpy as np

def compute_loss(position):
    """
    지형의 높이(손실값)를 계산합니다.

    비유:
    등산객이 현재 서 있는 (x, y) 좌표에서
    고도계를 읽는 것과 같습니다.
    """
    x, y = position
    return x ** 2 + 2 * y ** 2

def compute_gradient(position):
    """
    현재 위치에서의 그래디언트 ∇f = (∂f/∂x, ∂f/∂y)를 계산합니다.

    비유:
    "동쪽으로 한 발짝, 북쪽으로 한 발짝" 갔을 때
    고도가 각각 얼마나 변하는지 재는 것과 같습니다.
    """
    x, y = position
    df_dx = 2 * x   # x 방향 편미분 ∂f/∂x = 2x
    df_dy = 4 * y   # y 방향 편미분 ∂f/∂y = 4y
    return np.array([df_dx, df_dy])

position = np.array([1.0, 1.0])
grad = compute_gradient(position)

print("그래디언트 ∇f =", grad)
print("크기 |∇f|   =", np.linalg.norm(grad))


# %% [Block 2] 01-10-E 5단계 — 🙈 벽 ⑤: 지도를 볼 수 없다는 사실
import numpy as np

def compute_loss(position):
    """지형의 높이(손실값)를 계산합니다."""
    x, y = position
    return x ** 2 + 2 * y ** 2

def compute_gradient(position):
    """∇f = (∂f/∂x, ∂f/∂y) = (2x, 4y)"""
    x, y = position
    return np.array([2 * x, 4 * y])

def gradient_descent(start, learning_rate=0.1, num_steps=15):
    """
    경사하강법을 실행해 최솟값을 찾아갑니다.

    비유:
    안개 속에서 발바닥 감각(그래디언트)만으로
    한 걸음씩(학습률) 내리막을 내려가는 과정입니다.
    """
    position = np.array(start, dtype=float)
    path = [position.copy()]                      # 지나온 발자국 기록

    print(f"{'단계':>4} | {'x':>7} | {'y':>7} | {'f(x,y)':>8} | {'|grad|':>7}")
    for step in range(num_steps):
        grad = compute_gradient(position)     # 1) 발밑 경사 확인
        if step % 3 == 0:                       # 3걸음마다 기록 출력
            print(f"{step:>4} | {position[0]:>7.3f} | {position[1]:>7.3f} "
                  f"| {compute_loss(position):>8.4f} | {np.linalg.norm(grad):>7.4f}")
        position = position - learning_rate * grad # 2) 반대 방향으로 한 걸음 ★핵심
        path.append(position.copy())

    return position, np.array(path)

final_position, path = gradient_descent(start=[4.0, 3.0])
print(f"\n최종 위치: ({final_position[0]:.6f}, {final_position[1]:.6f})")
print(f"최종 손실: {compute_loss(final_position):.6f}")


# %% [Block 3] 01-10-E 5단계 — 🙈 벽 ⑤: 지도를 볼 수 없다는 사실
class GradientDescentOptimizer:
    """
    규칙 하나만 아는 아주 작은 옵티마이저.

    비유:
    "그래디언트의 반대 방향으로, 정해진 보폭만큼만 이동한다"는
    규칙 하나만 기억하는 작은 로봇 🤖 이라고 생각하세요.
    """

    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate    # 보폭 η 저장

    def step(self, position, grad):
        """한 걸음 이동시킵니다: w ← w - η∇f"""
        return position - self.learning_rate * grad

optimizer = GradientDescentOptimizer(learning_rate=0.1)
position = np.array([4.0, 3.0])
for _ in range(15):
    grad = compute_gradient(position)
    position = optimizer.step(position, grad)   # PyTorch의 optimizer.step()과 같은 이름!


# %% [Block 4] 01-10-F 6단계 — 👣 벽 ⑥: 보폭(학습률)의 딜레마
for lr in [0.01, 0.1, 0.25, 0.5, 0.9]:
    position = np.array([4.0, 3.0])
    for step in range(30):
        position = position - lr * compute_gradient(position)
    print(f"lr={lr}: x={position[0]:.6g}, y={position[1]:.6g}, "
          f"f={compute_loss(position):.6g}")


# %% [Block 5] 01-10-G 7단계 — 🕳️ 벽 ⑦: 발밑이 평평하다고 바닥은 아니다
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


# %% [Block 6] 01-10-H 8단계 — 🚀 신경망 학습 = 수백만 차원 경사하강법
optimizer.zero_grad()   # 지난 걸음의 기울기 기록을 지운다 (grad는 누적되므로!)
loss.backward()       # ⬅️ 역전파: 100만 개 편미분을 한 번에 계산 → ∇L 완성
optimizer.step()        # 👣 w ← w - η∇L : 5단계에서 직접 만든 그 한 줄!


# %% [Block 7] 📝 연습문제
def compute_gradient(position, a, b):
    """f(x,y) = a·x² + b·y² 의 그래디언트"""
    x, y = position
    return np.array([2 * a * x, 2 * b * y])


# %% [Block 8] 📝 연습문제
def numerical_gradient(f, position, h=1e-5):
    """각 방향으로 살짝 밀어보며 기울기를 재는 '실측' 방식"""
    grad = np.zeros_like(position)
    for i in range(len(position)):
        p_plus, p_minus = position.copy(), position.copy()
        p_plus[i] += h
        p_minus[i] -= h
        grad[i] = (f(p_plus) - f(p_minus)) / (2 * h)
    return grad
