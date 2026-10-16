# -*- coding: utf-8 -*-
"""
Zero to 머신러닝 딥러닝 Master — Medium 시리즈 #23 · 1부
편미분·그래디언트 쉽게 이해하기 — 안개 산에서 발밑 경사만으로 하산하기 (1부)

원문(책): 01-10 「편미분·그래디언트·경사하강법 (1부)」 https://wikidocs.net/439754
Medium : TODO (게시 후 추가)
책     : https://wikidocs.net/book/21464
홈페이지: https://mldict.net
블로그 : https://wikidocs.net/blog/@mldict/

실행: pip install -r requirements.txt && python 23_1_partial_derivative_gradient.py
※ 아래 코드는 책 본문의 코드 블록을 순서대로 옮긴 것입니다. [추가]/[수정] 표시가 있는 부분만 실행을 위해 보탰습니다.
"""

#======================================================================
# [1] 01-10-C 3단계 — 손실과 그래디언트를 코드로 옮기기
#     compute_loss 와 compute_gradient 를 정의하고 점 (1, 1)에서 ∇f = [2, 4], 크기 √20 ≈ 4.472 를 확인합니다.
#======================================================================
print("\n" + "=" * 60)
print("[1] 01-10-C 3단계 — 손실과 그래디언트를 코드로 옮기기")
print("=" * 60)
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


#======================================================================
# [2] 01-10-E 5단계 — 경사하강법으로 하산하기
#     학습률 0.1 로 15걸음 이동하며 3걸음마다 위치·손실·|grad| 를 출력합니다.
#======================================================================
print("\n" + "=" * 60)
print("[2] 01-10-E 5단계 — 경사하강법으로 하산하기")
print("=" * 60)
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


#======================================================================
# [3] 01-10-E 5단계 — 옵티마이저 클래스로 감싸기
#     w ← w − η∇f 한 줄을 GradientDescentOptimizer.step() 으로 만듭니다 (PyTorch optimizer.step() 과 같은 이름).
#     ⚠️ 책의 옵티마이저 코드는 출력이 없어, 15걸음 후 위치를 확인하는 출력 코드를 추가했습니다(표시된 부분).
#======================================================================
print("\n" + "=" * 60)
print("[3] 01-10-E 5단계 — 옵티마이저 클래스로 감싸기")
print("=" * 60)
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

# [추가 코드 — 책 본문에는 없음] 옵티마이저로 15걸음 이동한 결과 확인
print(f"옵티마이저 15걸음 후 위치: ({position[0]:.6f}, {position[1]:.6f})")
print(f"손실: {compute_loss(position):.6f}")
