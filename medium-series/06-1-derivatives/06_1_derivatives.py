# -*- coding: utf-8 -*-
"""
Zero to 머신러닝 딥러닝 Master — Medium 시리즈 #06 · 1부
미분이란? 자동차 속도계로 이해하는 미분·연쇄법칙·그래디언트 (1부)

원문(책): 00-04 「미분·경사하강법 — 안개 낀 산에서 내려오기 (1부)」 https://wikidocs.net/439840
Medium : TODO (게시 후 추가)
책     : https://wikidocs.net/book/21464
홈페이지: https://mldict.net
블로그 : https://wikidocs.net/blog/@mldict/

실행: pip install -r requirements.txt && python 06_1_derivatives.py
※ 아래 코드는 책 본문의 코드 블록을 순서대로 옮긴 것입니다. [추가]/[수정] 표시가 있는 부분만 실행을 위해 보탰습니다.
"""

#======================================================================
# [1] 1단계 — 미분이란? 전진차분 vs 중심차분
#     (f(x+h)−f(x))/h 와 (f(x+h)−f(x−h))/2h 를 그대로 옮겨, h를 바꿔가며 오차를 비교합니다.
#======================================================================
print("\n" + "=" * 60)
print("[1] 1단계 — 미분이란? 전진차분 vs 중심차분")
print("=" * 60)
def forward_diff(f, x, h=1e-4):
    """
    전진차분: (f(x+h) - f(x)) / h
    비유: 앞쪽만 보고 경사를 재는 것 — 한쪽으로 치우친 측정
    """
    return (f(x + h) - f(x)) / h          # 공식 그대로!

def numerical_diff(f, x, h=1e-4):
    """
    중심차분: (f(x+h) - f(x-h)) / (2h)
    비유: 앞뒤로 똑같이 한 걸음씩 가서 재는 것 — 좌우 오차가 상쇄됨
    """
    return (f(x + h) - f(x - h)) / (2 * h)  # 공식 그대로!

def square(x):
    """f(x) = x^2  (참값 미분: f'(x) = 2x, 따라서 f'(3) = 6)"""
    return x ** 2

# h를 바꿔가며 두 방식의 오차를 비교합니다
true_value = 6.0
for h in [1e-2, 1e-4, 1e-8]:
    fwd = forward_diff(square, 3.0, h)
    cen = numerical_diff(square, 3.0, h)
    print(f"h={h:.0e} | 전진={fwd:.8f} (오차 {abs(fwd-true_value):.1e})"
          f" | 중심={cen:.8f} (오차 {abs(cen-true_value):.1e})")


#======================================================================
# [2] 2단계 — 미분 공식과 연쇄법칙 검산
#     손으로 유도한 도함수(연쇄법칙·시그모이드)를 수치미분과 상대오차로 비교해 채점합니다.
#======================================================================
print("\n" + "=" * 60)
print("[2] 2단계 — 미분 공식과 연쇄법칙 검산")
print("=" * 60)
import math

def numerical_diff(f, x, h=1e-4):
    """중심차분 수치미분 (1단계에서 만든 함수 재사용)"""
    return (f(x + h) - f(x - h)) / (2 * h)

def check_grad(name, f, df, x):
    """
    손으로 유도한 도함수 df 가 맞는지 수치미분으로 채점합니다.

    비유: 계산기로 검산하는 것과 같습니다.
    두 값이 거의 같으면 유도가 옳다는 강력한 증거입니다.
    """
    analytic = df(x)                      # 공식으로 구한 값
    numeric  = numerical_diff(f, x)       # 정의로 구한 값

    # ★ 절대오차가 아니라 '상대오차'로 비교합니다 (아래 디버깅 포인트 참고)
    #   기울기가 1536처럼 크면 절대오차도 커지는 게 정상이기 때문입니다
    rel_error = abs(analytic - numeric) / max(1.0, abs(analytic))
    ok = "✅" if rel_error < 1e-5 else "❌"
    print(f"{ok} {name:<22} 공식={analytic:12.6f}  수치={numeric:12.6f}"
          f"  상대오차={rel_error:.1e}")

# ── 예제 ①: y=(x+1)^2 → y'=2(x+1) ───────────────────────
check_grad("(x+1)^2",
           lambda x: (x + 1) ** 2,
           lambda x: 2 * (x + 1),          # 연쇄법칙 결과
           3.0)

# ── 예제 ②: y=(3x^2+1)^4 → y'=24x(3x^2+1)^3 ─────────────
check_grad("(3x^2+1)^4",
           lambda x: (3 * x**2 + 1) ** 4,
           lambda x: 24 * x * (3 * x**2 + 1) ** 3,
           1.0)

# ── 예제 ③: 3중 합성 y=((2x+1)^2+3)^2 → y'=8((2x+1)^2+3)(2x+1)
check_grad("((2x+1)^2+3)^2",
           lambda x: ((2*x + 1)**2 + 3) ** 2,
           lambda x: 8 * ((2*x + 1)**2 + 3) * (2*x + 1),
           0.0)

# ── 예제 ④: 시그모이드 → σ'(x)=σ(x)(1-σ(x)) ──────────────
def sigmoid(x):
    """σ(x) = 1 / (1 + e^-x)  — 0~1 사이로 눌러 담는 S자 함수"""
    return 1 / (1 + math.exp(-x))

def sigmoid_grad(x):
    """σ'(x) = σ(x)(1-σ(x)) — 순전파 결과만으로 계산! (유도 ⑤단계)"""
    y = sigmoid(x)
    return y * (1 - y)

check_grad("sigmoid (x=0)", sigmoid, sigmoid_grad, 0.0)
check_grad("sigmoid (x=2)", sigmoid, sigmoid_grad, 2.0)


#======================================================================
# [3] 2단계 — 순전파·역전파 클래스 (Square, Sigmoid)
#     f(x)=x², σ(x) 의 forward/backward 를 클래스로 구현합니다. 시그모이드는 출력 y를 저장해 y(1−y)로 미분합니다.
#     ⚠️ Function 기반 클래스는 책 04-03장에서 만드는 것이라 본문 코드만으로는 실행되지 않습니다. 실행을 위해 최소한의 Function 정의와 동작 확인 코드를 추가했습니다(표시된 부분).
#======================================================================
print("\n" + "=" * 60)
print("[3] 2단계 — 순전파·역전파 클래스 (Square, Sigmoid)")
print("=" * 60)
# [추가 코드 — 책 본문에는 없음]
# 책에서 Function 기반 클래스는 04-03장(DeZero 직접 만들기)에서 완성합니다.
# 여기서는 아래 Square / Sigmoid 클래스가 실행되도록 최소한의 Function 만 정의합니다.
import numpy as np

class Function:
    def __call__(self, x):
        return self.forward(x)

class Square(Function):
    """
    입력값을 제곱합니다.  f(x) = x^2,  f'(x) = 2x

    비유: 종이를 복사해서 두 장으로 만드는 것처럼
    같은 값을 한 번 더 곱합니다.
    """

    def forward(self, x):
        self.x = x              # 역전파에서 쓸 입력을 기억해둡니다
        return x ** 2           # 순전파: 공식 f(x)=x² 그대로

    def backward(self, gy):
        x = self.x
        gx = 2 * x * gy        # f'(x)=2x 에 연쇄법칙의 gy를 곱함
        return gx

class Sigmoid(Function):
    """
    σ(x) = 1/(1+e^-x),  σ'(x) = σ(x)(1-σ(x))

    비유: 아무리 큰 값이 들어와도 0~1 사이로 눌러 담는
    '압축 스펀지'와 같습니다.
    """

    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        self.y = y              # ★ 입력이 아니라 '출력'을 저장! (유도 ⑤ 덕분)
        return y

    def backward(self, gy):
        gx = gy * self.y * (1 - self.y)   # y(1-y) — 지수 재계산 불필요!
        return gx

# [추가 코드 — 책 본문에는 없음] 위 클래스의 동작 확인
sq, sg = Square(), Sigmoid()
print("Square : f(3) =", sq(3.0), "| f'(3)·1 =", sq.backward(1.0))
print("Sigmoid: σ(0) =", sg(0.0), "| σ'(0)·1 =", sg.backward(1.0))


#======================================================================
# [4] 3단계 — 편미분과 그래디언트 (수치 그래디언트)
#     성분을 하나씩 ±h 만큼 움직여 ∇f 를 구하고, 공식 (2x, 2y)·(6x+2y, 2x+3y²) 와 비교합니다.
#======================================================================
print("\n" + "=" * 60)
print("[4] 3단계 — 편미분과 그래디언트 (수치 그래디언트)")
print("=" * 60)
import numpy as np

def numerical_gradient(f, p, h=1e-4):
    """
    다변수 함수 f 의 점 p 에서의 그래디언트를 수치적으로 구합니다.

    비유: 엘리베이터 버튼을 하나씩 눌러보는 것과 같습니다.
    한 번에 하나의 성분만 살짝 움직여서 반응을 관찰합니다.
    """
    grad = np.zeros_like(p)              # 결과를 담을 빈 벡터 (p와 같은 모양)

    for i in range(p.size):        # 성분을 하나씩 순회 (= 편미분)
        tmp = p[i]                       # 원래 값을 잠시 보관

        p[i] = tmp + h                   # i번째만 +h
        f_plus  = f(p)

        p[i] = tmp - h                   # i번째만 -h
        f_minus = f(p)

        grad[i] = (f_plus - f_minus) / (2 * h)   # 중심차분
        p[i] = tmp                       # ★ 반드시 원상복구! (안 하면 다음 성분이 오염됨)

    return grad

# ── 예제 ①: f(x,y) = x² + y² ─────────────────────────────
def f_circle(p):
    """f(x,y) = x^2 + y^2  → ∇f = (2x, 2y)"""
    return p[0]**2 + p[1]**2

point = np.array([3.0, 4.0])
formula = np.array([2*point[0], 2*point[1]])   # 공식 ∇f=(2x,2y)
numeric = numerical_gradient(f_circle, point.copy())

print("[예제①] 공식 :", formula)
print("[예제①] 수치 :", np.round(numeric, 6))
print("[예제①] 크기 :", np.linalg.norm(formula), "← 00-01장의 노름!")

# ── 예제 ②: f(x,y) = 3x² + 2xy + y³ ──────────────────────
def f_cross(p):
    """∂f/∂x = 6x+2y,  ∂f/∂y = 2x+3y²"""
    x, y = p[0], p[1]
    return 3*x**2 + 2*x*y + y**3

p2 = np.array([1.0, 2.0])
formula2 = np.array([6*1 + 2*2, 2*1 + 3*2**2])   # (10, 14)
print("[예제②] 공식 :", formula2)
print("[예제②] 수치 :", np.round(numerical_gradient(f_cross, p2.copy()), 6))
