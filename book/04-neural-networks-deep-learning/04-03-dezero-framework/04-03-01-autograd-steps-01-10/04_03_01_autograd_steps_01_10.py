# -*- coding: utf-8 -*-
"""
04-03-01 미분자동 계산 (1 10단계) — 오토그라드의 탄생

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-03장 딥러닝 프레임워크 — DeZero 직접 만들기/04-03-01 미분자동 계산 (1 10단계) — 오토그라드의 탄생.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 1단계 — 상자로서의 변수: Variable 클래스
import numpy as np  # NumPy: 파이썬의 수학 계산 라이브러리

class Variable:
    # Variable = "데이터 + 기울기 + 출신 기록"을 담는 상자
    def __init__(self, data):
        self.data = data       # 실제 숫자 데이터를 저장합니다
        self.grad = None       # 기울기(미분값) 저장 자리 (아직 비어있음)
        self.creator = None    # "나를 만든 함수" 기억 자리

x = Variable(np.array(1.0))  # 숫자 1.0을 상자에 넣기
print(x.data)                   # → 1.0


# %% [Block 2] 2단계 — 변수를 낳는 함수: Function 클래스
class Function:
    def __call__(self, input):
        x = input.data           # ① Variable 상자에서 데이터 꺼내기
        y = self.forward(x)      # ② 실제 계산 (자식 클래스가 담당)
        output = Variable(y)     # ③ 결과를 새 상자에 담기
        output.creator = self    # ④ "이 출력은 내가 만들었다!" 기록
        self.input = input       # ⑤ 역전파용 입력값 저장
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()  # 자식 클래스에서 반드시 구현!

    def backward(self, gy):
        raise NotImplementedError()  # 자식 클래스에서 반드시 구현!

# 구체적인 함수: Square(제곱)와 Exp(지수)
class Square(Function):
    def forward(self, x):
        return x ** 2             # x² 계산

class Exp(Function):
    def forward(self, x):
        return np.exp(x)          # eˣ 계산


# %% [Block 3] 3단계 — 함수 연결: 체인처럼 이어서 호출
A = Square()    # 첫 번째 제곱
B = Exp()       # 지수 함수
C = Square()    # 두 번째 제곱

x = Variable(np.array(0.5))  # 입력: x = 0.5
a = A(x)    # a = 0.5² = 0.25
b = B(a)    # b = e^0.25 ≈ 1.284
y = C(b)    # y = 1.284² ≈ 1.649

print("y =", y.data)  # → 1.648721270700128

# creator를 따라가면 계산 경로를 추적할 수 있습니다
# y.creator = C, C.input = b, b.creator = B, ...


# %% [Block 4] 4단계 — 수치 미분: "정답지"를 미리 만들기
def numerical_diff(f, x, eps=1e-4):
    # 수치 미분: 두 점 사이의 기울기를 직접 잽니다
    # 기울기 = (y변화량) / (x변화량) — 중학교 수학과 같습니다!
    x0 = Variable(x.data - eps)   # x를 살짝 왼쪽으로
    x1 = Variable(x.data + eps)   # x를 살짝 오른쪽으로
    y0 = f(x0)                     # 왼쪽 점의 y값
    y1 = f(x1)                     # 오른쪽 점의 y값
    return (y1.data - y0.data) / (2 * eps)  # 기울기 = Δy / Δx

def f(x):
    # 테스트 함수: y = (e^(x²))²
    A, B, C = Square(), Exp(), Square()
    return C(B(A(x)))

x = Variable(np.array(0.5))
dy = numerical_diff(f, x)
print("수치 미분 dy/dx =", dy)  # → 3.2974426293330694 (목표값!)


# %% [Block 5] 6단계 — 수동 역전파: backward()를 직접 추가하고 하나씩 호출
class Square(Function):
    def forward(self, x):
        return x ** 2
    def backward(self, gy):
        x = self.input.data
        return 2 * x * gy   # (x²)' = 2x, 연쇄법칙으로 gy를 곱합니다

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    def backward(self, gy):
        x = self.input.data
        return np.exp(x) * gy  # (eˣ)' = eˣ (자기 자신!)

# ── 수동 역전파: creator를 따라 하나씩 호출 ──
A, B, C = Square(), Exp(), Square()
x = Variable(np.array(0.5))
a = A(x); b = B(a); y = C(b)

y.grad = np.array(1.0)       # 시작: dy/dy = 1
b.grad = C.backward(y.grad)   # C(Square)의 역전파
a.grad = B.backward(b.grad)   # B(Exp)의 역전파
x.grad = A.backward(a.grad)   # A(Square)의 역전파

print("수동 역전파 dy/dx =", x.grad)  # → 3.297... (4단계와 일치!)


# %% [Block 6] 7단계 — 역전파 자동화: backward() 메서드 (★ 핵심!)
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        # ★ 6단계의 수동 반복을 자동화합니다!
        if self.grad is None:
            self.grad = np.ones_like(self.data)  # dy/dy = 1

        funcs = [self.creator]     # 처리할 함수 목록
        while funcs:                # 목록이 빌 때까지 반복
            f = funcs.pop()         # 함수를 하나 꺼냅니다
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)  # 역전파 실행!
            if x.creator is not None:
                funcs.append(x.creator)  # 더 올라갈 게 있으면 추가

# ── 사용: 이제 한 줄로 끝! ──
x = Variable(np.array(0.5))
a = A(x); b = B(a); y = C(b)
y.backward()  # ★ 이 한 줄로 전체 역전파 자동 실행!
print("자동 역전파 dy/dx =", x.grad)  # → 3.297... ✅


# %% [Block 7] 8단계 — 재귀에서 반복문으로 — ❌ 재귀 방식 (위험): 깊은 그래프에서 스택 오버플로!
# ❌ 재귀 방식 (위험): 깊은 그래프에서 스택 오버플로!
def backward_recursive(self):
    f = self.creator
    if f is not None:
        x = f.input
        x.grad = f.backward(self.grad)
        x.backward_recursive()  # 자기 자신을 다시 호출 → 위험!

# ✅ 반복문 방식 (안전): 7단계에서 이미 구현 완료!
# while funcs: 로 처리 → 깊이에 상관없이 안전합니다


# %% [Block 8] 9단계 — 함수를 더 편리하게: 파이썬 함수로 감싸기
def square(x):
    # Square 클래스를 만들고 바로 호출하는 편의 함수
    return Square()(x)  # Square()로 객체 생성 후 (x)로 호출

def exp(x):
    return Exp()(x)

# 사용: 훨씬 자연스럽습니다!
x = Variable(np.array(0.5))
y = square(exp(square(x)))  # C(B(A(x))) 대신 이렇게!
y.backward()
print("dy/dx =", x.grad)   # → 3.297... ✅


# %% [Block 9] 10단계 — 테스트: unittest로 자동 미분 검증
import unittest

class SquareTest(unittest.TestCase):

    def test_forward(self):
        # 순전파 테스트: 2² = 4 인가?
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertTrue(np.allclose(y.data, expected))

    def test_backward(self):
        # 역전파 테스트: (x²)' = 2x, x=3이면 6인가?
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)  # 2 × 3 = 6
        self.assertTrue(np.allclose(x.grad, expected))

    def test_gradient_check(self):
        # ★ 핵심 테스트: 자동 미분 vs 수치 미분 비교
        x = Variable(np.random.rand(1))  # 무작위 값으로 테스트
        y = square(x)
        y.backward()

        num_grad = numerical_diff(square, x)  # 수치 미분 (정답지)
        # 두 결과가 충분히 가까운지 확인합니다
        self.assertTrue(np.allclose(x.grad, num_grad))

# 테스트 실행
unittest.main()
