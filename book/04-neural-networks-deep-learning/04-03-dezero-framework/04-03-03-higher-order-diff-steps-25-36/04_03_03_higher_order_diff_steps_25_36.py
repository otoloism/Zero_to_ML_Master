# -*- coding: utf-8 -*-
"""
04-03-03 고차미분계산 (25 36단계) —헤시안과 뉴턴법

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-03장 딥러닝 프레임워크 — DeZero 직접 만들기/04-03-03 고차미분계산 (25 36단계) —헤시안과 뉴턴법.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 0] [추가] 책에서는 앞 절에서 import 한 모듈 — 단독 실행을 위해 보충 (원문에는 없음)
import numpy as np


# %% [Block 1] 25~26단계 — 계산 그래프 시각화: Graphviz로 그림 그리기
def get_dot_graph(output):
    """output에서 creator를 거슬러 올라가며 DOT 그래프 생성"""
    txt = ''
    funcs = [output.creator]      # 출력의 creator부터 시작
    seen = set()                  # 이미 방문한 함수 (중복 방지)

    while funcs:
        f = funcs.pop()
        if f not in seen:
            seen.add(f)
            for x in f.inputs:     # "입력 → 함수" 화살표
                txt += f'{id(x)} -> {id(f)}\n'
                if x.creator is not None:
                    funcs.append(x.creator)
            for y in f.outputs:    # "함수 → 출력" 화살표
                txt += f'{id(f)} -> {id(y())}\n'
    return 'digraph g {\n' + txt + '}'

# 사용 예시:
x = Variable(np.array(1.0))
y = (x + 3) ** 2
print(get_dot_graph(y))  # Graphviz가 이해하는 텍스트 출력


# %% [Block 2] 27단계 — 테일러 급수로 sin 함수 미분 검증
import math

def my_sin(x, threshold=1e-150):
    # sin(x) ≈ x - x³/3! + x⁵/5! - x⁷/7! + ...
    # 항을 하나씩 더해가며, 충분히 작아지면 멈춥니다
    y = 0                      # 결과를 담을 변수 (처음에 0)
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)  # 계수: ±1/(2i+1)!
        t = c * x ** (2 * i + 1)    # 각 항: c × x^(2i+1)
        y = y + t                    # 누적
        if abs(t.data) < threshold:  # 항이 충분히 작으면 종료
            break
    return y

x = Variable(np.array(np.pi / 4))  # π/4 = 45도
y = my_sin(x)
y.backward()

print("my_sin(π/4)  =", y.data)       # ≈ 0.7071 (sin 45°)
print("자동미분 결과 =", x.grad)       # ≈ 0.7071 (cos 45°)
print("실제 cos(π/4)=", np.cos(np.pi/4))  # → 0.7071... 일치! ✅


# %% [Block 3] 28단계 — 함수 최적화: 경사하강법으로 Rosenbrock 최솟값 찾기
def rosenbrock(x0, x1):
    # 구부러진 골짜기 함수, 최솟값: (1, 1)에서 f=0
    return 100 * (x1 - x0**2)**2 + (x0 - 1)**2

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 0.001  # 학습률 (한 걸음 크기)

for i in range(1000):
    y = rosenbrock(x0, x1)
    x0.grad, x1.grad = None, None  # 이전 기울기 초기화
    y.backward()                       # ★ 자동 미분!
    x0.data -= lr * x0.grad             # 기울기 반대로 이동
    x1.data -= lr * x1.grad

print(f"1000스텝 후: x0={x0.data:.4f}, x1={x1.data:.4f}")
# → x0=0.9268, x1=0.8581 (아직 (1,1)에 못 도달!)


# %% [Block 4] 29~31단계 — 뉴턴 방법: 수동 → 자동 2차 미분
def f(x):
    return x ** 4 - 2 * x ** 2  # 테스트 함수: y = x⁴ - 2x²

def gx(x):                       # 1차 미분 f'(x) = 4x³ - 4x
    return 4 * x ** 3 - 4 * x

def gx2(x):                      # 2차 미분 f''(x) = 12x² - 4
    return 12 * x ** 2 - 4

x = Variable(np.array(2.0))    # 시작: x = 2
for i in range(10):
    x.data -= gx(x.data) / gx2(x.data)  # x ← x - f'(x)/f''(x)
    print(f"step{i}: x = {x.data:.6f}")
# 5스텝 만에 x = 1.000000에 도달! ✅


# %% [Block 5] 32단계 — 고차 미분의 핵심: backward 자체를 그래프로 기록
class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0]       # ★ .data 없음! Variable 그대로!
        gx = 2 * x * gy           # Variable × Variable
                                  # → 연산자 오버로딩이 발동!
                                  # → 이 계산도 그래프에 기록됩니다!
        return gx


# %% [Block 6] 33단계 — 뉴턴 방법 자동화: 2차 미분도 자동으로!
def f(x):
    return x ** 4 - 2 * x ** 2

x = Variable(np.array(2.0))

for i in range(10):
    y = f(x)

    # ── 1차 미분 ──
    x.grad = None
    y.backward(create_graph=True)  # ★ 그래프 기록 ON!
    gx = x.grad                      # f'(x) = Variable!

    # ── 2차 미분 ──
    x.grad = None
    gx.backward()                    # ★ f'(x)를 다시 미분 → f''(x)!
    gx2 = x.grad                     # f''(x)

    # ── 뉴턴법 업데이트 ──
    x.data -= gx.data / gx2.data     # x ← x - f'(x)/f''(x)
    print(f"step{i}: x = {x.data:.6f}")


# %% [Block 7] 34~35단계 — sin 함수 고차 미분: 미분을 반복하면 돌아옵니다
x = Variable(np.array(1.0))
y = sin(x)

# 4계 미분까지 자동 계산
for i in range(4):
    if i == 0:
        y.backward(create_graph=True)
    else:
        gx = x.grad
        x.grad = None
        gx.backward(create_graph=True)
    print(f"{i+1}계 미분: {x.grad.data:.6f}")
