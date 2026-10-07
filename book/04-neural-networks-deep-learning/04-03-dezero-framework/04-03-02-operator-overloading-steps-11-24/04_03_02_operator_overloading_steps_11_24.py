# -*- coding: utf-8 -*-
"""
04-03-02 자연스러운 코드 (11 24단계) — 연산자 오버로딩

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-03장 딥러닝 프레임워크 — DeZero 직접 만들기/04-03-02 자연스러운 코드 (11 24단계) — 연산자 오버로딩.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 0] [추가] 책에서는 앞 절에서 import 한 모듈 — 단독 실행을 위해 보충 (원문에는 없음)
import numpy as np


# %% [Block 1] 11~12단계 — 가변 길이 인수: 입력이 여러 개인 함수
class Function:
    def __call__(self, *inputs):        # *inputs: 입력이 몇 개든 OK!
        xs = [x.data for x in inputs]  # ① 각 Variable에서 데이터 꺼내기
        ys = self.forward(*xs)           # ② 순전파 (자식 클래스가 담당)
        if not isinstance(ys, tuple):
            ys = (ys,)                    # ③ 출력도 항상 튜플로 통일
        outputs = [Variable(y) for y in ys]
        for output in outputs:
            output.set_creator(self)    # ④ "내가 만들었다" 기록
        self.inputs = inputs              # ⑤ 입력들 저장 (여러 개!)
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1                # 순전파: 그냥 더합니다
    def backward(self, gy):
        return gy, gy                  # 역전파: 같은 기울기를 양쪽에!
        # 덧셈 노드는 "그대로 양쪽으로 흘려보냅니다"

x0 = Variable(np.array(2.0))
x1 = Variable(np.array(3.0))
y = Add()(x0, x1)      # 2 + 3 = 5
print("y =", y.data)   # → 5.0


# %% [Block 2] 13~14단계 — 같은 변수 반복 사용: 기울기 누적 — backward() 내부의 핵심 변경 한 줄:
# backward() 내부의 핵심 변경 한 줄:
for x, gx in zip(f.inputs, gxs):
    if x.grad is None:
        x.grad = gx              # 처음이면 그냥 대입
    else:
        x.grad = x.grad + gx     # ★ 이미 있으면 누적! (= → +=)

# 테스트: y = x + x → dy/dx = 2
x = Variable(np.array(3.0))
y = Add()(x, x)     # y = 3 + 3 = 6
y.backward()
print("dy/dx =", x.grad)  # → 2.0 ✅ (1이 아닌 2!)


# %% [Block 3] 15~16단계 — 복잡한 그래프와 세대(generation) 정렬 — Function.__call__에서 세대 계산
# Function.__call__에서 세대 계산
self.generation = max([x.generation for x in inputs])
for output in outputs:
    output.generation = self.generation + 1  # 출력은 한 세대 뒤

# backward에서 세대 순 정렬
def add_func(f):
    if f not in seen:
        funcs.append(f)
        seen.add(f)
        funcs.sort(key=lambda x: x.generation)  # ★ 세대 순!


# %% [Block 4] 17~18단계 — 메모리 관리: weakref와 retain_grad
import weakref

class Function:
    def __call__(self, *inputs):
        ...
        # ★ outputs를 약한 참조로 저장합니다
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

def backward(self, retain_grad=False):
    ...
    if not retain_grad:
        for y in f.outputs:
            y().grad = None  # 중간 변수의 grad를 비워 메모리 절약!


# %% [Block 5] 19~20단계 — 변수 사용성: shape, len, print
class Variable:
    @property
    def shape(self):      # x.shape → (3, 2) 처럼 크기 확인
        return self.data.shape

    @property
    def ndim(self):       # x.ndim → 2 (몇 차원인지)
        return self.data.ndim

    @property
    def size(self):       # x.size → 6 (전체 원소 개수)
        return self.data.size

    def __len__(self):     # len(x) → 3 (첫 번째 축의 길이)
        return len(self.data)

    def __repr__(self):    # print(x) → variable([1,2,3])
        return 'variable(' + str(self.data) + ')'

x = Variable(np.array([[1,2,3],[4,5,6]]))
print(x.shape, x.ndim, x.size, len(x))  # → (2,3) 2 6 2


# %% [Block 6] 21~22단계 — 연산자 오버로딩: +, * 를 자연스럽게!
class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1             # 순전파: 곱합니다
    def backward(self, gy):
        x0, x1 = self.inputs
        return gy * x1, gy * x0    # 역전파: "서로 바꿔서" 곱합니다

class Variable:
    def __add__(self, other):     # x + y → Add()(x, y)
        return Add()(self, other)
    def __mul__(self, other):     # x * y → Mul()(x, y)
        return Mul()(self, other)
    def __radd__(self, other):    # 2 + x → Add()(x, 2)
        return Add()(self, other)
    def __rmul__(self, other):    # 3 * x → Mul()(x, 3)
        return Mul()(self, other)

# ★ 이제 자연스럽게 수학 공식처럼 쓸 수 있습니다!
x = Variable(np.array(2.0))
y = Variable(np.array(3.0))
z = x * y + x    # z = 2×3 + 2 = 8 (수학 공식 그대로!)
z.backward()
print("z =", z.data, "dz/dx =", x.grad, "dz/dy =", y.grad)


# %% [Block 7] 23단계 — 패키지 설계: 모듈 분리 — 디렉토리 구조:
# 디렉토리 구조:
# dezero/
#   ├── __init__.py     ← "이 폴더는 패키지입니다" 표시
#   ├── core.py         ← Variable, Function (핵심)
#   └── functions.py    ← Square, Exp, Add, Mul (연산들)

# 사용법:
from dezero import Variable    # 이제 한 줄로 가져올 수 있습니다!
from dezero.functions import square, exp


# %% [Block 8] 24단계 — 복잡한 함수 미분: 최적화 벤치마크로 검증 — ① Sphere 함수: z = x² + y² (가장 단순한 볼록함수)
# ── ① Sphere 함수: z = x² + y² (가장 단순한 볼록함수) ──
def sphere(x, y):
    z = x ** 2 + y ** 2    # "밥그릇" 모양: 최솟값 = (0,0)
    return z

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = sphere(x, y)
z.backward()
print("Sphere: dz/dx =", x.grad, "dz/dy =", y.grad)
# → dz/dx = 2.0, dz/dy = 2.0 (이론: 2x=2, 2y=2) ✅

# ── ② Matyas 함수: z = 0.26(x²+y²) - 0.48xy ──
def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z               # 타원 모양, 최솟값 = (0,0)

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = matyas(x, y)
z.backward()
print("Matyas: dz/dx =", x.grad, "dz/dy =", y.grad)
# → dz/dx = 0.04, dz/dy = 0.04
# 이론: 0.52x - 0.48y = 0.52-0.48 = 0.04 ✅

# ── ③ Goldstein-Price 함수 (가장 복잡!) ──
def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z  # 이 복잡한 수식도 y.backward() 한 줄로 미분 끝!

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = goldstein(x, y)
z.backward()
print("Goldstein: dz/dx =", x.grad, "dz/dy =", y.grad)
# → dz/dx = -5376.0, dz/dy = 8064.0
