# -*- coding: utf-8 -*-
"""
04-03-04 신경망 만들기 (37 51단계) —PyTorchnn Module 직접 구현

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-03장 딥러닝 프레임워크 — DeZero 직접 만들기/04-03-04 신경망 만들기 (37 51단계) —PyTorchnn Module 직접 구현.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 0] [추가] 책에서는 앞 절에서 import 한 모듈 — 단독 실행을 위해 보충 (원문에는 없음)
import numpy as np


# %% [Block 1] 37~38단계 — 텐서: 스칼라에서 다차원 배열로 확장
class Reshape(Function):
    # 형상(shape)을 바꾸는 함수: (2,3) → (6,) 또는 (3,2)
    def __init__(self, shape):
        self.shape = shape       # 바꿀 목표 형상
    def forward(self, x):
        self.x_shape = x.shape   # 원래 형상을 기억 (역전파용)
        return x.reshape(self.shape)
    def backward(self, gy):
        return reshape(gy, self.x_shape)  # 원래 형상으로 되돌리기

class Sum(Function):
    # 합계 함수: [1,2,3] → 6
    def forward(self, x):
        self.x_shape = x.shape
        return x.sum()
    def backward(self, gy):
        return gy * np.ones(self.x_shape)  # 모든 원소에 같은 기울기!

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = sum(x)           # 1+2+3+4+5+6 = 21
y.backward()
print(x.grad)         # → [[1,1,1],[1,1,1]] (모든 원소에 1씩)


# %% [Block 2] 39단계 — 브로드캐스트: 크기가 다른 텐서끼리 연산
x = Variable(np.array([1,2,3]))
b = Variable(np.array(10))    # 스칼라 (크기 1)
y = x + b                      # [11, 12, 13] — b가 자동 확장!
y.backward()
print(b.grad)                   # → 3 (3개 원소의 기울기가 합산됨)


# %% [Block 3] 40~41단계 — 행렬 곱(MatMul): 신경망의 핵심 연산
class MatMul(Function):
    def forward(self, x, W):
        return x.dot(W)          # 행렬 곱: x @ W
    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)   # dx: gy × Wᵀ (04-01-04과 동일!)
        gW = matmul(x.T, gy)   # dW: xᵀ × gy
        return gx, gW


# %% [Block 4] 42단계 — 선형 회귀: DeZero로 첫 번째 ML 모델!
np.random.seed(0)
x = np.random.rand(100, 1)         # 입력: 100개 데이터
y = 5 + 2 * x + np.random.rand(100, 1)  # 정답: y ≈ 2x + 5 (노이즈 포함)
x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1, 1)))    # 가중치: 처음에 0
b = Variable(np.zeros(1))          # 편향: 처음에 0
lr = 0.1

for i in range(100):
    y_pred = matmul(x, W) + b     # ① 예측: y = xW + b
    loss = mean_squared_error(y, y_pred)  # ② 손실: MSE

    W.grad, b.grad = None, None
    loss.backward()               # ③ 역전파 (자동!)

    W.data -= lr * W.grad         # ④ 가중치 갱신
    b.data -= lr * b.grad
    if i % 20 == 0: print(f"step{i}: W={W.data.item():.3f}, b={b.data.item():.3f}")


# %% [Block 5] 43단계 — 신경망: 2층 네트워크 구현 — 비선형 데이터: y = sin(2πx) + 노이즈
# 비선형 데이터: y = sin(2πx) + 노이즈
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# 가중치 초기화
I, H, O = 1, 10, 1  # 입력 1 → 은닉 10 → 출력 1
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(O))

def predict(x):
    h = sigmoid(matmul(x, W1) + b1)  # 1층: 선형변환 + 활성화
    y = matmul(h, W2) + b2              # 2층: 선형변환
    return y

lr = 0.2
for i in range(10000):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)
    W1.grad = W2.grad = b1.grad = b2.grad = None
    loss.backward()
    for p in [W1, b1, W2, b2]:  # 모든 파라미터 갱신
        p.data -= lr * p.grad


# %% [Block 6] 44~45단계 — Layer와 Model: PyTorch nn.Module의 정체
class Parameter(Variable):
    # Variable을 상속받되, "이것은 학습 대상"이라고 표시
    pass

class Layer:
    # 서랍 한 칸: Parameter들을 자동으로 찾아 모읍니다
    def params(self):
        for name in self.__dict__:
            obj = self.__dict__[name]
            if isinstance(obj, Parameter):
                yield obj              # Parameter만 골라서 반환!
            elif isinstance(obj, Layer):
                yield from obj.params()  # 하위 Layer의 Parameter까지!

class Linear(Layer):
    # 04-01-04의 Affine과 동일: y = xW + b
    def __init__(self, in_size, out_size):
        self.W = Parameter(np.random.randn(in_size, out_size))
        self.b = Parameter(np.zeros(out_size))
    def forward(self, x):
        return matmul(x, self.W) + self.b

# ★ 43단계의 코드가 이렇게 깔끔해집니다!
class TwoLayerNet(Layer):
    def __init__(self):
        self.l1 = Linear(1, 10)    # 1층: 1→10
        self.l2 = Linear(10, 1)   # 2층: 10→1
    def forward(self, x):
        h = sigmoid(self.l1(x))   # 1층 + 활성화
        return self.l2(h)          # 2층

model = TwoLayerNet()
for p in model.params():        # 모든 파라미터를 자동으로 순회!
    print(p.shape)               # → (1,10), (10,), (10,1), (1,)


# %% [Block 7] 46단계 — Optimizer: SGD, Momentum, Adam
class Optimizer:
    def __init__(self, params):
        self.params = list(params)  # 갱신할 파라미터 목록
    def step(self):
        for param in self.params:
            self.update_one(param)   # 각 파라미터를 갱신
    def zero_grad(self):
        for param in self.params:
            param.grad = None       # 기울기 초기화

class SGD(Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params)
        self.lr = lr
    def update_one(self, param):
        param.data -= self.lr * param.grad  # θ ← θ - lr·∇θ

# 사용법 (PyTorch와 거의 동일합니다!)
model = TwoLayerNet()
optimizer = SGD(model.params(), lr=0.2)

for i in range(10000):
    y_pred = model(x)
    loss = mean_squared_error(y, y_pred)
    optimizer.zero_grad()   # ① 기울기 초기화
    loss.backward()          # ② 역전파
    optimizer.step()          # ③ 파라미터 갱신


# %% [Block 8] 47~48단계 — Softmax + 교차엔트로피로 다중 분류 — spiral 데이터: 나선형으로 꼬인 3개 클래스
# spiral 데이터: 나선형으로 꼬인 3개 클래스
x, t = get_spiral(train=True)  # x: (300,2) 좌표, t: (300,) 클래스

model = Layer()
model.l1 = Linear(2, 10)     # 입력 2차원 → 은닉 10
model.l2 = Linear(10, 3)    # 은닉 10 → 출력 3 (3개 클래스)

optimizer = SGD(model.params(), lr=1.0)

for epoch in range(300):
    y = model.l2(sigmoid(model.l1(x)))
    loss = softmax_cross_entropy(y, t)  # 분류용 손실!
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"epoch {epoch}: loss = {loss.data:.4f}")


# %% [Block 9] 49~50단계 — Dataset과 DataLoader: 미니배치 자동화
class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError  # dataset[i]로 i번째 데이터
    def __len__(self):
        raise NotImplementedError  # len(dataset)으로 전체 크기

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size    # 한 번에 몇 개씩
        self.shuffle = shuffle          # 섞을지 여부

    def __iter__(self):
        # for문에서 자동으로 미니배치를 하나씩 돌려줍니다
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)  # 섞기!
        for i in range(0, len(self.dataset), self.batch_size):
            batch_idx = indices[i:i+self.batch_size]
            yield [self.dataset[j] for j in batch_idx]


# %% [Block 10] 51단계 — MNIST 학습: DeZero로 95%+ 정확도 달성! 🎉 — 데이터 준비
# ── 데이터 준비 ──
train_set = MNIST(train=True)       # 60,000장 학습 데이터
test_set = MNIST(train=False)      # 10,000장 테스트 데이터
train_loader = DataLoader(train_set, batch_size=100)
test_loader = DataLoader(test_set, batch_size=100, shuffle=False)

# ── 모델 & 옵티마이저 ──
class MLP(Layer):
    def __init__(self):
        self.l1 = Linear(784, 100)   # 28×28=784 → 100
        self.l2 = Linear(100, 10)   # 100 → 10 (0~9 숫자)
    def forward(self, x):
        h = relu(self.l1(x))      # ReLU 활성화 (04-01-04!)
        return self.l2(h)

model = MLP()
optimizer = SGD(model.params(), lr=0.01)

# ── 학습 루프 (PyTorch와 거의 동일!) ──
for epoch in range(5):
    for batch in train_loader:       # 미니배치 자동 순회!
        x, t = batch
        y = model(x)
        loss = softmax_cross_entropy(y, t)
        optimizer.zero_grad()         # ① 초기화
        loss.backward()                # ② 역전파
        optimizer.step()                # ③ 갱신

    # 정확도 측정
    acc = evaluate(model, test_loader)
    print(f"epoch {epoch+1}: acc = {acc:.1%}")


# %% [Block 11] 04-03-04-A 지금까지는 숫자 하나(스칼라)였습니다 — Sum 함수: 여러 숫자를 하나로 더하는 연산을 자동미분 가능하게 만든 클래스
# Sum 함수: 여러 숫자를 하나로 더하는 연산을 자동미분 가능하게 만든 클래스
class Sum(Function):
    def forward(self, x):
        self.x_shape = x.shape     # 원래 shape 저장 (역전파에서 복원용)
        return x.sum()             # 모든 원소를 더해서 스칼라로

    def backward(self, gy):
        # 스칼라 기울기를 원래 shape로 복제해서 뿌려줌
        gx = broadcast_to(gy, self.x_shape)
        return gx

# MatMul 함수: 행렬곱을 자동미분 가능하게 만든 클래스
class MatMul(Function):
    def forward(self, x, W):
        return x.dot(W)            # 행렬 x와 행렬 W를 곱함

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)      # 04-01-04 Affine: dx = dout @ W.T
        gW = matmul(x.T, gy)      # 04-01-04 Affine: dW = x.T @ dout
        return gx, gW


# %% [Block 12] 04-03-04-B 00-03의 최소제곱법을, 경사하강법 + 자동미분으로
import numpy as np
np.random.seed(0)
x = np.random.rand(100, 1)              # 0~1 난수 100개
y = 5 + 2*x + np.random.rand(100, 1)    # y ≈ 2x + 5 (약간의 잡음 포함)

W = Variable(np.zeros((1,1)))   # 기울기: 0에서 시작
b = Variable(np.zeros(1))       # 절편: 0에서 시작

def predict(x):
    return matmul(x, W) + b     # y = xW + b (04-01장의 그 식)

def mean_squared_error(x0, x1):
    diff = x0 - x1
    return sum(diff**2) / len(diff.data)   # MSE: 오차의 제곱 평균

lr = 0.1  # 학습률: 한 걸음의 크기

for i in range(100):
    y_pred = predict(x)                     # 예측
    loss = mean_squared_error(y, y_pred)     # 채점
    W.grad, b.grad = None, None             # 기울기 초기화
    loss.backward()                          # 역전파: 기울기 자동 계산
    W.data -= lr * W.grad.data              # 경사하강법: θ ← θ - η∇L
    b.data -= lr * b.grad.data
    if i % 20 == 0:
        print(f"step{i}: loss={loss.data:.4f}, W={W.data[0,0]:.3f}, b={b.data[0]:.3f}")


# %% [Block 13] 04-03-04-C 가중치를 "특별한 변수"로 표시하기 — Parameter: Variable에 "학습 대상" 스티커를 붙인 것 (기능 동일, 타입만 다름)
# Parameter: Variable에 "학습 대상" 스티커를 붙인 것 (기능 동일, 타입만 다름)
class Parameter(Variable):
    pass

# Layer: 가중치를 자동 정리해주는 "서랍장"
class Layer:
    def __init__(self):
        self._params = set()
    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)      # Parameter/Layer 넣으면 자동 등록
        super().__setattr__(name, value)
    def params(self):
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                yield from obj.params()  # 중첩 Layer도 재귀 탐색
            else:
                yield obj

# Linear: y = xW + b (04-01-04 Affine과 동일)
class Linear(Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.W = Parameter(np.random.randn(in_size, out_size) * 0.01)
        self.b = Parameter(np.zeros(out_size))
    def __call__(self, x):
        return matmul(x, self.W) + self.b

# Model: 여러 Layer를 조립한 완성품
class Model(Layer):
    def __init__(self):
        super().__init__()
        self.l1 = Linear(1, 10)
        self.l2 = Linear(10, 1)
    def forward(self, x):
        y = sigmoid(self.l1(x))   # l1 → 활성화(sigmoid) → l2
        return self.l2(y)


# %% [Block 14] 04-03-04-F Optimizer + 학습 루프 — PyTorch와 동일한 3줄 — SGD: 기본 경사하강법
# SGD: 기본 경사하강법
class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr
    def update_one(self, param):
        param.data -= self.lr * param.grad.data

# 학습 루프 — PyTorch와 완전히 같은 패턴!
model = Model()
optimizer = MomentumSGD(lr=0.1).setup(model)

for i in range(100):
    y_pred = model(x)                    # ① 예측
    loss = mean_squared_error(y, y_pred)  # ② 채점
    model.cleargrads()                    # ③ 기울기 초기화 (= zero_grad)
    loss.backward()                       # ④ 역전파        (= backward)
    optimizer.update()                     # ⑤ 파라미터 갱신  (= step)
