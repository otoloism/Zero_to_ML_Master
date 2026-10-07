# -*- coding: utf-8 -*-
"""
03-02-04 NumPy/PyTorch 오토그라드·옵티마이저 실습 — 미분을 대신 계산해주는 자동 계산기

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-02장 - Parameter Learning/03-02-04 Numpy PyTorch 오토그라드·옵티마이저 실습.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 0] [추가] 책에서는 앞 절에서 import 한 모듈 — 단독 실행을 위해 보충 (원문에는 없음)
import torch


# %% [Block 1] linear_regression_numpy.py
import numpy as np

# ── ① 파라미터를 아무 값에서나 출발시킵니다 ────────────────
# a = 절편(θ₀), b = 기울기(θ₁).
# 정답이 뭔지 모르니 일단 랜덤한 자리에 서서 시작합니다.
# (안개 낀 산의 아무 지점에 떨어진 상황을 떠올리세요)
# seed(42)는 "난수의 씨앗"을 고정해, 몇 번을 실행해도
# 똑같은 난수가 나오게 합니다. 책의 결과와 내 화면이 같아집니다.
np.random.seed(42)
a = np.random.randn(1)
b = np.random.randn(1)

print(a, b)   # 학습 전 값을 찍어 두면 나중에 비교할 수 있습니다

# ── ② 학습률(보폭) ────────────────────────────────────────
# 1e-1 은 0.1 을 뜻하는 지수 표기입니다 (1 × 10⁻¹).
# 한 번에 얼마나 크게 움직일지 정하는 값입니다.
# 너무 크면 튕겨 나가고, 너무 작으면 영원히 도착하지 못합니다.
lr = 1e-1

# ── ③ 반복 횟수 ──────────────────────────────────────────
# epoch(에폭) = 전체 데이터를 한 번 다 훑는 것.
n_epochs = 1000

for epoch in range(n_epochs):

    # ── 1단계: 예측 ────────────────────────────────────
    # 현재의 a, b 로 80개 데이터의 예측값을 "한 번에" 계산합니다.
    # 반복문 없이 배열끼리 곱하는 것이 NumPy의 힘(벡터화)입니다.
    yhat = a + b * x_train

    # ── 2단계: 오차 ────────────────────────────────────
    # 정답 − 예측. 부호 순서에 주의하세요!
    # 이 순서 때문에 뒤의 기울기에 마이너스가 붙습니다.
    error = (y_train - yhat)

    # ── 3단계: 손실 ────────────────────────────────────
    # 회귀 문제이므로 평균제곱오차(MSE)를 씁니다.
    # 제곱 → 부호 없애기 (+100과 −100이 상쇄되지 않도록)
    # mean → 데이터 개수와 무관한 하나의 점수로 요약
    loss = (error ** 2).mean()

    # ── 4단계: 기울기 ──────────────────────────────────
    # "a를 조금 키우면 손실이 늘까 줄까?"에 대한 답이 a_grad 입니다.
    # −2 의 2 는 제곱을 미분할 때 앞으로 내려온 숫자,
    # −(마이너스)는 error = y − yhat 안의 yhat 을 미분해서 생긴 부호입니다.
    a_grad = -2 * error.mean()

    # b 는 x 와 곱해져 있으므로 기울기에도 x 가 함께 곱해집니다.
    # (x가 큰 데이터일수록 b를 더 세게 밀어야 한다는 뜻입니다)
    b_grad = -2 * (x_train * error).mean()

    # ── 5단계: 갱신 ────────────────────────────────────
    # 기울기의 "반대" 방향으로 lr 만큼 한 발. 그래서 빼기(−)입니다.
    # 오른쪽을 다 계산한 뒤 왼쪽에 넣으므로 '동시 갱신'이 지켜집니다.
    a = a - lr * a_grad
    b = b - lr * b_grad

print(a, b)   # 학습 후 값. 1 과 2 에 가까워졌는지 확인!


# %% [Block 2] sanity_check.py — 내 구현이 맞는지 검산하기
# Sanity Check: do we get the same results as our gradient descent?
# sklearn 은 경사하강법이 아니라 수학 공식으로 한 번에 정답을 구합니다.
# 서로 다른 길로 갔는데 같은 곳에 도착하면 → 내 구현이 맞다는 강력한 증거!
from sklearn.linear_model import LinearRegression

linr = LinearRegression()   # 빈 모델을 하나 만들고
linr.fit(x_train, y_train)  # 데이터를 먹여 학습시킵니다 (fit = 맞추다)

# intercept_ = 절편(a), coef_ = 기울기(b).
# 뒤의 밑줄 _ 은 "학습을 통해 얻어진 값"이라는 sklearn의 이름 규칙입니다.
print(linr.intercept_, linr.coef_[0])


# %% [Block 3] autograd.py
lr = 1e-1
n_epochs = 1000

for epoch in range(n_epochs):

    # ── 순전파: 여기까지는 NumPy 버전과 코드가 똑같습니다 ──
    # 다만 a, b 가 requires_grad=True 인 텐서이므로,
    # 이 계산들이 "블랙박스에 녹화"되고 있다는 점이 다릅니다.
    yhat = a + b * x_train_tensor
    error = y_train_tensor - yhat
    loss = (error ** 2).mean()

    # ── 역전파: 손으로 유도하던 부분이 통째로 사라집니다 ──
    # a_grad = -2 * error.mean()  ← 이런 줄을 더 이상 쓰지 않습니다!
    # loss 에서 출발해 그래프를 거꾸로 훑으며 모든 기울기를 계산합니다.
    loss.backward()

    # ── 파라미터 갱신 ──────────────────────────────────
    # with torch.no_grad(): 는 "이 안에서는 녹화하지 마"라는 뜻입니다.
    # 파라미터를 고치는 일은 '학습의 일부'가 아니라 '정비 작업'이므로
    # 계산 그래프에 기록되면 안 됩니다. (기록하면 그래프가 무한히 자랍니다)
    with torch.no_grad():
        # -= 는 "왼쪽에서 오른쪽을 빼서 다시 왼쪽에 넣어라"는 뜻입니다.
        a -= lr * a.grad
        b -= lr * b.grad

    # ── 기울기 초기화 (아주 중요!) ─────────────────────
    # PyTorch는 backward() 를 부를 때마다 .grad 에 값을 "더합니다".
    # 지우지 않으면 이번 기울기 + 지난 기울기가 섞여 학습이 망가집니다.
    # 이름 끝의 언더스코어(_)는 "제자리에서 바꾼다"는 PyTorch 규칙입니다.
    a.grad.zero_()
    b.grad.zero_()


# %% [Block 4] optimizer.py
import torch.optim as optim   # 최적화 도구 모음. optim 이라는 별명이 관례입니다.

# ── 옵티마이저 만들기 ──────────────────────────────────
# 첫 번째 인자 [a, b] : "이 파라미터들을 네가 관리해라"라는 목록입니다.
#                       옵티마이저는 이 목록에 있는 것만 건드립니다.
# lr=lr              : 보폭(학습률)을 알려 줍니다.
# SGD = Stochastic Gradient Descent (확률적 경사하강법)
optimizer = optim.SGD([a, b], lr=lr)

for epoch in range(n_epochs):
    # ── 순전파 (변한 것 없음) ──────────────────────────
    yhat = a + b * x_train_tensor
    error = y_train_tensor - yhat
    loss = (error ** 2).mean()

    # ── 역전파 (변한 것 없음) ──────────────────────────
    loss.backward()

    # ── 갱신 : with torch.no_grad(): 블록이 통째로 사라졌습니다 ──
    # step() 이 등록된 모든 파라미터를 한꺼번에, 동시에 갱신합니다.
    # 내부에서 no_grad 처리까지 알아서 해 주므로 신경 쓸 필요가 없습니다.
    optimizer.step()

    # ── 기울기 초기화 : a.grad.zero_() 를 일일이 부르지 않아도 됩니다 ──
    # 등록된 모든 파라미터의 .grad 를 한 번에 0으로 만듭니다.
    optimizer.zero_grad()

print(a, b)


# %% [Block 5] loss_fn.py
import torch.nn as nn   # nn = neural network. 모델 부품과 손실 함수가 들어 있습니다.

# ── 손실 함수를 '만들어' 둡니다 ────────────────────────
# 이 줄은 손실값을 계산하는 것이 아니라,
# "평균을 내는 방식의 MSE 계산기"를 하나 제작해 두는 것입니다.
# 반복문 밖에 두는 이유: 매번 새로 만들 필요가 없기 때문입니다.
loss_fn = nn.MSELoss(reduction='mean')

optimizer = optim.SGD([a, b], lr=lr)

for epoch in range(n_epochs):
    # ── 예측 ────────────────────────────────────────────
    yhat = a + b * x_train_tensor

    # ── 손실 : error 를 직접 만들 필요가 없어졌습니다 ────
    # 앞서 만들어 둔 계산기에 값을 넣기만 합니다.
    # 아래 두 줄이 이 한 줄로 대체되었습니다:
    #     error = y_train_tensor - yhat
    #     loss  = (error ** 2).mean()
    loss = loss_fn(y_train_tensor, yhat)

    # ── 나머지는 그대로 ─────────────────────────────────
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


# %% [Block 6] manual_model.py
import torch
import torch.nn as nn

# class 이름은 PascalCase (첫 글자 대문자, 띄어쓰기 없이 붙여쓰기)
# 괄호 안의 nn.Module 은 "이 클래스를 물려받는다"는 뜻입니다.
class ManualLinearRegressor(nn.Module):
    """
    y = a + b·x 를 학습하는 가장 단순한 모델입니다.

    비유:
    자판기를 하나 조립하는 것과 같습니다.
    __init__ 에서 부품(손잡이 a, b)을 달고,
    forward 에서 "동전을 넣으면 무엇이 나오는지"를 정합니다.
    """

    def __init__(self):
        # super().__init__() : 부모(nn.Module)의 준비 작업을 먼저 실행합니다.
        # 이 줄을 빠뜨리면 파라미터 등록이 되지 않아 학습이 전혀 되지 않습니다.
        # 반드시 맨 첫 줄에 써야 합니다.
        super().__init__()

        # nn.Parameter 로 감싸면 "이건 학습할 값이야"라고 등록됩니다.
        # 그냥 텐서로 두면 model.parameters() 목록에 잡히지 않아
        # 옵티마이저가 갱신해 주지 않습니다. (조용히 학습이 안 되는 버그!)
        # self. 를 붙이는 이유: 이 객체에 계속 보관해 두기 위해서입니다.
        self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x):
        # 입력 x 를 받아 예측을 돌려줍니다. 우리의 가설 함수 그대로입니다.
        # 이 메서드는 직접 부르지 않습니다. model(x) 라고 쓰면 자동 호출됩니다.
        return self.a + self.b * x


# %% [Block 7] train_with_model.py
# 모델을 만들고 계산할 장치로 보냅니다.
# model은 data와 같은 device에 있어야 합니다. 안 그러면 에러가 납니다.
model = ManualLinearRegressor().to(device)

# state_dict() : 모델이 가진 학습 가능한 값들을 사전(dict) 형태로 보여 줍니다.
#                모델 저장·복원에도 쓰이는 중요한 메서드입니다.
print(model.state_dict())

loss_fn = nn.MSELoss(reduction='mean')

# model.parameters() : a, b 를 일일이 적을 필요 없이
#                      모델이 가진 학습 대상을 통째로 넘겨줍니다.
#                      파라미터가 1억 개여도 이 한 줄입니다.
optimizer = optim.SGD(model.parameters(), lr=lr)

for epoch in range(n_epochs):
    # train() : 모델을 '훈련 모드'로 설정합니다.
    # 주의 — 이 메서드는 training을 직접 하지는 않습니다!
    # Dropout 등이 있는 경우 training과 evaluation의 동작이 다르므로
    # 지금 어느 쪽인지 알려 주는 '스위치' 역할만 합니다.
    model.train()

    # model(x) 라고 쓰면 내부적으로 forward(x) 가 호출됩니다.
    yhat = model(x_train_tensor)

    loss = loss_fn(yhat, y_train_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


# %% [Block 8] nested_and_sequential.py
# ══════ ① Nested Model : 이미 만들어진 부품을 가져다 쓰기 ══════
class LayerLinearRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        # nn.Linear 가 a(bias)와 b(weight)를 알아서 만들어 줍니다.
        # in_features=1  : 입력이 숫자 1개 (집 넓이 하나)
        # out_features=1 : 출력도 숫자 1개 (집값 하나)
        # 나중에 특징이 4개로 늘면 in_features=4 로 바꾸기만 하면 됩니다.
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        # self.linear(x) 가 내부에서 weight·x + bias 를 계산해 줍니다.
        return self.linear(x)

# ══════ ② Sequential Model : class 를 만들 필요조차 없음 ══════
# nn.Sequential 은 "괄호 안의 부품들을 순서대로 통과시켜라"는 뜻입니다.
# 컨베이어 벨트에 기계를 순서대로 세워 두는 것과 같습니다.
model = nn.Sequential(nn.Linear(in_features=1, out_features=1)).to(device)


# %% [Block 9] train_step.py
def make_train_step(model, loss_fn, optimizer):
    """
    "한 걸음 학습하는 함수"를 만들어서 돌려주는 함수입니다.

    비유:
    붕어빵 틀을 찍어 내는 공장과 같습니다.
    모델·손실함수·옵티마이저를 넣어 주면,
    그 셋을 기억하는 전용 학습 함수가 하나 만들어져 나옵니다.
    """

    # 함수 안에 함수를 정의합니다. 이 안쪽 함수는 바깥의
    # model, loss_fn, optimizer 를 "기억"합니다. (클로저라고 부릅니다)
    def train_step(x, y):
        # Sets model to train mode
        # 모델을 훈련 모드로 전환합니다 (Dropout 등의 동작을 위해).
        model.train()

        # Make predictions
        # 예측합니다. forward 가 자동 호출됩니다.
        yhat = model(x)

        # Compute loss
        # 예측과 정답을 비교해 "얼마나 나쁜지" 점수를 냅니다.
        loss = loss_fn(yhat, y)

        # Compute gradients
        # 손실에서 출발해 모든 파라미터의 기울기를 자동 계산합니다.
        loss.backward()

        # Update parameters and zero gradients
        # 갱신 → 기울기 초기화. 순서를 바꾸면 안 됩니다!
        optimizer.step()
        optimizer.zero_grad()

        # Return the loss
        # .item() 은 값이 하나뿐인 텐서에서 "순수 파이썬 숫자"를 꺼냅니다.
        # 텐서째로 모아 두면 계산 그래프가 함께 쌓여 메모리가 새어 나갑니다.
        return loss.item()

    # 함수를 "실행"하는 게 아니라 "돌려줍니다". 괄호가 없다는 점에 주목!
    return train_step


# %% [Block 10] use_train_step.py
# ── 준비물 세 가지 ─────────────────────────────────────
model = nn.Sequential(nn.Linear(in_features=1, out_features=1)).to(device)
loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=lr)

# ── 전용 학습 함수를 하나 찍어 냅니다 ──────────────────
train_step = make_train_step(model, loss_fn, optimizer)

# 손실을 기록해 둘 빈 목록. 나중에 그래프로 그릴 수 있습니다.
losses = []

# ── 학습 루프가 단 두 줄로 줄었습니다 ──────────────────
for epoch in range(n_epochs):
    loss = train_step(x_train_tensor, y_train_tensor)
    losses.append(loss)   # 목록 끝에 추가


# %% [Block 11] custom_dataset.py
from torch.utils.data import Dataset, TensorDataset

class CustomDataset(Dataset):
    """
    x 와 y 를 짝지어 보관하는 상자입니다.

    비유:
    도서관의 책장과 같습니다.
    책장 번호를 말하면(index) 그 자리의 책을 꺼내 주고(__getitem__),
    "책이 몇 권이냐"고 물으면 총 권수를 알려 줍니다(__len__).
    """

    def __init__(self, x_tensor, y_tensor):
        # 받은 두 텐서를 그대로 보관해 둡니다.
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        # 이 던더 메서드가 있으면 dataset[3] 처럼 대괄호로 꺼낼 수 있습니다.
        # (특징, 정답) 을 한 쌍의 튜플로 돌려줍니다.
        return (self.x[index], self.y[index])

    def __len__(self):
        # 이 던더 메서드가 있으면 len(dataset) 이 동작합니다.
        # DataLoader 가 "몇 번 반복해야 하는지" 알아내는 데 씁니다.
        return len(self.x)

# ── 사용하기 ────────────────────────────────────────────
# .float() : 자료형을 float32 로 통일합니다.
#            NumPy 기본은 float64인데 딥러닝은 float32가 표준입니다.
x_train_tensor = torch.from_numpy(x_train).float()
y_train_tensor = torch.from_numpy(y_train).float()

train_data = CustomDataset(x_train_tensor, y_train_tensor)
print(train_data[0])   # 0번 데이터의 (x, y) 쌍

# ── 더 간단한 방법 : TensorDataset ─────────────────────
# Dataset이 tensor 몇 개에 불과하다면
# 새 class를 선언하는 과정 없이 TensorDataset 을 쓰는 편이 간단합니다.
train_data = TensorDataset(x_train_tensor, y_train_tensor)
print(train_data[0])


# %% [Block 12] dataloader.py
from torch.utils.data import DataLoader

# ── 로더 만들기 ────────────────────────────────────────
# dataset    : 어떤 데이터를 나눌지
# batch_size : 한 번에 몇 개씩 꺼낼지 (16개씩 → 80개면 5묶음)
# shuffle    : 매 epoch마다 순서를 섞을지
#              섞지 않으면 모델이 "데이터 순서"를 외워버릴 수 있습니다.
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)

# ── 미니배치 하나만 꺼내 확인해 보기 ───────────────────
# iter() : 로더를 "순서대로 꺼낼 수 있는 상태"로 만듭니다.
# next() : 거기서 하나를 꺼냅니다.
# 디버깅할 때 아주 유용한 습관입니다.
print(next(iter(train_loader)))

# ── 미니배치 학습 루프 ─────────────────────────────────
for epoch in range(n_epochs):

    # 안쪽 for 문이 하나 늘었습니다.
    # 로더를 for 문에 넣으면 미니배치를 하나씩 꺼내 줍니다.
    # (x_batch, y_batch) 튜플이 자동으로 두 변수에 나뉘어 들어갑니다.
    for x_batch, y_batch in train_loader:

        # 바로 여기서 device 로 보냅니다.
        # 전체가 아니라 "한 번에 한 mini-batch만" 보내므로
        # GPU 메모리를 훨씬 적게 씁니다.
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # 6단원에서 만든 학습 함수를 그대로 재사용합니다.
        # 전체 데이터를 넣든 미니배치를 넣든 함수는 똑같이 동작합니다.
        loss = train_step(x_batch, y_batch)
        losses.append(loss)


# %% [Block 13] random_split.py
from torch.utils.data.dataset import random_split

# ── 전체 데이터로 시작합니다 ───────────────────────────
# 주의: x_train 이 아니라 x (전체) 입니다.
# 나누는 일 자체를 PyTorch에게 맡기는 것이므로,
# 미리 나눠 둔 데이터를 넣으면 안 됩니다.
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

dataset = TensorDataset(x_tensor, y_tensor)

# ── 80 : 20 으로 무작위 분할 ───────────────────────────
# [80, 20] 의 합이 전체 개수(100)와 같아야 합니다. 다르면 에러가 납니다.
# 왼쪽부터 순서대로 train, val 에 담깁니다.
train_dataset, val_dataset = random_split(dataset, [80, 20])

# ── 각각의 로더를 만듭니다 ─────────────────────────────
# 훈련용: 16개씩 나눠서 학습
train_loader = DataLoader(dataset=train_dataset, batch_size=16)

# 검증용: 20개를 한 번에. 학습하지 않고 점수만 재므로 나눌 이유가 없습니다.
val_loader   = DataLoader(dataset=val_dataset, batch_size=20)


# %% [Block 14] 다. 구현 문제 — no_grad() : 검증에서는 기울기가 필요 없으므로 녹화를 끕니다.
# no_grad() : 검증에서는 기울기가 필요 없으므로 녹화를 끕니다.
with torch.no_grad():
    val_losses = []

    for x_val, y_val in val_loader:
        x_val = x_val.to(device)
        y_val = y_val.to(device)

        # eval() : 평가 모드로 전환 (Dropout 끄기 등)
        model.eval()

        yhat = model(x_val)
        val_loss = loss_fn(yhat, y_val)
        val_losses.append(val_loss.item())

    # backward() 도, step() 도 없다는 점이 핵심입니다.
    # 검증 데이터로는 절대 학습하지 않습니다.
