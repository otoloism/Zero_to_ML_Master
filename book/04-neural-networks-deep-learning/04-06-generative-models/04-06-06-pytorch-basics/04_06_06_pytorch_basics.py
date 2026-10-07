# -*- coding: utf-8 -*-
"""
04-06-06 신경망 —PyTorch기초

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-06장 이미지 생성 모델의 원리/04-06-06 신경망 —PyTorch기초.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-06-06-B 🟢 초급 ⭐ · PyTorch 텐서(Tensor)
import torch
import numpy as np

# 1) 텐서 생성 — numpy 배열과 문법이 거의 동일하다
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])   # 2x2 행렬 텐서
print(x.shape)   # torch.Size([2, 2]) — numpy의 .shape와 동일한 개념
print(x.dtype)   # torch.float32 — 기본 실수 타입

# 2) 기본 연산 — 사칙연산, 행렬곱 모두 numpy와 동일한 감각
y = x + 1
z = x @ x   # @ 연산자: 행렬곱 (np.matmul과 동일)

# 3) GPU로 이동 — DeZero에는 없던, PyTorch가 주는 가장 큰 실무적 이득
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)   # GPU가 있으면 GPU 메모리로, 없으면 그대로 CPU에

# 4) numpy로 되돌리기 — 시각화(matplotlib)나 저장 시 자주 사용
x_np = x.cpu().numpy()   # GPU에 있다면 먼저 cpu()로 내려온 뒤 변환해야 한다


# %% [Block 2] 04-06-06-C 🔵 중급 ⭐ · 자동 미분(Autograd)
import torch

# requires_grad=True: "이 텐서에 대한 미분을 계속 추적해줘"라는 등록
x = torch.tensor(2.0, requires_grad=True)

# 순전파: y = x^2 + 3x + 1  (이 연산 과정이 자동으로 그래프에 기록된다)
y = x ** 2 + 3 * x + 1

# 역전파: dy/dx 를 계산하라는 명령. DeZero의 backward()와 완전히 동일한 개념.
y.backward()

print(x.grad)   # dy/dx = 2x + 3 = 2*2 + 3 = 7.0


# %% [Block 3] 04-06-06-D 🔵 중급 · 선형 회귀를 PyTorch로
import torch

# 가짜 데이터: y = 2x + 1 에 약간의 노이즈
torch.manual_seed(0)
x = torch.linspace(0, 1, 100).unsqueeze(1)
y = 2 * x + 1 + 0.1 * torch.randn(100, 1)

# 학습 대상 파라미터 W, b — requires_grad=True로 자동 미분 대상 등록
W = torch.zeros(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.1
for epoch in range(100):
    y_pred = x @ W + b                 # 순전파: 예측값 계산
    loss = ((y_pred - y) ** 2).mean()   # MSE 손실

    loss.backward()                    # 역전파: W.grad, b.grad 자동 계산

    with torch.no_grad():          # 파라미터 갱신은 그래프 기록에서 제외
        W -= lr * W.grad
        b -= lr * b.grad
        W.grad.zero_()                 # 기울기 초기화 — 안 하면 누적되어 오차 발생!
        b.grad.zero_()

    if epoch % 20 == 0:
        print(f"epoch {epoch}, loss={loss.item():.4f}")


# %% [Block 4] 04-06-06-E 🔵 중급 · 옵티마이저 (torch.optim.Adam)
import torch.optim as optim

W = torch.zeros(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 옵티마이저에 "누구를 갱신할지"(파라미터)와 학습률을 알려준다
optimizer = optim.Adam([W, b], lr=0.1)

for epoch in range(100):
    y_pred = x @ W + b
    loss = ((y_pred - y) ** 2).mean()

    optimizer.zero_grad()   # W.grad.zero_(), b.grad.zero_()를 한 번에
    loss.backward()         # 기울기 계산
    optimizer.step()        # W -= lr*..., b -= lr*... 을 Adam 공식으로 자동 수행


# %% [Block 5] 04-06-06-F 🔵 중급 ⭐ · nn.Module — 신경망을 클래스로
import torch.nn as nn

class SimpleClassifier(nn.Module):
    """
    MNIST 분류를 위한 아주 단순한 2층 신경망.

    비유:
    입력(28x28 이미지 픽셀)을 일렬로 펴서 컨베이어 벨트에 올린 뒤,
    두 개의 작업대(층)를 거치며 숫자 0~9 중 하나로 분류될 때까지 다듬는다.
    """
    def __init__(self):
        super().__init__()   # nn.Module의 초기화를 먼저 실행 (필수)
        self.fc1 = nn.Linear(28 * 28, 128)   # 입력층 → 은닉층 (784 → 128)
        self.relu = nn.ReLU()                # 비선형 활성화 함수
        self.fc2 = nn.Linear(128, 10)     # 은닉층 → 출력층 (128 → 10개 클래스)

    def forward(self, x):
        x = x.view(x.size(0), -1)   # (batch, 28, 28) → (batch, 784)로 평탄화
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x   # 최종 점수(logits) 반환 — softmax는 손실 함수 안에서 처리

model = SimpleClassifier()
print(model)


# %% [Block 6] 04-06-06-G 🔴 심화 🚀 · torchvision과 MNIST 분류 전체 파이프라인
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1) 변환(transform): 이미지를 텐서로 바꾸고 픽셀값을 정규화
transform = transforms.Compose([
    transforms.ToTensor(),                      # PIL 이미지 → [0,1] 범위 텐서
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 전체 평균·표준편차로 정규화
])

# 2) 데이터셋 — 처음 실행 시 자동으로 다운로드
train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# 3) 데이터로더 — 배치 단위로 잘라서, 매 에폭 순서를 섞어 공급
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

model = SimpleClassifier()
criterion = nn.CrossEntropyLoss()   # 분류 문제 표준 손실 함수 (softmax + NLL을 합친 것)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)          # 순전파 — 내부적으로 forward() 실행
        loss = criterion(outputs, labels)
        loss.backward()                  # 역전파
        optimizer.step()                 # 파라미터 갱신

    print(f"epoch {epoch}, loss={loss.item():.4f}")
