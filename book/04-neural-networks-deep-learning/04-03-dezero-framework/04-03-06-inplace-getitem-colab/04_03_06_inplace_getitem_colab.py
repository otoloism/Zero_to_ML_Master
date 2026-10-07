# -*- coding: utf-8 -*-
"""
04-03-06 인플레이스 연산 · get_item 함수 · 구글 콜랩

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-03장 딥러닝 프레임워크 — DeZero 직접 만들기/04-03-06 인플레이스 연산 · get_item 함수 · 구글 콜랩.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-03-06-A 52단계 — GPU 지원: "같은 코드, 다른 백엔드" — 52단계: NumPy ↔ CuPy 자동 전환
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 52단계: NumPy ↔ CuPy 자동 전환
# "같은 요리법, 다른 주방(CPU vs GPU)"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import numpy as np

# GPU가 없는 환경에서도 코드가 멈추지 않도록 try-except로 감쌉니다.
# CuPy가 없으면 cp = np 로 대체해서, 이름만 cp인 CPU 모드로 동작합니다.
try:
    import cupy as cp       # GPU용 NumPy 쌍둥이 라이브러리
    gpu_enable = True       # GPU 사용 가능
except ImportError:
    cp = np                 # GPU 없음 → numpy로 대체
    gpu_enable = False      # CPU 모드

def get_array_module(x):
    """배열 x가 NumPy인지 CuPy인지 자동 감지해 알맞은 모듈을 반환.

    비유: "지금 가스레인지(CPU)야, 인덕션(GPU)야?"를 자동 판별하는 센서.
    이 한 줄만 forward 안에 추가하면 CPU/GPU 모두 동작합니다.
    """
    if not gpu_enable:
        return np
    return cp.get_array_module(x)   # CuPy 배열 → cp, NumPy 배열 → np

class Variable:
    """DeZero의 데이터 상자. data에 실제 값(배열)을 담습니다."""

    def __init__(self, data):
        self.data = data    # NumPy 또는 CuPy 배열
        self.grad = None    # 기울기 저장소

    def to_gpu(self):
        """CPU 메모리 → GPU 메모리 이동.
        비유: 일반 도로에서 고속도로 진입로로 올라타는 것."""
        if self.data is not None and gpu_enable:
            self.data = cp.asarray(self.data)   # numpy → cupy
        return self

    def to_cpu(self):
        """GPU 메모리 → CPU 메모리 이동.
        비유: 고속도로에서 일반 도로로 내려오는 것."""
        if self.data is not None:
            self.data = np.asarray(self.data)   # cupy → numpy
        return self

# ── 핵심: forward 안에서 get_array_module 한 줄만 추가 ──────────
class Square:
    def forward(self, x):
        xp = get_array_module(x)    # ← 이 한 줄이 CPU/GPU 자동 전환의 핵심!
        return xp.square(x)         # np.square 또는 cp.square 자동 선택

# ── 사용 예시 ──────────────────────────────────────────────────
x = Variable(np.array([1.0, 2.0, 3.0]))    # CPU 데이터
x.to_gpu()                                   # GPU로 이동
print("GPU 사용 가능?", gpu_enable)
x.to_cpu()
print(x.data)


# %% [Block 2] 04-03-06-B 53단계 — 모델 저장/읽기: "시험 공부 결과를 파일에 적기"
import numpy as np

def save_weights(model, path):
    """model 안의 모든 Parameter 값을 .npz 압축 파일로 저장합니다.

    비유: 오랜 공부(학습)로 얻은 지식(가중치)을 노트(파일)에 적어두는 것.
          다음에는 처음부터 다시 공부하지 않고 노트만 펼치면 됩니다.
    """
    params_dict = {}
    for i, param in enumerate(model.params()):
        # param 자체가 아닌 실제 숫자 배열(param.data)만 저장합니다.
        params_dict[f'param{i}'] = param.data

    np.savez(path, **params_dict)    # .npz 파일로 압축 저장
    print(f"✅ {len(params_dict)}개 파라미터 → '{path}' 저장 완료")

def load_weights(model, path):
    """save_weights로 저장했던 파일을 불러와 model에 복원합니다.

    비유: 노트에 적어둔 지식을 다시 머릿속(모델)으로 옮기는 것.
    """
    npz = np.load(path)
    for i, param in enumerate(model.params()):
        param.data = npz[f'param{i}']    # 순서대로 가중치 덮어쓰기
    print(f"✅ '{path}' → 파라미터 복원 완료")

# ── 사용 예시 ──────────────────────────────────────────────────
save_weights(model, 'my_model.npz')        # 학습 완료 후 저장
model_new = MyModel()                      # 같은 구조의 새 모델
load_weights(model_new, 'my_model.npz')    # 가중치 복원
y = model_new(x_test)                      # 바로 추론(inference) 실행!


# %% [Block 3] 04-03-06-C 54단계 — Dropout: "학습 때만 일부러 일부 뉴런을 끄기"
import numpy as np
import dezero

def dropout(x, dropout_ratio=0.5):
    """학습 중에만 일부 뉴런의 출력을 0으로 만드는 정규화 기법.

    x             : 입력 데이터 (Variable)
    dropout_ratio : 꺼버릴 뉴런의 비율 (0.5 = 절반을 끔)

    비유: 축구팀이 매 연습 경기마다 다른 선수를 무작위로 쉬게 합니다.
          특정 스타 선수에게만 의존하지 않게 됩니다.
          실제 시합(추론)에서는 전원이 출전합니다.
    """
    if dezero.Config.train:
        # ① 랜덤 마스크 생성: dropout_ratio보다 큰 것만 True(살아남음)
        mask = np.random.rand(*x.shape) > dropout_ratio

        # ② Inverted Dropout: 켜진 뉴런 출력을 스케일 보정
        # → 추론 때 아무 처리 없어도 기대값이 동일하게 유지됩니다
        scale = 1 - dropout_ratio

        return x * mask / scale   # mask 적용 + 스케일 보정
    else:
        return x   # 추론 모드: 모든 뉴런을 그대로 사용

# ── ReLU mask vs Dropout mask 비교 ──────────────────────────────
# ┌───────────────┬──────────────────────────┬──────────────────────────┐
# │  구분         │ mask 결정 방식           │ 적용 시점                │
# ├───────────────┼──────────────────────────┼──────────────────────────┤
# │ ReLU mask     │ 입력값 < 0 인 위치       │ 항상 (학습·추론 모두)    │
# │ Dropout mask  │ 매번 무작위              │ 학습 시만 (추론 때는 OFF)│
# └───────────────┴──────────────────────────┴──────────────────────────┘

dezero.Config.train = True    # 학습 시: 랜덤 뉴런 OFF
dezero.Config.train = False   # 추론 시: 모든 뉴런 ON


# %% [Block 4] 04-03-06-D 55~57단계 — CNN: im2col과 Conv2d
from dezero.utils import im2col_array, col2im_array

class Conv2d:
    """합성곱 연산을 하나의 행렬곱으로 계산하는 클래스.

    핵심:
    - im2col  : 슬라이딩 창들을 2D 행렬로 변환 (4중 for문 → 1번으로!)
    - dot     : 행렬 곱셈 한 번으로 합성곱 완성
    - col2im  : im2col의 역연산 (역전파에서 기울기를 원래 위치로 복원)
    """

    def __init__(self, stride=1, pad=0):
        self.stride = stride    # 창을 몇 칸씩 이동할지 (기본 1칸)
        self.pad    = pad       # 이미지 테두리에 0을 얼마나 덧붙일지

    def forward(self, x, W, b):
        KH, KW = W.shape[2], W.shape[3]

        # ① im2col: 슬라이딩 창 영역들을 2D 행렬로 펼치기
        col = im2col_array(x, (KH, KW), self.stride, self.pad)

        # ② 필터도 2D로 펼치기: (출력채널, 입력채널×KH×KW)
        Weight = W.reshape(W.shape[0], -1)

        # ③ 행렬 곱셈 한 번으로 합성곱 전체를 완성!
        y = col.dot(Weight.T)
        if b is not None:
            y += b    # 편향 추가

        # ④ 출력을 이미지 형태로 복원
        N, C, H, W_ = x.shape
        OH = (H + 2 * self.pad - KH) // self.stride + 1
        OW = (W_ + 2 * self.pad - KW) // self.stride + 1
        y  = y.reshape(N, OH, OW, -1).transpose(0, 3, 1, 2)

        self.x_shape, self.col, self.W = x.shape, col, W
        return y

    def backward(self, gy):
        x_shape, W = self.x_shape, self.W
        KH, KW = W.shape[2], W.shape[3]

        # gx: col2im으로 기울기를 원래 이미지 위치로 복원
        gcol = gy.transpose(0, 2, 3, 1).reshape(-1, W.shape[0]).dot(
            W.reshape(W.shape[0], -1))
        gx = col2im_array(gcol, x_shape, (KH, KW), self.stride, self.pad)

        # gW: MatMul.backward와 같은 패턴 (04-01-04 Affine 역전파)
        gy2 = gy.transpose(0, 2, 3, 1).reshape(-1, W.shape[0])
        gW  = self.col.T.dot(gy2).T.reshape(W.shape)

        # gb: 배치·높이·너비 축을 모두 더해 채널별 합을 구함
        gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb


# %% [Block 5] 04-03-06-E 55단계 — Pooling: "각 구역에서 가장 강한 신호만 남기기"
import numpy as np

class Pooling:
    """최댓값만 골라내는 맥스 풀링(Max Pooling) 연산.

    비유: 신문 각 문단에서 가장 중요한 키워드 하나만 형광펜으로 칠하는 것.
          세부 내용은 버리지만 핵심은 살아남습니다.
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size    # 풀링 창 크기 (예: 2×2)
        self.stride      = stride         # 창 이동 간격

    def forward(self, x):
        from dezero.utils import im2col_array
        N, C, H, W = x.shape
        KH = KW = self.kernel_size

        # im2col로 풀링 영역을 행렬로 변환
        col = im2col_array(x, (KH, KW), self.stride, 0, to_matrix=False)
        col = col.reshape(N, C, KH * KW, -1)

        # 최댓값의 위치(인덱스)를 저장 → 역전파에서 사용
        self.indexes = col.argmax(axis=2)
        y = col.max(axis=2)    # 각 영역에서 최댓값만 선택
        return y

    def backward(self, gy):
        """역전파: 최댓값이었던 위치에만 기울기를 전달, 나머지는 0.

        DeZero 전체를 관통하는 패턴:
        "순전파에서 어디를 골랐는지 기억 → 역전파에서 그 자리에만 기울기 전달"
        (ReLU mask, GetItem.backward와 동일한 원리)
        """
        pool_size = self.kernel_size * self.kernel_size
        gy_flat   = gy.flatten()

        # 0으로 채운 배열에 최댓값 위치에만 기울기를 넣음
        dcol = np.zeros((gy_flat.size, pool_size))
        dcol[np.arange(gy_flat.size), self.indexes.flatten()] = gy_flat
        return dcol   # col2im으로 원래 이미지 형태로 복원


# %% [Block 6] 04-03-06-F 58단계 — VGG16: "3×3 레고 블록을 16층으로 쌓기"
from dezero.models import Model
from dezero.layers import Conv2d, Linear
from dezero.functions import relu, pooling, reshape, dropout

class VGG16(Model):
    """VGG16: 2014년 ImageNet 준우승 모델.

    비유: 같은 규격의 레고 블록 13개를 쌓고,
          꼭대기에 판단 블록 3개를 얹은 16층 고층 빌딩.
          아래층은 단순한 선·색을, 위층은 고양이 얼굴 같은
          복잡한 특징을 인식합니다.
    """
    def __init__(self):
        super().__init__()
        self.conv1_1 = Conv2d(64,  kernel_size=3, stride=1, pad=1)
        self.conv1_2 = Conv2d(64,  kernel_size=3, stride=1, pad=1)
        self.conv2_1 = Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv2_2 = Conv2d(128, kernel_size=3, stride=1, pad=1)
        # ... conv3_x, conv4_x, conv5_x 도 같은 방식으로 이어집니다 ...
        self.fc6 = Linear(4096)     # 특징 맵 → 4096
        self.fc7 = Linear(4096)     # 4096 → 4096
        self.fc8 = Linear(1000)     # 4096 → 1000개 클래스 (ImageNet)

    def forward(self, x):
        x = relu(self.conv1_1(x))
        x = relu(self.conv1_2(x))
        x = pooling(x, 2, 2)        # 2×2 최대 풀링 → 크기 절반

        x = relu(self.conv2_1(x))
        x = relu(self.conv2_2(x))
        x = pooling(x, 2, 2)
        # ... Block 3~5도 같은 패턴 ...

        # Flatten: (배치, 채널, 높이, 너비) → (배치, 채널×높이×너비)
        x = reshape(x, (x.shape[0], -1))

        x = dropout(relu(self.fc6(x)))   # Dropout으로 과적합 방지
        x = dropout(relu(self.fc7(x)))
        x = self.fc8(x)    # 마지막은 활성화 없이 점수(logit)만 출력
        return x            # softmax는 손실 함수 안에서 처리


# %% [Block 7] 04-03-06-G 59단계 — RNN: "매일 일기장을 이어 쓰는 신경망"
from dezero.layers import Layer, Linear
from dezero.functions import tanh

class RNN(Layer):
    """순환 신경망(RNN) 계층.

    핵심 공식: h_t = tanh(Wx·x_t + Wh·h_{t-1} + b)
    비유: 매일 일기장을 이어 쓰는 학생.
          오늘 경험(x_t)과 어제 일기(h_{t-1})를 합쳐 새 일기(h_t)를 씁니다.
    """

    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        self.x2h = Linear(hidden_size, in_size=in_size)
        self.h2h = Linear(hidden_size, in_size=hidden_size, nobias=True)
        self.h = None    # 이전 시각의 기억

    def reset_state(self):
        """새 시퀀스 시작 시 기억 초기화."""
        self.h = None

    def forward(self, x):
        if self.h is None:
            h_new = tanh(self.x2h(x))
        else:
            # 04-04-05의 그 식: h_t = tanh(Wx·x_t + Wh·h_{t-1} + b)
            h_new = tanh(self.x2h(x) + self.h2h(self.h))
        self.h = h_new    # 새 기억 저장
        return h_new

# ── 사인파 예측 학습 루프 ─────────────────────────────────────────
import math, numpy as np
from dezero import Variable
from dezero.models import Model
from dezero.optimizers import Adam
from dezero.functions import mean_squared_error

class SimpleRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = RNN(hidden_size)
        self.fc  = Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, x):
        h = self.rnn(x)
        return self.fc(h)

x_data  = np.array([math.sin(2 * math.pi * i / 100) for i in range(1000)])
model   = SimpleRNN(hidden_size=10, out_size=1)
optimizer = Adam().setup(model)

for epoch in range(100):
    model.reset_state()    # 새 시퀀스 시작: 기억 초기화
    loss = 0
    for t in range(len(x_data) - 1):
        x      = Variable(np.array(x_data[t]).reshape(1, 1))
        y_true = Variable(np.array(x_data[t + 1]).reshape(1, 1))
        y_pred = model(x)
        loss   = loss + mean_squared_error(y_true, y_pred)   # 시점별 손실 누적

    model.cleargrads()
    loss.backward()     # 누적 손실을 한 번에 역전파 → BPTT 자동 수행!
    optimizer.update()

    if epoch % 10 == 0:
        print(f"epoch {epoch}: loss = {loss.data:.4f}")


# %% [Block 8] 04-03-06-H 60단계 — LSTM: "3개의 문이 달린 장기 기억 노트"
from dezero.layers import Layer, Linear
from dezero.functions import sigmoid, tanh

class LSTM(Layer):
    """장기 단기 기억(LSTM) 계층. 04-04-06 코드를 DeZero로 재구현.

    비유: 시험공부 노트 정리법
    - c (기억 셀) = 노트 전체 내용
    - f (망각 게이트) = 시험에 안 나오는 내용 지우기
    - i (입력 게이트) = 중요한 새 내용 추가하기
    - o (출력 게이트) = 지금 문제에 필요한 것만 꺼내 쓰기
    """

    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        H, I = hidden_size, in_size
        self.x2f = Linear(H, in_size=I);  self.h2f = Linear(H, in_size=H, nobias=True)
        self.x2i = Linear(H, in_size=I);  self.h2i = Linear(H, in_size=H, nobias=True)
        self.x2o = Linear(H, in_size=I);  self.h2o = Linear(H, in_size=H, nobias=True)
        self.x2u = Linear(H, in_size=I);  self.h2u = Linear(H, in_size=H, nobias=True)
        self.h = None    # 단기 기억 (은닉 상태)
        self.c = None    # 장기 기억 (기억 셀) ← RNN과의 핵심 차이!

    def reset_state(self):
        self.h = None;  self.c = None

    def forward(self, x):
        if self.h is None:
            f = sigmoid(self.x2f(x));  i = sigmoid(self.x2i(x))
            o = sigmoid(self.x2o(x));  u = tanh(self.x2u(x))
        else:
            f = sigmoid(self.x2f(x) + self.h2f(self.h))   # ① 얼마나 잊을까?
            i = sigmoid(self.x2i(x) + self.h2i(self.h))   # ② 얼마나 기억할까?
            o = sigmoid(self.x2o(x) + self.h2o(self.h))   # ③ 얼마나 내보낼까?
            u = tanh(self.x2u(x) + self.h2u(self.h))       # 기억 후보

        if self.c is None:
            c_new = i * u
        else:
            c_new = f * self.c + i * u   # 핵심: 일부 잊고, 일부 새로 기억

        h_new = o * tanh(c_new)    # 단기 기억 = 장기 기억을 출력 게이트로 걸러냄
        self.h = h_new;  self.c = c_new
        return h_new


# %% [Block 9] 🧪 부록 A — 인플레이스 연산: — "시험 채점 전에 답안지를 고치면 안 된다!" — ⚠️ 인플레이스 (위험): 원래 데이터가 사라짐!
# ⚠️ 인플레이스 (위험): 원래 데이터가 사라짐!
# x.data += 1  ← self.inputs[0]이 가리키는 데이터도 함께 바뀜

# ✅ 새 메모리 할당 (안전): 원래 x는 그대로!
# x = x + 1    ← 새로운 Variable이 만들어지고, 원래 x는 보존

# ── 구체적 예시: Square 함수 ──
class Square(Function):
    """입력을 제곱하는 함수.
    비유: '답안지 복사본을 만들어 두는 선생님' — 원본은 안전!"""

    def forward(self, x):
        return x ** 2                       # 새 배열 반환 (안전)

    def backward(self, gy):
        x = self.inputs[0]                # ← 순전파 때의 x를 다시 사용!
        return 2 * x * gy                  # d(x²)/dx = 2x


# %% [Block 10] 🧪 부록 B — get_item 함수: "선택한 자리에만 칭찬 돌려주기"
import numpy as np

class GetItem:
    """슬라이싱(x[0], x[:, 1:3] 등)을 미분 가능하게 만드는 함수.

    DeZero 전체를 관통하는 패턴:
    "순전파에서 어디를 골랐는지 기억 → 역전파에서 그 자리에만 기울기 전달"
    (ReLU mask, Pooling argmax와 완전히 동일한 원리)
    """

    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        self.x_shape = x.shape    # 역전파에서 원본 모양이 필요하므로 저장
        y = x[self.slices]        # 실제 슬라이싱 수행
        return y

    def backward(self, gy):
        # ① 원본과 같은 모양의 배열을 0으로 채우기
        gx = np.zeros(self.x_shape, dtype=gy.dtype)
        # ② 잘라냈던 위치에만 기울기를 채워 넣기 (나머지는 0)
        gx[self.slices] = gy
        return gx

# ── 사용 예시 ──────────────────────────────────────────────────
x = np.array([[1., 2., 3.],
              [4., 5., 6.]])

f  = GetItem(0)
y  = f.forward(x)
print("y =", y)         # [1. 2. 3.]

gy = np.array([1., 1., 1.])
gx = f.backward(gy)
print("gx =")
print(gx)
# [[1. 1. 1.]   ← 선택된 행: 기울기 채움
#  [0. 0. 0.]]  ← 선택 안 된 행: 0

g   = GetItem(slice(0, 1))
y2  = g.forward(x)
gy2 = np.array([[1., 1., 1.]])
gx2 = g.backward(gy2)
print(gx2)
# [[1. 1. 1.]
#  [0. 0. 0.]]


# %% [Block 11] ☁️ 부록 C — 구글 콜랩: "무료 GPU를 빌려 쓰는 클라우드 주방" — ① 먼저: 런타임 → 런타임 유형 변경 → 하드웨어 가속기를 GPU로 설정!
# ① 먼저: 런타임 → 런타임 유형 변경 → 하드웨어 가속기를 GPU로 설정!

# ② 느낌표(!)를 붙이면 파이썬이 아닌 터미널 명령을 실행합니다.
# !pip install dezero

# ③ 설치 확인
import dezero
print("DeZero 버전:", dezero.__version__)

# ④ GPU 사용 가능 여부 확인
import cupy as cp
print("GPU 사용 가능?", cp.cuda.is_available())
# → True 라고 나오면 준비 완료!

# ⑤ DeZero Variable을 GPU로 실행해 보기
import numpy as np
from dezero import Variable

x = Variable(np.array([1.0, 2.0, 3.0]))    # CPU 데이터 생성
x.to_gpu()                                   # GPU 메모리로 이동
y = x ** 2                                   # GPU 위에서 계산 (자동!)
y.to_cpu()                                   # 결과 확인용 CPU로 복귀
print(y.data)                                # [1. 4. 9.]

# ⚠️  세션이 끊기면 설치했던 것과 변수들이 모두 사라집니다.
#     다시 이어서 작업하려면 ② pip install부터 다시 실행해야 합니다.
# ⚠️  무료 GPU는 사용 시간 제한이 있습니다.
#     오래 걸리는 학습은 중간중간 save_weights로 저장해두는 것이 좋습니다.
