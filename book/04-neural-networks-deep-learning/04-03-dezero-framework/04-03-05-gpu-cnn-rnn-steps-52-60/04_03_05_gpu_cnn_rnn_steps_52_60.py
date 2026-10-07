# -*- coding: utf-8 -*-
"""
04-03-05 DeZero의 도전 (52 60단계) — GPU·CNN·RNN까지

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-03장 딥러닝 프레임워크 — DeZero 직접 만들기/04-03-05 DeZero의 도전 (52 60단계) — GPU·CNN·RNN까지.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 0] [추가] 책에서는 앞 절에서 import 한 모듈 — 단독 실행을 위해 보충 (원문에는 없음)
import numpy as np


# %% [Block 1] 52 GPU 지원 — "일반 도로에서 고속도로로 차선 변경"
gpu_enable = False
try:
    import cupy as cp       # CuPy: NumPy와 API가 같은 GPU 라이브러리
    gpu_enable = True
except ImportError:
    pass  # GPU 없으면 CPU 모드

def get_array_module(x):
    """배열이 GPU에 있으면 cupy, CPU에 있으면 numpy 반환.
    비유: "지금 고속도로야? 일반 도로야?" 자동 판별기"""
    if not gpu_enable:
        return np
    return cp.get_array_module(x)

class Variable:
    def to_gpu(self):
        """CPU → GPU 이동 (일반 도로 → 고속도로 진입)"""
        if self.data is not None:
            self.data = cp.asarray(self.data)
        return self

    def to_cpu(self):
        """GPU → CPU 이동 (고속도로 → 일반 도로 복귀)"""
        if self.data is not None:
            self.data = cp.asnumpy(self.data)
        return self

# 사용법: 모델과 데이터를 GPU로 옮기면 끝!
for param in model.params():
    param.to_gpu()
# 이후 학습 루프는 CPU 때와 완전히 동일합니다


# %% [Block 2] 53 모델 저장/읽기 — "게임 세이브/로드"
class Layer:
    def _flatten_params(self, params_dict, parent_key=''):
        """모든 파라미터를 'l1/W', 'l1/b' 형태로 펼쳐서 딕셔너리에 담음.
        비유: 서랍장 안의 모든 물건에 주소 라벨을 붙여 테이블에 늘어놓기"""
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name
            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)  # 재귀 탐색
            else:
                params_dict[key] = obj.data

    def save_weights(self, path):
        """모델 가중치를 .npz 파일로 저장 (게임 세이브)"""
        params_dict = {}
        self._flatten_params(params_dict)
        np.savez_compressed(path, **params_dict)

    def load_weights(self, path):
        """저장된 가중치를 불러와 복원 (게임 로드)"""
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param[...] = npz[key]    # [...] = 배열 내용만 교체

# 사용법
model.save_weights('my_model.npz')       # 저장
model2 = Model()
model2.load_weights('my_model.npz')      # 복원 → 바로 예측 가능!


# %% [Block 3] 54 Dropout — "랜덤 결석 훈련법"
class Config:
    train = True   # True=학습모드(Dropout ON), False=추론모드(Dropout OFF)

class Dropout(Function):
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio

    def forward(self, x):
        if Config.train:
            xp = get_array_module(x)
            # 랜덤 마스크: dropout_ratio보다 큰 것만 1, 나머지 0
            self.mask = xp.random.rand(*x.shape) > self.dropout_ratio
            # Inverted Dropout: 켜진 뉴런 출력을 스케일업 → 추론 때 보정 불필요
            scale = 1.0 / (1.0 - self.dropout_ratio)
            return x * self.mask * scale
        else:
            return x   # 추론 모드: 그대로 통과

    def backward(self, gy):
        scale = 1.0 / (1.0 - self.dropout_ratio)
        return gy * self.mask * scale   # 꺼진 뉴런은 기울기도 0

# 사용법
Config.train = True    # 학습 시: 50% 뉴런 랜덤 OFF
Config.train = False   # 추론 시: 모든 뉴런 ON


# %% [Block 4] 55 CNN 메커니즘 — "돋보기로 사진 훑기" — 55단계: 합성곱의 직관적 구현 (이해용)
# 55단계: 합성곱의 직관적 구현 (이해용)
def conv2d_naive(input_data, kernel, stride=1, pad=0):
    """돋보기(kernel)를 이미지 위에서 한 칸씩 이동하며 특징 추출"""
    N, C, H, W = input_data.shape
    FN, C, FH, FW = kernel.shape
    out_h = (H + 2*pad - FH) // stride + 1
    out_w = (W + 2*pad - FW) // stride + 1
    img = np.pad(input_data, [(0,0),(0,0),(pad,pad),(pad,pad)], mode='constant')
    output = np.zeros((N, FN, out_h, out_w))

    for n in range(N):           # 각 이미지
        for f in range(FN):      # 각 필터
            for i in range(out_h):    # 세로 이동
                for j in range(out_w):  # 가로 이동
                    h_s, w_s = i*stride, j*stride
                    region = img[n, :, h_s:h_s+FH, w_s:w_s+FW]  # 돋보기 영역
                    output[n, f, i, j] = np.sum(region * kernel[f])  # 곱하고 더함
    return output

# 풀링: 영역 안에서 가장 큰 값만 남김 (핵심 특징 요약)
def max_pooling_naive(x, pool_h, pool_w, stride=2):
    """신문 각 문단에서 가장 중요한 키워드 하나만 형광펜 칠하기"""
    N, C, H, W = x.shape
    out_h, out_w = (H - pool_h) // stride + 1, (W - pool_w) // stride + 1
    output = np.zeros((N, C, out_h, out_w))
    for n in range(N):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    h_s, w_s = i*stride, j*stride
                    output[n, c, i, j] = np.max(x[n, c, h_s:h_s+pool_h, w_s:w_s+pool_w])
    return output


# %% [Block 5] 56 im2col 함수 — "퍼즐 조각을 일렬로 세우기"
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """필터가 적용될 각 영역을 꺼내서 행(row)으로 편다.
    비유: 돋보기가 볼 모든 위치의 영역을 카드로 만들어 세로로 쌓기"""
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1
    img = np.pad(input_data, [(0,0),(0,0),(pad,pad),(pad,pad)], mode='constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for i in range(filter_h):
        i_max = i + stride * out_h
        for j in range(filter_w):
            j_max = j + stride * out_w
            col[:, :, i, j, :, :] = img[:, :, i:i_max:stride, j:j_max:stride]

    # (N, C, FH, FW, OH, OW) → (N*OH*OW, C*FH*FW) 로 reshape
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col
    # 이제 합성곱 = col.dot(kernel_col) ← 행렬 곱셈 하나!


# %% [Block 6] 57 conv2d / pooling — "CNN 연산을 Function으로 포장"
class Conv2d(Function):
    """합성곱 = im2col + 행렬곱 + 자동미분"""
    def __init__(self, stride=1, pad=0):
        self.stride = stride
        self.pad = pad

    def forward(self, x, W, b=None):
        FN, C, FH, FW = W.shape
        N, C, H, Width = x.shape
        out_h = (H + 2*self.pad - FH) // self.stride + 1
        out_w = (Width + 2*self.pad - FW) // self.stride + 1
        col = im2col(x, FH, FW, self.stride, self.pad)  # 핵심 트릭!
        W_col = W.reshape(FN, -1).T
        out = col.dot(W_col)             # 합성곱 = 행렬곱 하나!
        if b is not None:
            out += b
        out = out.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)
        self.x_shape, self.col, self.W_col = x.shape, col, W_col
        return out

    def backward(self, gy):
        """역전파: 입력/필터/편향의 기울기를 자동 계산"""
        W, = self.inputs[1:2]
        FN, C, FH, FW = W.shape
        gy_2d = gy.transpose(0,2,3,1).reshape(-1, FN)
        gb = gy_2d.sum(axis=0)                     # 편향 기울기
        gW = self.col.T.dot(gy_2d).T.reshape(FN,C,FH,FW)  # 필터 기울기
        gcol = gy_2d.dot(self.W_col.T)
        gx = col2im(gcol, self.x_shape, FH, FW, self.stride, self.pad)
        return gx, gW, gb

# Conv2d Layer: Linear처럼 가중치를 자체 보관
class Conv2dLayer(Layer):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, pad=0):
        super().__init__()
        self.W = Parameter(np.random.randn(out_ch, in_ch, kernel_size, kernel_size)
                           * np.sqrt(1.0 / (in_ch * kernel_size**2)))
        self.b = Parameter(np.zeros(out_ch))
        self.stride, self.pad = stride, pad
    def __call__(self, x):
        return Conv2d(self.stride, self.pad)(x, self.W, self.b)


# %% [Block 7] 58 VGG16 구현 — "16층 고층 빌딩 세우기"
class VGG16(Model):
    def __init__(self, pretrained=False):
        super().__init__()
        # Block 1~5: conv → relu → conv → relu → pool 반복
        self.conv1_1 = Conv2dLayer(3, 64, 3, pad=1)    # RGB → 64장
        self.conv1_2 = Conv2dLayer(64, 64, 3, pad=1)
        self.conv2_1 = Conv2dLayer(64, 128, 3, pad=1)
        self.conv2_2 = Conv2dLayer(128, 128, 3, pad=1)
        self.conv3_1 = Conv2dLayer(128, 256, 3, pad=1)
        self.conv3_2 = Conv2dLayer(256, 256, 3, pad=1)
        self.conv3_3 = Conv2dLayer(256, 256, 3, pad=1)
        self.conv4_1 = Conv2dLayer(256, 512, 3, pad=1)
        self.conv4_2 = Conv2dLayer(512, 512, 3, pad=1)
        self.conv4_3 = Conv2dLayer(512, 512, 3, pad=1)
        self.conv5_1 = Conv2dLayer(512, 512, 3, pad=1)
        self.conv5_2 = Conv2dLayer(512, 512, 3, pad=1)
        self.conv5_3 = Conv2dLayer(512, 512, 3, pad=1)
        # FC: 분류기
        self.fc6 = Linear(512*7*7, 4096)
        self.fc7 = Linear(4096, 4096)
        self.fc8 = Linear(4096, 1000)     # 1000개 클래스

    def forward(self, x):
        # Block 1~5: conv → relu → pool 반복 (사이즈 절반씩 줄어듦)
        x = relu(self.conv1_1(x)); x = relu(self.conv1_2(x)); x = pooling(x, 2, 2)
        x = relu(self.conv2_1(x)); x = relu(self.conv2_2(x)); x = pooling(x, 2, 2)
        x = relu(self.conv3_1(x)); x = relu(self.conv3_2(x))
        x = relu(self.conv3_3(x)); x = pooling(x, 2, 2)
        x = relu(self.conv4_1(x)); x = relu(self.conv4_2(x))
        x = relu(self.conv4_3(x)); x = pooling(x, 2, 2)
        x = relu(self.conv5_1(x)); x = relu(self.conv5_2(x))
        x = relu(self.conv5_3(x)); x = pooling(x, 2, 2)
        # Flatten + FC + Dropout
        x = reshape(x, (x.shape[0], -1))
        x = dropout(relu(self.fc6(x)))
        x = dropout(relu(self.fc7(x)))
        return self.fc8(x)   # softmax는 손실함수 안에서


# %% [Block 8] 59 RNN 시계열 처리 — "순서를 기억하는 신경망"
class RNN(Layer):
    """핵심 공식: h_t = tanh(x_t · W_x + h_{t-1} · W_h + b)
    비유: 매일 일기를 쓰는 학생 — 오늘 경험 + 어제 기억 → 새 기억"""
    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        H = hidden_size
        I = in_size if in_size else H
        self.x2h = Linear(I, H)    # 입력 → 은닉 변환
        self.h2h = Linear(H, H)    # 이전 기억 → 현재 기억 변환
        self.h = None               # 기억 저장소

    def reset_state(self):
        """기억 초기화 (새 시퀀스 시작 시)"""
        self.h = None

    def __call__(self, x):
        if self.h is None:
            h_prev = np.zeros((x.shape[0], self.hidden_size))
        else:
            h_prev = self.h
        # 핵심: 새 기억 = tanh(현재 입력 변환 + 이전 기억 변환)
        h_new = tanh(self.x2h(x) + self.h2h(h_prev))
        self.h = h_new     # 다음 시각으로 전달
        return h_new


# %% [Block 9] 60 LSTM과 데이터 로더 — "장기 기억 노트 + 자동 급식"
class LSTM(Layer):
    """비유: 시험공부 노트 정리법
    - c (기억 셀) = 노트 전체 내용
    - f (망각 게이트) = 불필요한 내용 지우기
    - i (입력 게이트) = 새 중요 내용 추가하기
    - o (출력 게이트) = 지금 필요한 것만 꺼내 쓰기"""
    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        H = hidden_size
        I = in_size if in_size else H
        # 4개 게이트 각각의 입력/은닉 변환
        self.x2f = Linear(I, H);  self.h2f = Linear(H, H)  # 망각
        self.x2i = Linear(I, H);  self.h2i = Linear(H, H)  # 입력
        self.x2o = Linear(I, H);  self.h2o = Linear(H, H)  # 출력
        self.x2g = Linear(I, H);  self.h2g = Linear(H, H)  # 기억 후보
        self.H = H
        self.h = None   # 단기 기억
        self.c = None   # 장기 기억 (핵심!)

    def reset_state(self):
        self.h = None; self.c = None

    def __call__(self, x):
        H = self.H
        if self.h is None:
            h_prev = np.zeros((x.shape[0], H))
            c_prev = np.zeros((x.shape[0], H))
        else:
            h_prev, c_prev = self.h, self.c

        f = sigmoid(self.x2f(x) + self.h2f(h_prev))  # ① 얼마나 잊을까?
        i = sigmoid(self.x2i(x) + self.h2i(h_prev))  # ② 얼마나 기억할까?
        o = sigmoid(self.x2o(x) + self.h2o(h_prev))  # ③ 얼마나 내보낼까?
        g = tanh(self.x2g(x) + self.h2g(h_prev))     # 기억 후보

        c_new = f * c_prev + i * g   # 장기 기억 업데이트
        h_new = o * tanh(c_new)       # 단기 기억 = 장기 기억의 필터링 결과

        self.h, self.c = h_new, c_new
        return h_new


# %% [Block 10] 60 LSTM과 데이터 로더 — "장기 기억 노트 + 자동 급식"
class DataLoader:
    """비유: 뷔페의 자동 급식 — 한 번에 먹을 수 있는 양(batch)만큼 접시에 담아줌"""
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = self.data_size // batch_size
        self.reset()

    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(self.data_size)
        else:
            self.index = np.arange(self.data_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration
        i = self.iteration
        batch_idx = self.index[i*self.batch_size:(i+1)*self.batch_size]
        batch = [self.dataset[idx] for idx in batch_idx]
        x = np.array([b[0] for b in batch])
        t = np.array([b[1] for b in batch])
        self.iteration += 1
        return x, t

# 사용법: LSTM + DataLoader로 사인파 예측
lstm = LSTM(hidden_size=100, in_size=1)
fc = Linear(100, 1)
data_loader = DataLoader(train_data, batch_size=32)

for epoch in range(10):
    lstm.reset_state()
    for x_batch, t_batch in data_loader:
        h = lstm(Variable(x_batch))
        y_pred = fc(h)
        loss = mean_squared_error(y_pred, Variable(t_batch))
        # cleargrads → backward → update (항상 같은 3줄!)
