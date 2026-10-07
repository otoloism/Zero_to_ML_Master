# -*- coding: utf-8 -*-
"""
04-01-02 신경망 —활성화 함수와 순전파

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-01장 딥러닝 이론과 구현/04-01-02 신경망 —활성화 함수와 순전파.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-01-02-B AND, NAND, OR 게이트 — "퍼셉트론의 첫 번째 임무"
import numpy as np  # NumPy: 배열/행렬 연산 핵심 라이브러리

def AND(x1, x2):
    """AND 게이트: 둘 다 1이어야 1.
    비유: '두 사람이 동시에 OK해야 통과하는 문'"""
    x = np.array([x1, x2])            # 입력을 배열로
    w = np.array([0.5, 0.5])           # 가중치
    b = -0.7                            # 편향 (높을수록 발화 어려움)
    tmp = np.sum(w * x) + b            # 00-01 내적: w·x + b
    return 1 if tmp > 0 else 0          # 계단함수: 양수→1, 그외→0

def NAND(x1, x2):
    """NAND 게이트: AND의 정반대 (가중치 부호 뒤집기).
    비유: '둘 다 OK할 때만 막는 문'"""
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])        # AND와 부호 반대!
    b = 0.7
    tmp = np.sum(w * x) + b
    return 1 if tmp > 0 else 0

def OR(x1, x2):
    """OR 게이트: 둘 중 하나만 1이면 1.
    비유: '한 사람만 OK해도 통과하는 문'"""
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2                            # AND보다 편향이 작아 → 쉽게 발화
    tmp = np.sum(w * x) + b
    return 1 if tmp > 0 else 0

# ── 테스트: 모든 입력 조합 ──
for x1, x2 in [(0,0), (0,1), (1,0), (1,1)]:
    print(f"AND({x1},{x2})={AND(x1,x2)}"
          f"  NAND({x1},{x2})={NAND(x1,x2)}"
          f"  OR({x1},{x2})={OR(x1,x2)}")


# %% [Block 2] 04-01-02-C XOR — "직선 하나로는 풀 수 없는 문제"
def XOR(x1, x2):
    """XOR = NAND + OR → AND (2층 조합).
    비유: '막대기 두 개로 영역을 나누는 것'"""
    s1 = NAND(x1, x2)                  # 1층: NAND
    s2 = OR(x1, x2)                    # 1층: OR
    y = AND(s1, s2)                    # 2층: AND → 최종 출력
    return y

for x1, x2 in [(0,0), (0,1), (1,0), (1,1)]:
    print(f"XOR({x1},{x2}) = {XOR(x1,x2)}")


# %% [Block 3] 04-01-02-D 활성화 함수 ① 시그모이드 — "계단을 S자 곡선으로 교체"
import numpy as np

def sigmoid(x):
    """시그모이드: 어떤 숫자든 0~1 사이로 압축.
    비유: 온도계 — 아무리 뜨거워도 100도(1)를 넘지 않고,
    아무리 추워도 0도(0) 아래로 내려가지 않는 특수 온도계."""
    return 1 / (1 + np.exp(-x))        # e^(-x) 계산

def step_function(x):
    """계단함수: 0보다 크면 1, 아니면 0.
    비유: 스위치 — ON 아니면 OFF, 중간은 없음."""
    return np.array(x > 0, dtype=np.int32)

# ── 비교 출력 ──
x_values = [-5, -2, -1, 0, 1, 2, 5]
print("  x  | 계단함수 | 시그모이드")
print("─────┼─────────┼──────────")
for x in x_values:
    step = 1 if x > 0 else 0
    sig = sigmoid(x)
    print(f" {x:3d} |    {step}    | {sig:.4f}")


# %% [Block 4] 04-01-02-E 활성화 함수 ② ReLU — "음수는 차단, 양수는 그대로"
def relu(x):
    """ReLU: 음수는 0으로 차단, 양수는 그대로 통과.
    비유: '수도꼭지' — 물(양수)은 흐르지만, 역류(음수)는 막힘."""
    return np.maximum(0, x)               # 0과 x 중 큰 값

# ── 비교 출력 ──
test_x = np.array([-3, -1, 0, 1, 3, 5])
print("입력 x :", test_x)
print("ReLU(x):", relu(test_x))
print("σ(x)   :", np.round(sigmoid(test_x), 3))


# %% [Block 5] 04-01-02-F 3층 신경망 순전파 — "공장 조립 라인 가동!"
import numpy as np

def init_network():
    """가중치와 편향을 초기화합니다.
    비유: '공장 조립 라인의 설정값을 정하는 것'"""
    network = {}
    # 1층: 입력 2개 → 은닉 3개
    network['W1'] = np.array([[0.1, 0.3, 0.5],
                             [0.2, 0.4, 0.6]])    # shape: (2, 3)
    network['b1'] = np.array([0.1, 0.2, 0.3])     # shape: (3,)

    # 2층: 은닉 3개 → 은닉 2개
    network['W2'] = np.array([[0.1, 0.4],
                             [0.2, 0.5],
                             [0.3, 0.6]])        # shape: (3, 2)
    network['b2'] = np.array([0.1, 0.2])          # shape: (2,)

    # 3층: 은닉 2개 → 출력 2개
    network['W3'] = np.array([[0.1, 0.3],
                             [0.2, 0.4]])        # shape: (2, 2)
    network['b3'] = np.array([0.1, 0.2])          # shape: (2,)
    return network

def forward(network, x):
    """순전파: 입력 x를 받아 출력 y를 계산합니다.
    비유: '원재료가 조립 라인을 통과하며 완성품이 되는 과정'"""
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 1층: 입력 → 은닉1 (행렬곱 + 편향 + 시그모이드)
    a1 = np.dot(x, W1) + b1             # 00-02 행렬곱: (1,2)@(2,3)=(1,3)
    z1 = sigmoid(a1)                    # 활성화 함수 통과

    # 2층: 은닉1 → 은닉2
    a2 = np.dot(z1, W2) + b2            # (1,3)@(3,2)=(1,2)
    z2 = sigmoid(a2)

    # 3층: 은닉2 → 출력 (항등함수: 그대로 출력)
    a3 = np.dot(z2, W3) + b3            # (1,2)@(2,2)=(1,2)
    y = a3                               # 출력층: 활성화 없이 그대로

    return y

# ── 실행 ──
network = init_network()
x = np.array([1.0, 0.5])                # 입력
y = forward(network, x)                # 순전파!
print("출력 y:", y)


# %% [Block 6] 04-01-02-G 소프트맥스 — "점수를 확률로 바꾸기"
def softmax(a):
    """소프트맥스: 점수(logit)를 확률 분포로 변환.
    비유: '투표 결과를 득표율(%)로 변환하기'
    — 5표, 3표, 2표 → 50%, 30%, 20%처럼!"""
    c = np.max(a)                       # 오버플로 방지: 최댓값 빼기
    exp_a = np.exp(a - c)               # e^(a-c) → 큰 수 방지
    sum_exp_a = np.sum(exp_a)            # 전체 합
    return exp_a / sum_exp_a              # 각각 / 전체합 = 확률!

# ── 테스트 ──
scores = np.array([2.0, 1.0, 0.1])       # 신경망의 출력값 (점수)
probs = softmax(scores)
print("점수(logit):", scores)
print("확률(softmax):", np.round(probs, 3))
print("합계:", np.round(np.sum(probs), 3))  # 반드시 1.0!
print("가장 높은 클래스:", np.argmax(probs))


# %% [Block 7] 04-01-02-H MNIST 손글씨 인식 — "진짜 이미지를 분류하는 신경망"
import numpy as np
import pickle  # 저장된 가중치 파일을 불러오는 도구

# ── 가중치가 이미 학습된 신경망 불러오기 ──
# (실제로는 sample_weight.pkl 파일 필요 — 책 예제 코드 참조)

def predict(network, x):
    """MNIST 추론: 이미지(784차원) → 숫자(0~9).
    비유: '사진을 보고 숫자를 맞추는 AI 선생님'"""
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1             # 1층 순전파
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2            # 2층 순전파
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3            # 3층 순전파
    y = softmax(a3)                     # 출력을 확률로 변환

    return y

# ── 정확도 측정 (의사 코드) ──
# 실제 실행시에는 MNIST 데이터와 가중치 파일이 필요합니다.
# accuracy_cnt = 0
# for i in range(len(x_test)):
#     y = predict(network, x_test[i])
#     predicted = np.argmax(y)       # 확률 최대인 클래스
#     if predicted == t_test[i]:     # 정답과 비교
#         accuracy_cnt += 1
# print(f"정확도: {accuracy_cnt / len(x_test):.4f}")
# → 정확도: 0.9352 (약 93.5%)

print("MNIST 3층 신경망 구조:")
print("  입력: 784 (28×28 픽셀)")
print("  은닉1: 50 뉴런 (sigmoid)")
print("  은닉2: 100 뉴런 (sigmoid)")
print("  출력: 10 클래스 (softmax → 0~9)")
print("  학습된 가중치 사용 → 정확도: ~93.5%")
