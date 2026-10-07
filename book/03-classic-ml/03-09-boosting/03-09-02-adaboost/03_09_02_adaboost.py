# -*- coding: utf-8 -*-
"""
03-09-02 부스팅 알고리즘 AdaBoost — 적응형 부스팅

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-09장 - 부스팅 알고리즘/03-09-02 부스팅 알고리즘 AdaBoost — 적응형 부스팅.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 3단계 — 손으로 굴리는 3라운드 — 가중치는 어디로 움직이나 — AdaBoost를 라이브러리 없이 손으로 3라운드 굴려 봅니다.
# ─────────────────────────────────────────────────────────────

# AdaBoost를 라이브러리 없이 손으로 3라운드 굴려 봅니다.

# 데이터: x = 1~10, 정답 y 는 +1 / -1

# ─────────────────────────────────────────────────────────────

import numpy as np

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], dtype=float)

y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])

#   ↑ 이 정답은 일부러 '직선 하나로는 절대 못 나누게' 만들었습니다.

#     (+가 앞쪽과 뒤쪽에 흩어져 있어서 세로선 한 개로 못 가릅니다.)

w = np.ones(10) / 10          # ① 처음엔 모든 샘플에 똑같은 가중치 0.1

#   np.ones(10) → [1,1,...,1] 열 개, /10 → 합이 1이 되도록

def best_stump(X, y, w):

    """가중치 w 기준으로 '가장 덜 틀리는' 그루터기를 찾습니다.

    비유: 자를 하나 들고 1.5, 2.5, ... 자리마다 대 보면서
          제일 적게 틀리는 자리를 고르는 것과 같습니다.
    """

    best = None

    for th in np.arange(0.5, 10.5, 1.0):        # 임계값 후보: 0.5, 1.5, ..., 9.5

        for s in (1, -1):                       # 부호 방향 두 가지

            pred = np.where(X[:, 0] <= th, s, -s)

            #   np.where(조건, 참일 때 값, 거짓일 때 값)

            #   → th 이하면 s, 아니면 -s 로 예측

            err = w[pred != y].sum()            # 틀린 샘플의 가중치만 더함 = 가중 오차율

            #   pred != y → [False, True, ...] 불리언 배열

            #   w[불리언배열] → True 위치의 값만 골라 냄 (불리언 인덱싱)

            if best is None or err < best[0]:

                best = (err, th, s, pred)

    return best

alphas, stumps = [], []

for rnd in range(1, 4):                          # 3 라운드

    err, th, s, pred = best_stump(X, y, w)

    alpha = 0.5 * np.log((1 - err) / err)        # ② 발언권 계산 (유도한 그 식)

    wrong = np.where(pred != y)[0] + 1           # 틀린 샘플 번호(1부터 세기)

    print("─ 라운드 %d ─────────────────────────────" % rnd)

    print("  선택한 규칙 : x <= %.1f 이면 %+d, 아니면 %+d" % (th, s, -s))

    print("  가중 오차율 ε = %.4f" % err)

    print("  발언권    α = %.4f" % alpha)

    print("  틀린 샘플   : %s" % wrong)

    # ③ 가중치 갱신: 틀린 샘플은 e^(+α)배, 맞힌 샘플은 e^(-α)배

    w = w * np.exp(-alpha * y * pred)

    #   y*pred 는 맞히면 +1, 틀리면 -1 이므로

    #   맞힘 → exp(-α) (작아짐),  틀림 → exp(+α) (커짐)

    w = w / w.sum()                              # 합이 1이 되도록 정규화

    print("  갱신된 가중치: %s" % np.round(w, 4))

    print()

    alphas.append(alpha)

    stumps.append((th, s))


# %% [Block 2] 세 그루터기를 합치면? — 세 그루터기를 α 로 가중 합해 최종 예측을 만듭니다.
# 세 그루터기를 α 로 가중 합해 최종 예측을 만듭니다.
score = np.zeros(10)
for (th, s), a in zip(stumps, alphas):
    #   zip(A, B) → A와 B에서 하나씩 짝지어 꺼냅니다.
    score += a * np.where(X[:, 0] <= th, s, -s)   # 발언권을 곱해서 누적
final = np.where(score > 0, 1, -1)                # 점수 부호로 최종 판정
print("샘플 번호 :", list(range(1, 11)))
print("정답      :", y.tolist())            # tolist(): 넘파이 배열 → 파이썬 리스트
print("누적 점수 :", np.round(score, 3).tolist())
print("최종 예측 :", final.tolist())
print()
acc_each = []
for (th, s) in stumps:
    p = np.where(X[:, 0] <= th, s, -s)
    acc_each.append((p == y).mean())
print("그루터기 각각의 정확도 : %.1f%%, %.1f%%, %.1f%%"
      % tuple(a * 100 for a in acc_each))
print("세 개를 합친 정확도    : %.1f%%" % ((final == y).mean() * 100))
print()
print("→ 직선 한 개로는 절대 못 나누는 문제를,")
print("   직선 세 개의 '가중 투표'가 완전히 풀어냈습니다.")


# %% [Block 3] 4단계 — 사이킷런 AdaBoostClassifier 실습 — 사이킷런 AdaBoostClassifier 기본 사용법
# ─────────────────────────────────────────────────────────────
# 사이킷런 AdaBoostClassifier 기본 사용법
# (강의 자료 20페이지 코드와 같은 구조입니다)
# ─────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')     # 버전 안내 경고를 감춥니다(학습에 방해되므로)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=156)
# ① 모델 만들기 — 강의 자료와 같은 하이퍼파라미터
clf = AdaBoostClassifier(n_estimators=30,      # 약한 학습기 30개
                         random_state=10,      # 재현성 고정
                         learning_rate=0.1)    # 각 학습기의 발언권을 0.1배로 줄임
clf.fit(X_train, y_train)                      # ② 학습
pred = clf.predict(X_test)                     # ③ 예측
print('AdaBoost 정확도: {:.4f}'.format(accuracy_score(y_test, pred)))
#   '{:.4f}'.format(x) → x를 소수점 넷째 자리까지 문자열로 만듭니다.
# ④ 내부를 들여다봅니다 — 방금 배운 ε 과 α 가 그대로 들어 있습니다
print('\n학습된 약한 학습기 개수 :', len(clf.estimators_))
print('첫 번째 학습기의 종류    :', type(clf.estimators_[0]).__name__)
print('첫 번째 학습기의 깊이    :', clf.estimators_[0].get_depth())
print()
print('  m |  오차율 ε  |  발언권 α')
for m in range(5):
    print('%3d | %10.4f | %9.4f'
          % (m + 1, clf.estimator_errors_[m], clf.estimator_weights_[m]))
print('...')
print('α의 총합 : %.4f' % clf.estimator_weights_.sum())


# %% [Block 4] 5단계 — 학습률과 트리 수의 시소 관계 — 학습률과 학습기 개수를 바꿔 가며 시험 정확도를 비교합니다.
# ─────────────────────────────────────────────────────────────
# 학습률과 학습기 개수를 바꿔 가며 시험 정확도를 비교합니다.
# ─────────────────────────────────────────────────────────────
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
print('  n_est | lr    | 훈련 정확도 | 시험 정확도')
print('  ------+-------+-------------+------------')
for n, lr in [(30, 0.1), (30, 0.5), (30, 1.0),
              (100, 0.1), (300, 0.1), (400, 0.1), (400, 1.0)]:
    m = AdaBoostClassifier(n_estimators=n, learning_rate=lr, random_state=10)
    m.fit(X_train, y_train)
    tr = accuracy_score(y_train, m.predict(X_train))
    te = accuracy_score(y_test, m.predict(X_test))
    print('  %5d | %5.2f | %11.4f | %10.4f' % (n, lr, tr, te))
print()
print('→ lr=0.1 인데 n_estimators 가 30밖에 안 되면 보폭도 작고 걸음도 적어 덜 학습됩니다.')
print('   같은 lr=0.1 에서 학습기를 400개로 늘리면 성능이 올라갑니다.')


# %% [Block 5] 실습 — [실습 3 정답 예시] 라운드를 늘리며 훈련 정확도를 추적합니다.
# [실습 3 정답 예시] 라운드를 늘리며 훈련 정확도를 추적합니다.
import numpy as np
X = np.array([[i] for i in range(1, 11)], dtype=float)
y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
w = np.ones(10) / 10
alphas, stumps = [], []
print('  라운드 |   ε    |   α    | 누적 훈련 정확도')
for rnd in range(1, 11):
    best = None
    for th in np.arange(0.5, 10.5, 1.0):
        for s in (1, -1):
            pred = np.where(X[:, 0] <= th, s, -s)
            err = w[pred != y].sum()
            if best is None or err < best[0]:
                best = (err, th, s, pred)
    err, th, s, pred = best
    err = max(err, 1e-10)                 # ε=0 이면 log 가 무한대가 되므로 방어
    alpha = 0.5 * np.log((1 - err) / err)
    alphas.append(alpha); stumps.append((th, s))
    score = np.zeros(10)
    for (t2, s2), a2 in zip(stumps, alphas):
        score += a2 * np.where(X[:, 0] <= t2, s2, -s2)
    acc = (np.where(score > 0, 1, -1) == y).mean()
    print('  %6d | %.4f | %6.4f | %14.1f%%' % (rnd, err, alpha, acc * 100))
    w = w * np.exp(-alpha * y * pred)
    w = w / w.sum()
