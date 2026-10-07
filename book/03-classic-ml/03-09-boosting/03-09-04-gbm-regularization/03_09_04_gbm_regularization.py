# -*- coding: utf-8 -*-
"""
03-09-04 GBM의 규제 — 학습률·조기 종료·확률적·히스토그램 부스팅

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-09장 - 부스팅 알고리즘/03-09-04 GBM의 규제 — 학습률·조기 종료·확률적·히스토그램 부스팅.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 1단계 — 학습률과 수축(shrinkage) 규제 — 학습률이 예측에 어떻게 곱해지는지 눈으로 확인합니다.
# ─────────────────────────────────────────────────────────────
# 학습률이 예측에 어떻게 곱해지는지 눈으로 확인합니다.
# ─────────────────────────────────────────────────────────────
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)
X_new = np.array([[0.8]])
# ① 학습률을 직접 반영한 손 구현
eta = 0.3
F = np.full(len(y), y.mean())       # 초기값 F0 = y의 평균
                                    # np.full(길이, 값) → 그 값으로 채운 배열
preds_new = y.mean()                # 새 입력에 대한 누적 예측도 같이 추적
print('  단계 | 이번 트리 예측 | ×η 후 더한 값 | 누적 예측 | 훈련 SSE')
for m in range(1, 6):
    r = y - F                                    # 잔차 = 음의 기울기
    h = DecisionTreeRegressor(max_depth=2, random_state=42).fit(X, r)
    step = h.predict(X_new)[0]                   # 새 입력에서 이번 트리가 내놓은 값
    F = F + eta * h.predict(X)                   # ★ 학습률을 곱해서 더한다
    preds_new = preds_new + eta * step
    print('  %4d | %14.4f | %13.4f | %9.4f | %8.4f'
          % (m, step, eta * step, preds_new, np.sum((y - F) ** 2)))
# ② 사이킷런과 대조
g = GradientBoostingRegressor(max_depth=2, n_estimators=5,
                              learning_rate=eta, random_state=42).fit(X, y)
print()
print('손 구현 누적 예측 : %.8f' % preds_new)
print('사이킷런 GBRT     : %.8f' % g.predict(X_new)[0])
print('차이              : %.2e' % abs(preds_new - g.predict(X_new)[0]))


# %% [Block 2] 2단계 — 강의 자료 9페이지 재현 — '92'는 어디서 나온 숫자인가 — 강의 자료 9페이지의 두 설정을 실제로 비교합니다.
# ─────────────────────────────────────────────────────────────
# 강의 자료 9페이지의 두 설정을 실제로 비교합니다.
# ─────────────────────────────────────────────────────────────
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)
print('  설정                              | 훈련 MSE | 평가')
print('  ----------------------------------+----------+------------------')
for lr, n, note in [(1.0, 3,  '강의 왼쪽  (과소적합)'),
                    (0.05, 3, '보폭도 작고 걸음도 적음'),
                    (0.05, 92, '강의 오른쪽 (적절)'),
                    (1.0, 92, '보폭이 커서 잡음까지 학습')]:
    m = GradientBoostingRegressor(max_depth=2, n_estimators=n,
                                  learning_rate=lr, random_state=42).fit(X, y)
    print('  lr=%-5.2f n_estimators=%-4d       | %8.5f | %s'
          % (lr, n, mean_squared_error(y, m.predict(X)), note))
print()
print('※ 잡음의 표준편차가 0.05 이므로, "어쩔 수 없는 오차"의 MSE는 약 %.5f 입니다.'
      % 0.05 ** 2)
print('   훈련 MSE가 이보다 훨씬 작아졌다면 잡음까지 외운 것(과대적합)입니다.')


# %% [Block 3] 이제 '92'의 정체를 밝힙니다 — 조기 종료를 켜고 학습시키면 몇 그루에서 멈출까요?
# ─────────────────────────────────────────────────────────────
# 조기 종료를 켜고 학습시키면 몇 그루에서 멈출까요?
# ─────────────────────────────────────────────────────────────
from sklearn.ensemble import GradientBoostingRegressor
# 강의 자료 10페이지의 설정 그대로
gbrt = GradientBoostingRegressor(max_depth=2,
                                 learning_rate=0.05,
                                 n_estimators=500,       # 최대 500그루까지 허용
                                 n_iter_no_change=10,    # 10번 연속 안 좋아지면 중단
                                 random_state=42)
gbrt.fit(X, y)
print('n_estimators 로 허용한 최대 트리 수 :', 500)
print('실제로 학습된 트리 수 (n_estimators_) :', gbrt.n_estimators_)
print()
print('★ 강의 자료 9페이지의 "n_estimators=92" 는')
print('   사람이 고른 숫자가 아니라 조기 종료가 스스로 찾아낸 지점이었습니다.')


# %% [Block 4] 3단계 — 조기 종료 — 알아서 멈추는 모델 — 강의 자료 10페이지 코드 그대로 + 내부 동작 관찰
# ─────────────────────────────────────────────────────────────
# 강의 자료 10페이지 코드 그대로 + 내부 동작 관찰
# ─────────────────────────────────────────────────────────────
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)
# 강의 자료 원문 코드
gbrt = GradientBoostingRegressor(max_depth=2,
                                 learning_rate=0.05,
                                 n_estimators=500,
                                 n_iter_no_change=10, random_state=42)
gbrt.fit(X, y)
print('조기 종료 ON  → 실제 트리 수 : %3d' % gbrt.n_estimators_)
# 비교: 조기 종료를 끄면?
gbrt_off = GradientBoostingRegressor(max_depth=2, learning_rate=0.05,
                                     n_estimators=500, random_state=42).fit(X, y)
print('조기 종료 OFF → 실제 트리 수 : %3d' % gbrt_off.n_estimators_)
print()
# tol 을 바꾸면 멈추는 시점이 어떻게 달라지나?
print('  n_iter_no_change |    tol   | 멈춘 트리 수')
print('  -----------------+----------+-------------')
for niter in [3, 10, 30]:
    for tol in [1e-2, 1e-4, 1e-6]:
        m = GradientBoostingRegressor(max_depth=2, learning_rate=0.05,
                                      n_estimators=500, n_iter_no_change=niter,
                                      tol=tol, random_state=42).fit(X, y)
        print('  %16d | %8.0e | %11d' % (niter, tol, m.n_estimators_))
print()
print('→ 인내심(n_iter_no_change)이 길수록, 기준(tol)이 깐깐할수록 더 오래 학습합니다.')


# %% [Block 5] 직접 조기 종료 구현하기 — 무슨 일이 일어나는지 보기 — staged_predict 로 조기 종료를 손으로 구현해 봅니다.
# ─────────────────────────────────────────────────────────────
# staged_predict 로 조기 종료를 손으로 구현해 봅니다.
# (사이킷런 내부가 하는 일을 눈에 보이게 만드는 코드입니다)
# ─────────────────────────────────────────────────────────────
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
np.random.seed(42)
X = np.random.rand(300, 1) - 0.5
y = 3 * X[:, 0] ** 2 + 0.08 * np.random.randn(300)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
g = GradientBoostingRegressor(max_depth=3, n_estimators=400,
                              learning_rate=0.1, random_state=42).fit(X_tr, y_tr)
# staged_predict: 트리를 1개, 2개, ... 400개 썼을 때의 예측을 차례로 돌려줍니다.
val_errors = [mean_squared_error(y_val, p) for p in g.staged_predict(X_val)]
tr_errors = [mean_squared_error(y_tr, p) for p in g.staged_predict(X_tr)]
best_n = int(np.argmin(val_errors)) + 1     # argmin: 최솟값의 위치(0부터)
print('  트리 수 | 훈련 MSE  | 검증 MSE  |')
for n in sorted({1, 5, 20, 50, best_n, 200, 400}):   # sorted(): 오름차순 정렬
    mark = '  ← 검증 최소!' if n == best_n else ''
    print('  %7d | %9.5f | %9.5f |%s' % (n, tr_errors[n-1], val_errors[n-1], mark))
print()
print('최적 트리 수 : %d' % best_n)
print('훈련 오차는 400그루까지 계속 줄지만(%.5f), 검증 오차는 %d그루에서 최소(%.5f)이고'
      % (tr_errors[-1], best_n, val_errors[best_n-1]))
print('그 뒤로는 다시 올라갑니다(400그루: %.5f). 이것이 과대적합의 정확한 모습입니다.'
      % val_errors[-1])


# %% [Block 6] 4단계 — 확률적 그레이디언트 부스팅 — subsample — subsample 을 바꿔 가며 성능·시간·트리 간 유사도를 관찰합니다.
# ─────────────────────────────────────────────────────────────
# subsample 을 바꿔 가며 성능·시간·트리 간 유사도를 관찰합니다.
# ─────────────────────────────────────────────────────────────
import time
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
np.random.seed(0)
X = np.random.rand(2000, 8)                       # 샘플 2000개, 특징 8개
y = (3 * X[:, 0] ** 2 + 2 * X[:, 1] - X[:, 2]
     + 0.3 * np.random.randn(2000))
y[:20] = y[:20] + 6                               # ★ 이상치 20개를 일부러 투입
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
print('  subsample | 훈련 MSE | 시험 MSE | 학습 시간')
print('  ----------+----------+----------+-----------')
for ss in [1.0, 0.8, 0.5, 0.25]:
    t0 = time.time()                              # time.time(): 현재 시각(초)
    m = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                  max_depth=3, subsample=ss,
                                  random_state=0).fit(X_tr, y_tr)
    dt = time.time() - t0
    print('  %9.2f | %8.4f | %8.4f | %8.2f초'
          % (ss, mean_squared_error(y_tr, m.predict(X_tr)),
             mean_squared_error(y_te, m.predict(X_te)), dt))
print()
print('→ subsample 을 줄이면 훈련 MSE 는 올라가고(편향 ↑),')
print('   시험 MSE 는 오히려 내려갈 수 있습니다(분산 ↓). 학습 시간도 줄어듭니다.')
print('   이상치가 섞여 있을 때 이 효과가 특히 뚜렷합니다.')


# %% [Block 7] subsample vs 배깅의 부트스트랩 — 헷갈리지 마세요 — subsample < 1 일 때만 쓸 수 있는 보너스: OOB(Out-Of-Bag) 개선량
# subsample < 1 일 때만 쓸 수 있는 보너스: OOB(Out-Of-Bag) 개선량
# 각 트리가 "자기가 보지 않은 데이터"에서 얼마나 손실을 줄였는지 기록해 줍니다.
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
m = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                              max_depth=3, subsample=0.5,
                              random_state=0).fit(X_tr, y_tr)
oob = m.oob_improvement_          # 트리마다의 OOB 손실 개선량(양수면 도움이 됐다는 뜻)
cum = np.cumsum(oob)              # cumsum: 누적 합
best = int(np.argmax(cum)) + 1    # 누적 개선이 가장 큰 지점 = 사실상 최적 트리 수
print('OOB 개선량 배열 길이 :', len(oob))
print('처음 5개 트리의 개선량:', np.round(oob[:5], 5))
print('마지막 5개 트리       :', np.round(oob[-5:], 5))
print()
print('누적 개선이 최대가 되는 트리 수 : %d' % best)
print('→ 검증셋을 따로 떼지 않고도 "몇 그루가 적당한지" 추정할 수 있습니다.')
print('   (음수가 섞이기 시작하면 그 트리들은 오히려 해가 되고 있다는 신호입니다.)')


# %% [Block 8] 왜 빨라지는가 — 분할 탐색 비용의 정체 — 대용량 데이터에서 두 방식의 학습 시간을 실제로 재 봅니다.
# ─────────────────────────────────────────────────────────────
# 대용량 데이터에서 두 방식의 학습 시간을 실제로 재 봅니다.
# ─────────────────────────────────────────────────────────────
import time
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# 샘플 5만 개, 특징 20개짜리 데이터
Xb, yb = make_regression(n_samples=50000, n_features=20, noise=1.0, random_state=42)
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.2, random_state=42)
t0 = time.time()
gb = GradientBoostingRegressor(max_depth=3, n_estimators=100,
                               random_state=42).fit(Xb_tr, yb_tr)
t_gb = time.time() - t0
t0 = time.time()
hgb = HistGradientBoostingRegressor(max_depth=3, max_iter=100,
                                    random_state=42).fit(Xb_tr, yb_tr)
t_hgb = time.time() - t0
print('  모델                          | 학습 시간 | 시험 MSE')
print('  ------------------------------+-----------+-----------')
print('  GradientBoostingRegressor     | %7.1f초 | %9.1f'
      % (t_gb, mean_squared_error(yb_te, gb.predict(Xb_te))))
print('  HistGradientBoostingRegressor | %7.1f초 | %9.1f'
      % (t_hgb, mean_squared_error(yb_te, hgb.predict(Xb_te))))
print()
print('속도 차이: 약 %.0f배' % (t_gb / t_hgb))
print('→ 샘플 5만 개에서 이미 이 정도입니다. 100만 개면 격차가 더 벌어집니다.')


# %% [Block 9] max_bins의 효과 — 눈금을 성기게 하면? — max_bins 를 줄이면 어떻게 될까요? (규제 효과 vs 과소적합)
# max_bins 를 줄이면 어떻게 될까요? (규제 효과 vs 과소적합)
import numpy as np, time
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
print('  max_bins | 학습 시간 | 훈련 MSE | 시험 MSE | 해석')
print('  ---------+-----------+----------+----------+---------------------')
for mb, note in [(255, '기본값(최대)'), (64, '눈금 성김'),
                 (16, '많이 성김'), (4, '너무 성김 → 과소적합')]:
    t0 = time.time()
    m = HistGradientBoostingRegressor(max_bins=mb, max_iter=100,
                                      max_depth=3, random_state=42).fit(Xb_tr, yb_tr)
    dt = time.time() - t0
    print('  %8d | %7.2f초 | %8.1f | %8.1f | %s'
          % (mb, dt, mean_squared_error(yb_tr, m.predict(Xb_tr)),
             mean_squared_error(yb_te, m.predict(Xb_te)), note))
print()
print('→ max_bins 는 256보다 큰 값을 줄 수 없습니다(내부적으로 8비트 정수를 쓰기 때문).')
print('   너무 줄이면 값의 구분이 뭉개져 과소적합이 발생합니다.')


# %% [Block 10] 실무 튜닝 순서 — 무엇부터 만질 것인가 — 위 순서대로 실제로 튜닝해 봅니다 (GridSearchCV 대신 단계별 탐색)
# ─────────────────────────────────────────────────────────────

# 위 순서대로 실제로 튜닝해 봅니다 (GridSearchCV 대신 단계별 탐색)

# ─────────────────────────────────────────────────────────────

import warnings

warnings.filterwarnings('ignore')

import numpy as np

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.ensemble import GradientBoostingClassifier

d = load_breast_cancer()

Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(

    d.data, d.target, test_size=0.2, random_state=156)

def cv(**kw):

    """교차검증 정확도의 평균을 돌려주는 도우미 함수.

    **kw 는 '키워드 인자를 딕셔너리로 받는다'는 뜻입니다.
    cv(max_depth=3) 처럼 부르면 kw = {'max_depth': 3} 이 됩니다.
    """

    base = dict(n_estimators=200, learning_rate=0.05, random_state=0)

    base.update(kw)                              # 기본 설정 위에 kw 를 덮어씀

    m = GradientBoostingClassifier(**base)       # ** 는 딕셔너리를 인자로 펼치기

    return cross_val_score(m, Xc_tr, yc_tr, cv=5, scoring='accuracy').mean()

print('[2단계] max_depth 탐색')

for depth in [2, 3, 4, 5]:

    print('   max_depth=%d → 교차검증 정확도 %.4f' % (depth, cv(max_depth=depth)))

print('\n[3단계] subsample 탐색 (max_depth=3 고정)')

for ss in [1.0, 0.8, 0.5]:

    print('   subsample=%.1f → 교차검증 정확도 %.4f'

          % (ss, cv(max_depth=3, subsample=ss)))

print('\n[4단계] min_samples_leaf 탐색')

for leaf in [1, 5, 20]:

    print('   min_samples_leaf=%2d → 교차검증 정확도 %.4f'

          % (leaf, cv(max_depth=3, subsample=0.8, min_samples_leaf=leaf)))


# %% [Block 11] 실습 — [실습 2 정답 예시] η × M 이 정말 대략 일정한가?
# [실습 2 정답 예시] η × M 이 정말 대략 일정한가?
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)
print('  learning_rate | 멈춘 트리 수 M | η × M')
print('  --------------+----------------+-------')
for lr in [0.2, 0.1, 0.05, 0.02]:
    m = GradientBoostingRegressor(max_depth=2, learning_rate=lr,
                                  n_estimators=2000, n_iter_no_change=10,
                                  random_state=42).fit(X, y)
    print('  %13.2f | %14d | %6.2f' % (lr, m.n_estimators_, lr * m.n_estimators_))
print()
print('→ η×M 이 완전히 일정하지는 않지만 같은 자릿수 안에 머무릅니다.')
print('   "보폭을 절반으로 줄이면 걸음은 대략 두 배" 라는 감각을 잡아 두세요.')
