# -*- coding: utf-8 -*-
"""
03-09-03 그레이디언트 부스팅 — 잔차를 학습하는 릴레이

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-09장 - 부스팅 알고리즘/03-09-03 그레이디언트 부스팅 — 잔차를 학습하는 릴레이.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 2단계 — 강의 자료 7페이지 재현 — 트리 3그루 — 강의 자료 7페이지 재현 — 잔차를 이어 학습하는 트리 3그루
# ─────────────────────────────────────────────────────────────
# 강의 자료 7페이지 재현 — 잔차를 이어 학습하는 트리 3그루
# ─────────────────────────────────────────────────────────────
import numpy as np
from sklearn.tree import DecisionTreeRegressor
# ① 강의 자료와 같은 모양의 데이터 (2차 곡선 + 잡음)
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5          # x1: -0.5 ~ 0.5
y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)   # y = 3x² + 잡음
# ② h1 : 원래 정답 y 를 학습
h1 = DecisionTreeRegressor(max_depth=2, random_state=42).fit(X, y)
# ③ y - h1(x1)  = 첫 번째 잔차  (강의 자료 2행 왼쪽 그래프의 십자 표시)
y2 = y - h1.predict(X)
h2 = DecisionTreeRegressor(max_depth=2, random_state=42).fit(X, y2)
# ④ y - h1(x1) - h2(x1) = 두 번째 잔차 (3행 왼쪽 그래프)
y3 = y2 - h2.predict(X)
h3 = DecisionTreeRegressor(max_depth=2, random_state=42).fit(X, y3)
# ⑤ 각 단계에서 "남은 오차"가 얼마나 되는지 표로 확인
def sse(v):
    """SSE = Sum of Squared Errors, 오차 제곱합. 작을수록 잘 맞힌 것."""
    return float(np.sum(v ** 2))
print(' 단계 | 앙상블 식                    | 남은 오차(SSE) | 설명력')
print(' -----+------------------------------+----------------+--------')
tot = sse(y - y.mean())                      # 평균만 쓸 때의 총 변동
rows = [('0', '(평균으로만 예측)', y - y.mean()),
        ('1', 'h(x1)=h1(x1)', y2),
        ('2', 'h(x1)=h1(x1)+h2(x1)', y3),
        ('3', 'h(x1)=h1+h2+h3', y3 - h3.predict(X))]
for name, expr, resid in rows:
    r2 = 1 - sse(resid) / tot                # 결정계수 R² (1에 가까울수록 좋음)
    print(' %4s | %-28s | %14.4f | %6.4f' % (name, expr, sse(resid), r2))
# ⑥ 새 입력 x1 = 0.8 에 대한 예측 (강의 자료 오른쪽 열 빨간 선의 값)
X_new = np.array([[0.8]])
p1, p2, p3 = (float(h.predict(X_new)[0]) for h in (h1, h2, h3))
print()
print('x1 = 0.8 일 때')
print('  h1 = %+.4f' % p1)
print('  h2 = %+.4f   → h1+h2 = %.4f' % (p2, p1 + p2))
print('  h3 = %+.4f   → h1+h2+h3 = %.4f' % (p3, p1 + p2 + p3))


# %% [Block 2] 사이킷런과 완전히 같은가?
from sklearn.ensemble import GradientBoostingRegressor
# 강의 자료 7페이지의 코드 그대로
gbrt = GradientBoostingRegressor(max_depth=2,
                                 n_estimators=3,
                                 learning_rate=1.0,   # 각 트리를 100% 반영
                                 random_state=42)
gbrt.fit(X, y)
print('직접 만든 3그루 합 : %.10f' % (p1 + p2 + p3))
print('GradientBoosting   : %.10f' % gbrt.predict(X_new)[0])
print('차이               : %.2e' % abs((p1 + p2 + p3) - gbrt.predict(X_new)[0]))
print()
# staged_predict: 트리를 1개, 2개, 3개 썼을 때의 예측을 차례로 내놓습니다.
for i, pred in enumerate(gbrt.staged_predict(X_new), 1):
    print('  트리 %d개까지 사용한 예측 : %.4f' % (i, pred[0]))


# %% [Block 3] 직관 — 손실 함수를 바꾸면 '잔차'도 바뀐다 — 유도한 식이 진짜인지 컴퓨터로 검산합니다.
# ─────────────────────────────────────────────────────────────
# 유도한 식이 진짜인지 컴퓨터로 검산합니다.
# 방법: 수치 미분(아주 조금 움직였을 때 손실이 얼마나 변하는가)과 비교
# ─────────────────────────────────────────────────────────────
import numpy as np
def loss_sq(y, F):
    """제곱 오차 손실 L = 0.5*(y-F)^2"""
    return 0.5 * (y - F) ** 2
y_true = 3.0            # 실제 정답
F_pred = 2.2            # 현재 모델의 예측
eps = 1e-6              # 아주 작은 값 (수치 미분의 h)
# ① 수치 미분: (L(F+h) - L(F-h)) / (2h)  ← 중앙 차분, 오차가 가장 작은 방식
numeric = (loss_sq(y_true, F_pred + eps) - loss_sq(y_true, F_pred - eps)) / (2 * eps)
# ② 우리가 손으로 유도한 식: dL/dF = -(y - F)
analytic = -(y_true - F_pred)
print('수치 미분으로 구한 ∂L/∂F : %.8f' % numeric)
print('손으로 유도한  ∂L/∂F     : %.8f' % analytic)
print('두 값의 차이               : %.2e' % abs(numeric - analytic))
print()
print('그러므로 음의 기울기 −∂L/∂F = %.4f' % (-analytic))
print('잔차          y − F        = %.4f' % (y_true - F_pred))
print('→ 정확히 같습니다. 잔차는 곧 음의 기울기입니다.')
# ③ 로그 손실(분류)에서도 같은 일이 벌어지는지 확인
def sigmoid(z):
    """시그모이드: 실수를 0~1 확률로 바꿔 줍니다."""
    return 1 / (1 + np.exp(-z))
def loss_log(y, F):
    p = sigmoid(F)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))
y_c, F_c = 1.0, 0.4
num_c = (loss_log(y_c, F_c + eps) - loss_log(y_c, F_c - eps)) / (2 * eps)
print()
print('[분류·로그 손실]')
print('  수치 미분 −∂L/∂F : %.6f' % (-num_c))
print('  y − sigmoid(F)   : %.6f' % (y_c - sigmoid(F_c)))
print('  → 분류에서도 유사 잔차는 "정답 − 예측 확률" 입니다.')


# %% [Block 4] 초기값 \\(F_0\\)는 왜 평균인가? — 사이킷런의 GradientBoosting 이 정말 'y의 평균'에서 시작하는지 확인합니다.
# ─────────────────────────────────────────────────────────────
# 사이킷런의 GradientBoosting 이 정말 'y의 평균'에서 시작하는지 확인합니다.
# ─────────────────────────────────────────────────────────────
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)
g = GradientBoostingRegressor(n_estimators=1, max_depth=2,
                              learning_rate=0.0,     # 트리 기여를 0으로 → 초기값만 남음
                              random_state=42).fit(X, y)
print('y 의 평균          : %.6f' % y.mean())
print('학습률 0 일 때 예측 : %.6f' % g.predict([[0.3]])[0])
print('init_ 의 상수      : %.6f' % g.init_.constant_[0][0])
print()
print('→ 셋이 같습니다. 첫 걸음은 "아무 정보 없이 평균으로 찍기"에서 시작합니다.')
print()
# 수학적 근거: Σ(y_i - c)^2 을 c 로 미분해 0으로 두면 c = y의 평균
c_grid = np.linspace(y.mean() - 0.3, y.mean() + 0.3, 7)
print('  상수 c      | 손실 Σ(y−c)²')
for c in c_grid:
    print('  %10.4f | %12.4f' % (c, np.sum((y - c) ** 2)))
print('\n→ y.mean() = %.4f 에서 손실이 최소입니다.' % y.mean())


# %% [Block 5] 5단계 — 회귀 모델과 분류 모델 — 두 형제 — 회귀와 분류를 나란히 실행해 봅니다.
# ─────────────────────────────────────────────────────────────
# 회귀와 분류를 나란히 실행해 봅니다.
# ─────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
# ── ① 회귀: 당뇨병 진행도 예측 ───────────────────────────────
dr = load_diabetes()
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    dr.data, dr.target, test_size=0.2, random_state=156)
reg = GradientBoostingRegressor(random_state=0)   # 기본값: 트리 100개, lr=0.1, depth=3
reg.fit(Xr_tr, yr_tr)
print('[회귀] GradientBoostingRegressor')
print('  훈련 R² : %.4f' % r2_score(yr_tr, reg.predict(Xr_tr)))
print('  시험 R² : %.4f' % r2_score(yr_te, reg.predict(Xr_te)))
print('  시험 RMSE: %.2f' % np.sqrt(mean_squared_error(yr_te, reg.predict(Xr_te))))
print('  손실 함수 :', reg.loss)
# ── ② 분류: 유방암 양성/악성 ────────────────────────────────
dc = load_breast_cancer()
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    dc.data, dc.target, test_size=0.2, random_state=156)
clf = GradientBoostingClassifier(random_state=0)
clf.fit(Xc_tr, yc_tr)
print()
print('[분류] GradientBoostingClassifier')
print('  훈련 정확도 : %.4f' % accuracy_score(yc_tr, clf.predict(Xc_tr)))
print('  시험 정확도 : %.4f' % accuracy_score(yc_te, clf.predict(Xc_te)))
print('  손실 함수  :', clf.loss)
# ③ 분류기 내부는 확률이 아니라 '점수(로그 오즈)'를 더하고 있습니다.
raw = clf.decision_function(Xc_te[:3])       # 트리들의 합 = 로그 오즈
prob = clf.predict_proba(Xc_te[:3])[:, 1]    # 시그모이드를 통과한 확률
print()
print('  샘플 | 트리들의 합(로그오즈) | 시그모이드 통과 후 확률')
for i in range(3):
    print('  %4d | %21.4f | %22.4f' % (i, raw[i], prob[i]))
print('  → 1/(1+exp(-%.4f)) = %.4f  ✔' % (raw[0], 1 / (1 + np.exp(-raw[0]))))


# %% [Block 6] 실습 — [실습 1 정답 예시] 이상치가 하나 섞였을 때 두 손실 함수의 반응
# [실습 1 정답 예시] 이상치가 하나 섞였을 때 두 손실 함수의 반응
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
np.random.seed(0)
Xo = np.linspace(-1, 1, 60).reshape(-1, 1)      # reshape(-1,1): 1열짜리 2차원으로
yo = 2 * Xo[:, 0] + 0.1 * np.random.randn(60)
yo_bad = yo.copy()
yo_bad[30] = 25.0                                # ★ 이상치 하나 투입 (원래 값은 약 0)
print('  손실 함수        | 이상치 없음 예측 | 이상치 있음 예측 | 흔들린 폭')
print('  -----------------+------------------+------------------+----------')
for loss in ['squared_error', 'absolute_error', 'huber']:
    a = GradientBoostingRegressor(loss=loss, random_state=0).fit(Xo, yo)
    b = GradientBoostingRegressor(loss=loss, random_state=0).fit(Xo, yo_bad)
    pa = a.predict([[0.0]])[0]                   # x=0 에서의 예측(정답은 약 0)
    pb = b.predict([[0.0]])[0]
    print('  %-16s | %16.4f | %16.4f | %8.4f'
          % (loss, pa, pb, abs(pb - pa)))
print()
print('→ squared_error 가 이상치 쪽으로 가장 많이 끌려갑니다(오차를 제곱하니까).')
print('   absolute_error / huber 는 흔들림이 절반 수준으로 줄어듭니다.')
print('   실무에서 "이상한 값이 섞여 있다" 싶으면 손실 함수부터 바꿔 보세요.')


# %% [Block 7] 실습 — [실습 2 정답 예시] 트리 개수별 시험 RMSE — 최적 지점 찾기
# [실습 2 정답 예시] 트리 개수별 시험 RMSE — 최적 지점 찾기
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
d = load_diabetes()
Xt, Xv, yt, yv = train_test_split(d.data, d.target, test_size=0.3, random_state=0)
g = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                              max_depth=3, random_state=0).fit(Xt, yt)
# staged_predict: 트리 1개, 2개, ... 500개까지의 예측을 차례로 내놓는 제너레이터
errors = [mean_squared_error(yv, p) for p in g.staged_predict(Xv)]
best = int(np.argmin(errors)) + 1                # argmin: 가장 작은 값의 위치
print('  트리 수 | 시험 MSE')
for n in [1, 10, 30, best, 200, 500]:
    print('  %7d | %9.2f %s'
          % (n, errors[n - 1], '  ← 최적!' if n == best else ''))
print()
print('최적 트리 수 = %d (500개를 다 쓰면 오히려 나빠집니다)' % best)
print('500개일 때 대비 %.1f%% 개선' % ((errors[-1] - errors[best-1]) / errors[-1] * 100))
