# -*- coding: utf-8 -*-
"""
03-08-06 Ridge · Lasso · ElasticNet 실습 (Regularization Practice)

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-08장 - 정규화 심화 및 대규모 학습/03-08-06 Ridge · Lasso · ElasticNet 실습.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 데이터셋에 관한 안내
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

SEED = 42          # 난수 씨앗을 고정해야 매번 같은 결과가 나온다 (재현성)

# as_frame=True : 결과를 pandas DataFrame으로 받는다 (컬럼 이름이 살아 있다)
data = load_diabetes(as_frame=True)
X = data.data      # 특성 10개
y = data.target    # 1년 뒤 당뇨병 진행도 (숫자 → 회귀 문제)

# train_test_split : 데이터를 훈련용/시험용으로 나눈다
#   test_size=0.25 → 25%를 시험용으로 떼어둔다
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED)

print("전체 데이터 :", X.shape)          # (샘플 수, 특성 수)
print("훈련 데이터 :", x_train.shape)
print("시험 데이터 :", x_test.shape)
print()
print("특성 이름 :", list(x_train.columns))
print()
print("각 특성의 대략적 뜻")
print("  age : 나이 / sex : 성별 / bmi : 체질량지수 / bp : 평균 혈압")
print("  s1~s6 : 여섯 가지 혈청(피검사) 수치")
print()
print("주의: 이 데이터는 이미 표준화되어 있다. 그래서 별도 스케일링이 필요 없다.")
print("      (직접 만든 데이터라면 반드시 StandardScaler를 먼저 거쳐야 한다!)")


# %% [Block 2] 2단원 — 여러 모델을 한 표에 모으는 도구 만들기 — 모델들의 성적을 모아두는 상자
# 모델들의 성적을 모아두는 상자
model_scores = {}          # {} 는 딕셔너리. '이름 → 값' 형태로 저장한다

def add_model(name, pred, y_true):
    """모델 하나의 예측 결과를 채점해서 상자에 넣는다."""
    mse = mean_squared_error(y_true, pred)
    model_scores[name] = mse
    return mse

def plot_all(top=None):
    """상자에 모인 모델들을 MSE가 낮은 순으로 줄 세워 막대로 그린다."""
    # sorted(..., key=lambda kv: kv[1]) : 값(MSE)을 기준으로 정렬
    #   lambda 는 이름 없는 짧은 함수를 만드는 문법이다.
    #   kv 는 ('모델이름', mse) 짝이고, kv[1] 이 mse 다.
    ranked = sorted(model_scores.items(), key=lambda kv: kv[1])
    if top:
        ranked = ranked[:top]

    worst = max(model_scores.values())        # 가장 나쁜 값 (막대 길이 기준)

    print("%-26s %12s" % ("model", "mse"))
    print("-" * 62)
    for name, mse in ranked:
        bar_len = int(30 * mse / worst)       # 막대 길이를 비율로 계산
        print("%-26s %12.4f  %s" % (name, mse, "█" * bar_len))

# 기준선이 되는 '규제 없는' 선형회귀부터 넣어둔다
linear = LinearRegression()
linear.fit(x_train, y_train)                  # fit : 데이터로 학습시킨다
pred = linear.predict(x_test)                 # predict : 학습된 모델로 예측한다
add_model('LinearRegression', pred, y_test)

plot_all()


# %% [Block 3] 3단원 — Ridge (L2 Regularization) — 값이 커질 수록 큰 규제입니다.
# 값이 커질 수록 큰 규제입니다.
alphas = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]

for alpha in alphas:
    ridge = Ridge(alpha=alpha, random_state=SEED)
    ridge.fit(x_train, y_train)
    pred = ridge.predict(x_test)
    # 'Ridge(alpha={})'.format(alpha) : {} 자리에 alpha 값을 끼워 넣어 이름을 만든다
    add_model('Ridge(alpha={})'.format(alpha), pred, y_test)

plot_all()


# %% [Block 4] 4단원 — coef_ 읽기 : 어떤 특성이 중요한가
def plot_coef(columns, coef):
    """특성 이름과 계수를 짝지어 크기순으로 정렬해 보여준다."""
    # zip(a, b) : 두 목록을 짝지어 (a1,b1), (a2,b2), ... 로 묶는다
    # list(...)  : 짝지어진 결과를 목록으로 만든다
    coef_df = pd.DataFrame(list(zip(columns, coef)))
    coef_df.columns = ['feature', 'coef']

    # sort_values('coef', ascending=False) : coef 열을 기준으로 큰 것부터 정렬
    # reset_index(drop=True) : 정렬 후 번호를 0부터 다시 매긴다 (옛 번호는 버림)
    coef_df = coef_df.sort_values('coef', ascending=False).reset_index(drop=True)

    scale = max(abs(coef_df['coef'])) or 1        # 막대 길이 기준 (0으로 나누기 방지)
    print("%-10s %12s" % ("feature", "coef"))
    print("-" * 56)
    for _, row in coef_df.iterrows():             # iterrows : 한 줄씩 꺼낸다
        n = int(20 * abs(row['coef']) / scale)
        # 양수는 오른쪽, 음수는 왼쪽으로 뻗는 막대를 만든다
        bar = (" " * (20 - n) + "▓" * n + "│") if row['coef'] < 0 else ("│" + "█" * n)
        print("%-10s %12.3f %s" % (row['feature'], row['coef'], bar))
    return coef_df

ridge = Ridge(alpha=0.1, random_state=SEED)
ridge.fit(x_train, y_train)
plot_coef(x_train.columns, ridge.coef_)


# %% [Block 5] 5단원 — Ridge : alpha에 따른 계수의 변화 — 규제가 아주 강한 모델과 아주 약한 모델을 각각 만든다
# 규제가 아주 강한 모델과 아주 약한 모델을 각각 만든다
ridge_100 = Ridge(alpha=100, random_state=SEED)
ridge_100.fit(x_train, y_train)
ridge_pred_100 = ridge_100.predict(x_test)

ridge_001 = Ridge(alpha=0.001, random_state=SEED)
ridge_001.fit(x_train, y_train)
ridge_pred_001 = ridge_001.predict(x_test)

print("【 alpha=100 — 강한 규제 】")
plot_coef(x_train.columns, ridge_100.coef_)


# %% [Block 6] 5단원 — Ridge : alpha에 따른 계수의 변화
print("【 alpha=0.001 — 거의 규제 없음 】")
plot_coef(x_train.columns, ridge_001.coef_)


# %% [Block 7] 두 결과를 나란히 놓고 비교 — 계수의 '크기'가 얼마나 줄었는지 숫자로 확인
# 계수의 '크기'가 얼마나 줄었는지 숫자로 확인
print("%-14s %14s %14s %10s" % ("alpha", "계수의 L1 노름", "계수의 L2 노름", "0의 개수"))
print("-" * 58)
for a in [100, 10, 1, 0.1, 0.001]:
    model = Ridge(alpha=a, random_state=SEED)
    model.fit(x_train, y_train)
    c = model.coef_
    # 03-08-01에서 배운 두 노름으로 '가중치 전체의 크기'를 잰다
    print("%-14s %14.2f %14.2f %10d" % (
        a, np.sum(np.abs(c)), np.sqrt(np.sum(c ** 2)), int((c == 0).sum())))
print()
print("→ alpha가 100배 커지면 계수의 크기가 30배 넘게 줄어든다.")
print("→ 그런데 '0의 개수'는 계속 0이다. Ridge는 아무리 세게 눌러도 0으로 만들지 못한다!")
print("   03-08-02에서 '(1 - eta*lambda/n)을 계속 곱해도 0에 닿지 않는다'고 한 그대로다.")


# %% [Block 8] 6단원 — Lasso (L1 Regularization) — 값이 커질 수록 큰 규제입니다.
# 값이 커질 수록 큰 규제입니다.
alphas = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]

for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(x_train, y_train)
    pred = lasso.predict(x_test)
    add_model('Lasso(alpha={})'.format(alpha), pred, y_test)

plot_all()


# %% [Block 9] 7단원 — 하이라이트 : 계수가 0이 되는 순간
print("alpha를 키우면서 0이 되는 계수가 몇 개인지 세어본다")
print()
print("%-10s %-52s %8s" % ("alpha", "계수 10개 (0인 것은 · 로 표시)", "0의 개수"))
print("-" * 74)

for alpha in [0.001, 0.01, 0.1, 1, 5, 10]:
    lasso = Lasso(alpha=alpha)
    lasso.fit(x_train, y_train)
    c = lasso.coef_

    # 계수를 보기 좋게 문자열로 만든다. 0이면 가운데점으로 표시
    marks = "".join("  ·  " if v == 0 else "%5.0f" % v for v in c)
    print("%-10s %-52s %8d" % (alpha, marks, int((c == 0).sum())))

print()
print("→ alpha=0.001 에서는 10개 전부 살아 있다.")
print("→ alpha=1 이 되자 7개가 죽고 3개만 남았다. 이것이 '특성 선택'이다.")
print("→ alpha=10 에서는 10개 전부 0. 모델이 아무것도 못 배운 상태다.")


# %% [Block 10] 살아남은 특성은 무엇인가 — alpha=1 에서 살아남은 3개가 무엇인지 확인한다
# alpha=1 에서 살아남은 3개가 무엇인지 확인한다
lasso_1 = Lasso(alpha=1)
lasso_1.fit(x_train, y_train)

print("【 Lasso(alpha=1) — 10개 중 3개만 살아남았다 】")
plot_coef(x_train.columns, lasso_1.coef_)
print()
print("→ bmi(체질량지수), s5(혈청수치), bp(혈압) 세 개가 살아남았다.")
print("   Lasso가 '당뇨병 예측에 가장 중요한 세 가지'를 스스로 골라낸 것이다.")
print("   특성이 수백 개인 실무 데이터에서 이 기능은 대단히 유용하다.")


# %% [Block 11] 살아남은 특성은 무엇인가 — 마지막으로 Ridge와 Lasso를 같은 alpha에서 정면 비교한다
# 마지막으로 Ridge와 Lasso를 같은 alpha에서 정면 비교한다
print("%-8s | %-28s | %-28s" % ("alpha", "Ridge (0의 개수)", "Lasso (0의 개수)"))
print("-" * 72)
for alpha in [0.01, 0.1, 1, 10, 100]:
    r = Ridge(alpha=alpha, random_state=SEED).fit(x_train, y_train)
    l = Lasso(alpha=alpha).fit(x_train, y_train)
    r_zero, l_zero = int((r.coef_ == 0).sum()), int((l.coef_ == 0).sum())
    print("%-8s | %-28s | %-28s" % (
        alpha,
        "%2d개 / 10개  %s" % (r_zero, "▁" * (10 - r_zero)),
        "%2d개 / 10개  %s" % (l_zero, "▁" * (10 - l_zero))))
print()
print("(막대는 '살아남은 계수'의 개수를 나타낸다)")
print()
print("→ Ridge는 alpha를 100까지 올려도 0이 하나도 없다.")
print("→ Lasso는 alpha=1에서 벌써 7개를 제거하고, 10에서 전멸시킨다.")
print("→ 03-08-02의 마름모(꼭짓점이 축 위) vs 원(매끈함) 그림이 그대로 재현되었다.")


# %% [Block 12] 8단원 — ElasticNet : 두 규제를 섞기
alpha = 0.01
ratios = [0.2, 0.5, 0.8]

for ratio in ratios:
    elasticnet = ElasticNet(alpha=alpha, l1_ratio=ratio, random_state=SEED)
    elasticnet.fit(x_train, y_train)
    pred = elasticnet.predict(x_test)
    add_model('ElasticNet(l1_ratio={})'.format(ratio), pred, y_test)

# top=8 : 성적이 좋은 상위 8개만 본다 (모델이 너무 많아졌으므로)
plot_all(top=8)


# %% [Block 13] 8단원 — ElasticNet : 두 규제를 섞기 — l1_ratio가 0에서 1로 갈 때 무슨 일이 일어나는지 촘촘히 본다
# l1_ratio가 0에서 1로 갈 때 무슨 일이 일어나는지 촘촘히 본다
print("%-14s %12s %10s %s" % ("l1_ratio", "MSE", "0의 개수", "성격"))
print("-" * 62)
for ratio in [0.01, 0.2, 0.5, 0.8, 0.99]:
    en = ElasticNet(alpha=0.01, l1_ratio=ratio, random_state=SEED, max_iter=10000)
    en.fit(x_train, y_train)
    mse = mean_squared_error(y_test, en.predict(x_test))
    n_zero = int((en.coef_ == 0).sum())
    if ratio <= 0.2:
        role = "Ridge에 가까움"
    elif ratio >= 0.8:
        role = "Lasso에 가까움"
    else:
        role = "절반씩 섞임"
    print("%-14s %12.2f %10d   %s" % (ratio, mse, n_zero, role))
print()
print("→ l1_ratio를 0에서 1로 올릴수록 MSE가 개선되며 Lasso 쪽 성질이 강해진다.")
print("   이 데이터에서는 L1 성분이 많을수록 좋았다는 뜻이다.")
print("→ 즉 l1_ratio는 'Ridge와 Lasso 사이 어디에 설 것인가'를 정하는 다이얼이다.")


# %% [Block 14] 8단원 — ElasticNet : 두 규제를 섞기 — ElasticNet의 계수도 확인해 본다 (강의 자료는 alpha=5로 두 개를 비교했다)
# ElasticNet의 계수도 확인해 본다 (강의 자료는 alpha=5로 두 개를 비교했다)
elsticnet_20 = ElasticNet(alpha=5, l1_ratio=0.2, random_state=SEED)
elsticnet_20.fit(x_train, y_train)
elasticnet_pred_20 = elsticnet_20.predict(x_test)

elsticnet_80 = ElasticNet(alpha=5, l1_ratio=0.8, random_state=SEED)
elsticnet_80.fit(x_train, y_train)
elasticnet_pred_80 = elsticnet_80.predict(x_test)

print("l1_ratio=0.2 (Ridge 성향) 의 0의 개수 :", int((elsticnet_20.coef_ == 0).sum()), "/ 10")
print("l1_ratio=0.8 (Lasso 성향) 의 0의 개수 :", int((elsticnet_80.coef_ == 0).sum()), "/ 10")
print()
print("→ 같은 alpha라도 l1_ratio가 크면 더 많은 계수를 0으로 만든다.")
print("   L1 성분이 많아질수록 '제거하는 성질'이 강해지기 때문이다.")


# %% [Block 15] 9단원 — 총정리 : 무엇을 어떻게 고를까 — 감으로 고르지 말고 교차검증으로 고르는 방법
# 감으로 고르지 말고 교차검증으로 고르는 방법
from sklearn.model_selection import GridSearchCV

# param_grid : 시도해볼 값들의 목록. 딕셔너리로 준다
param_grid = {'alpha': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5]}

# cv=5 : 훈련 데이터를 5조각으로 나눠 5번 교차검증한다
#   scoring='neg_mean_squared_error' : sklearn은 '클수록 좋은' 점수를 쓰므로
#                                       MSE에 마이너스를 붙인 값을 쓴다
grid = GridSearchCV(Lasso(), param_grid, cv=5,
                    scoring='neg_mean_squared_error')
grid.fit(x_train, y_train)

print("교차검증이 고른 최적 alpha :", grid.best_params_)
print("그때의 검증 MSE           : %.2f" % (-grid.best_score_))
print()

# 이 alpha로 최종 시험 성적을 확인
best_lasso = grid.best_estimator_
final_mse = mean_squared_error(y_test, best_lasso.predict(x_test))
print("시험 데이터에서의 최종 MSE : %.2f" % final_mse)
print("살아남은 특성 개수         : %d / 10" % int((best_lasso.coef_ != 0).sum())) 
print()
print("→ 주목: 교차검증이 고른 alpha(0.001)는 우리가 3~7단원에서")
print("   '시험 데이터를 보고' 최고라 판단했던 alpha(0.1)와 다르다!")
print("→ 어느 쪽이 맞을까? 교차검증 쪽이 맞다.")
print("   alpha=0.1이 좋아 보였던 것은 '그 시험 데이터에서 우연히' 잘 맞은 것일 수 있다.")
print("   교차검증은 훈련 데이터를 5번 쪼개 평균을 내므로 우연에 훨씬 덜 휘둘린다.")
print("→ 이것이 하이퍼파라미터를 시험 데이터로 고르면 안 되는 이유를 보여주는 실물 증거다.")


# %% [Block 16] 실습 — 실습 1 정답 — 첫 계수가 죽는 alpha 찾기
# 실습 1 정답 — 첫 계수가 죽는 alpha 찾기
print("%-10s %10s %s" % ("alpha", "0의 개수", "변화"))
print("-" * 40)
prev = 0
for alpha in np.arange(0.1, 1.05, 0.1):
    lasso = Lasso(alpha=alpha)
    lasso.fit(x_train, y_train)
    n_zero = int((lasso.coef_ == 0).sum())
    mark = "  ← 새로 죽음!" if n_zero > prev else ""
    print("%-10.1f %10d%s" % (alpha, n_zero, mark))
    prev = n_zero


# %% [Block 17] 실습 — 실습 2 정답 — 잡음 특성 20개를 섞고 Lasso가 걸러내는지 본다
# 실습 2 정답 — 잡음 특성 20개를 섞고 Lasso가 걸러내는지 본다
rng = np.random.default_rng(0)

# 정답과 아무 상관없는 무작위 숫자 20열을 만든다
noise = pd.DataFrame(
    rng.normal(0, 0.05, (len(X), 20)),
    columns=["noise%02d" % i for i in range(20)],   # 리스트 컴프리헨션으로 이름 생성
    index=X.index)

# pd.concat([a, b], axis=1) : 두 표를 '옆으로' 이어붙인다
X_noisy = pd.concat([X, noise], axis=1)
xn_train, xn_test, yn_train, yn_test = train_test_split(
    X_noisy, y, test_size=0.25, random_state=SEED)

print("특성 개수 : 진짜 10개 + 잡음 20개 = %d개" % X_noisy.shape[1])
print()
print("%-14s %10s %14s %14s" % ("모델", "MSE", "살아남은 진짜", "살아남은 잡음"))
print("-" * 60)

for name, model in [("규제 없음", LinearRegression()),
                    ("Ridge(0.1)", Ridge(alpha=0.1, random_state=SEED)),
                    ("Lasso(0.1)", Lasso(alpha=0.1)),
                    ("Lasso(1)", Lasso(alpha=1))]:
    model.fit(xn_train, yn_train)
    mse = mean_squared_error(yn_test, model.predict(xn_test))
    c = model.coef_
    real_alive = int((c[:10] != 0).sum())      # 앞 10개가 진짜 특성
    noise_alive = int((c[10:] != 0).sum())     # 뒤 20개가 잡음
    print("%-14s %10.1f %10d/10 %12d/20" % (name, mse, real_alive, noise_alive))
print()
print("→ 규제가 없거나 Ridge면 잡음 20개 전부에 0이 아닌 가중치가 붙는다(20/20).")
print("   존재하지도 않는 패턴을 억지로 학습한 것이다.")
print("→ Lasso(0.1)은 잡음 20개 중 12개를 잘라내면서 MSE도 가장 좋다(2778).")
print("   잡음이 섞이자 Ridge(2905)와의 격차가 3단원 때보다 훨씬 벌어졌다.")
print("→ Lasso(1)은 잡음을 전멸시키지만(0/20) 진짜 특성도 3개만 남긴다.")
print("   규제가 지나치면 '잡음과 함께 신호도 잘라낸다'는 점을 잊지 말 것.")
print("→ 결론: 특성 중 상당수가 쓸모없을 때 Lasso의 이득이 가장 크게 나타난다.")


# %% [Block 18] 실습 — 실습 3 정답 — alpha와 l1_ratio를 동시에 탐색
# 실습 3 정답 — alpha와 l1_ratio를 동시에 탐색
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
}

grid = GridSearchCV(ElasticNet(random_state=SEED, max_iter=10000),
                    param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(x_train, y_train)

print("탐색한 조합 수 :", len(param_grid['alpha']) * len(param_grid['l1_ratio']), "가지")
print("최적 조합      :", grid.best_params_)
print("검증 MSE       : %.2f" % (-grid.best_score_))
print("시험 MSE       : %.2f" % mean_squared_error(y_test, grid.best_estimator_.predict(x_test)))
print()
print("→ 손잡이가 두 개면 조합이 20가지다. 손으로 다 해보는 것은 비현실적이다.")
print("   GridSearchCV가 이 지루한 일을 대신해 준다.")
