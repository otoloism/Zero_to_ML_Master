# -*- coding: utf-8 -*-
"""
03-08-03 대규모 데이터와 학습 곡선 (Learning with Large Datasets & Learning Curve)

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-08장 - 정규화 심화 및 대규모 학습/03-08-03 대규모 데이터와 학습 곡선.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 6단원 — 코드로 직접 그려 확인하기
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

def make_data(rng, n):
    """진짜 규칙이 3차 곡선인 데이터를 n개 만든다."""
    x = rng.uniform(-3, 3, n)                        # -3 ~ 3 사이 균등하게 뽑기
    y = 0.5 * x**3 - 2 * x + 1 + rng.normal(0, 3, n) # 3차 곡선 + 잡음(표준편차 3)
    #   x.reshape(-1, 1) : sklearn은 입력을 2차원(샘플수, 특성수)으로 요구한다.
    #                      -1 은 "나머지는 알아서 계산하라"는 뜻
    return x.reshape(-1, 1), y

def learning_curve(degree, m_list, trials=30, seed=0):
    """훈련 데이터 개수를 늘려가며 J_train, J_cv 를 측정한다."""
    rng = np.random.default_rng(seed)
    X_cv, y_cv = make_data(rng, 500)     # 검증 데이터는 500개로 '고정'

    result = []
    for m in m_list:
        tr_errors, cv_errors = [], []
        # 같은 m으로 여러 번 반복해 평균을 낸다(운에 좌우되지 않도록)
        for _ in range(trials):
            X_tr, y_tr = make_data(rng, m)
            # make_pipeline : 전처리와 모델을 한 줄로 이어 붙이는 도구
            #   PolynomialFeatures(d) : x를 x, x², ..., x^d 로 늘린다
            model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            model.fit(X_tr, y_tr)
            tr_errors.append(mean_squared_error(y_tr, model.predict(X_tr)))
            cv_errors.append(mean_squared_error(y_cv, model.predict(X_cv)))
        # 중앙값(median)을 쓰는 이유: 가끔 튀는 극단값에 평균이 휘둘리지 않도록
        result.append((m, np.mean(tr_errors), np.median(cv_errors)))
    return result

m_list = [20, 40, 80, 150, 300, 600]

print("【 9차 모델 — 너무 복잡한 그릇 】")
print("%6s %10s %10s %10s" % ("m", "J_train", "J_cv", "간격"))
print("-" * 40)
for m, tr, cv in learning_curve(9, m_list):
    print("%6d %10.2f %10.2f %10.2f" % (m, tr, cv, cv - tr))


# %% [Block 2] 6단원 — 코드로 직접 그려 확인하기
print("【 1차 모델 — 너무 단순한 그릇 】")
print("%6s %10s %10s %10s" % ("m", "J_train", "J_cv", "간격"))
print("-" * 40)
for m, tr, cv in learning_curve(1, m_list):
    print("%6d %10.2f %10.2f %10.2f" % (m, tr, cv, cv - tr))
print()
print("→ m=20에서 이미 간격이 1.3에 불과하고, m=600까지 가도 오차가 12 근처에서 꿈쩍 않는다.")
print("   데이터를 30배 늘렸는데 성능은 제자리다. 이것이 High bias다.")


# %% [Block 3] 실습 — 실습 1 정답 — 3차 모델(정답과 같은 복잡도)
# 실습 1 정답 — 3차 모델(정답과 같은 복잡도)
print("【 3차 모델 — 딱 맞는 그릇 】")
print("%6s %10s %10s %10s" % ("m", "J_train", "J_cv", "간격"))
print("-" * 40)
for m, tr, cv in learning_curve(3, m_list):
    print("%6d %10.2f %10.2f %10.2f" % (m, tr, cv, cv - tr))
print()
print("→ m=40부터 이미 간격이 거의 0이고, 도달 오차도 9(불가피 오차)에 붙어 있다.")
print("   '적은 데이터로도 빨리 좋아지고, 도달점도 낮다' — 이것이 이상적인 곡선이다.")


# %% [Block 4] 실습 — 실습 3 정답 — 9차 모델에 Ridge 규제를 걸면 과대적합이 잡힐까?
# 실습 3 정답 — 9차 모델에 Ridge 규제를 걸면 과대적합이 잡힐까?
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# 주의 1: alpha마다 '같은 데이터'로 비교해야 한다.
#         데이터가 매번 달라지면 alpha 효과인지 운인지 구분할 수 없다.
# 주의 2: 9차 다항 특성은 x^9까지 만들어 스케일이 수천 배 차이 난다.
#         03-08-02에서 배운 대로 '규제 전에 반드시 표준화'해야 한다.

def ridge_curve(alpha, use_scaler, m=20, trials=30):
    rng = np.random.default_rng(0)          # alpha마다 같은 씨앗 → 같은 데이터
    X_cv, y_cv = make_data(rng, 500)
    tr_list, cv_list = [], []
    for _ in range(trials):
        X_tr, y_tr = make_data(rng, m)
        if use_scaler:
            model = make_pipeline(PolynomialFeatures(9), StandardScaler(), Ridge(alpha=alpha))
        else:
            model = make_pipeline(PolynomialFeatures(9), Ridge(alpha=alpha))
        model.fit(X_tr, y_tr)
        tr_list.append(mean_squared_error(y_tr, model.predict(X_tr)))
        cv_list.append(mean_squared_error(y_cv, model.predict(X_cv)))
    return np.mean(tr_list), np.median(cv_list)

print("m=20 고정, 9차 모델. 표준화를 넣은 경우와 뺀 경우를 나란히 비교")
print("%10s | %12s | %12s" % ("alpha", "표준화 O J_cv", "표준화 X J_cv"))
print("-" * 42)
for alpha in [0.0001, 0.01, 1, 100]:
    _, cv_yes = ridge_curve(alpha, use_scaler=True)
    _, cv_no  = ridge_curve(alpha, use_scaler=False)
    print("%10s | %12.2f | %12.2f" % (alpha, cv_yes, cv_no))
print()
print("→ 표준화를 넣으면 J_cv가 22.7 → 13.6 → 10.5 로 내려갔다가 다시 11.5로 올라간다.")
print("   alpha=1 부근이 바닥인 U자 곡선이다. 03-04-01에서 본 그 모양이다.")
print("   데이터를 한 개도 더 모으지 않았는데 규제만으로 과대적합이 크게 잡혔다.")
print("→ 표준화를 빼면 같은 alpha에서 J_cv가 훨씬 높다(1일 때 10.5 vs 19.8).")
print("   규제가 x^9 같은 큰 스케일 특성만 때리느라 제 역할을 못 한 것이다.")
print("   03-08-02에서 강조한 '규제 전 표준화'가 왜 필수인지 숫자로 드러난다.")
