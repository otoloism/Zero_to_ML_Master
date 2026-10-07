# -*- coding: utf-8 -*-
"""
03-09-01 부스팅 알고리즘 부스팅의 개념과 배깅과의 차이

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-09장 - 부스팅 알고리즘/03-09-01 부스팅 알고리즘 부스팅의 개념과 배깅과의 차이.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 구현 관점 — "약한 학습기 하나"와 "강한 학습기 하나"를 실제로 비교해 봅니다.
# ─────────────────────────────────────────────────────────────
# "약한 학습기 하나"와 "강한 학습기 하나"를 실제로 비교해 봅니다.
# ─────────────────────────────────────────────────────────────
import numpy as np                                  # 수치 계산 도구 상자
from sklearn.datasets import load_breast_cancer     # 예제 데이터(유방암 진단)
from sklearn.tree import DecisionTreeClassifier     # 결정 트리 분류기
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = load_breast_cancer()                         # 데이터 불러오기
# train_test_split: 데이터를 학습용/시험용으로 쪼갭니다.
#   test_size=0.2  → 20%를 시험용으로 떼어 놓습니다.
#   random_state=156 → 쪼개는 방식을 고정(매번 같은 결과가 나오게)합니다.
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=156)
# ① 약한 학습기: 깊이 1 = 질문을 딱 한 번만 던지는 '그루터기'
weak = DecisionTreeClassifier(max_depth=1, random_state=0)
weak.fit(X_train, y_train)                          # fit = 학습(공부시키기)
weak_acc = accuracy_score(y_test, weak.predict(X_test))
# ② 강한 학습기: 깊이 제한 없음 = 훈련 데이터를 통째로 외우는 나무
strong = DecisionTreeClassifier(random_state=0)
strong.fit(X_train, y_train)
strong_acc = accuracy_score(y_test, strong.predict(X_test))
# 훈련 데이터에 대한 정확도도 같이 봅니다(외웠는지 확인용).
print("약한 학습기(깊이 1)  훈련 %.4f / 시험 %.4f"
      % (accuracy_score(y_train, weak.predict(X_train)), weak_acc))
print("강한 학습기(무제한)  훈련 %.4f / 시험 %.4f"
      % (accuracy_score(y_train, strong.predict(X_train)), strong_acc))
print("깊이 1 트리가 던진 단 하나의 질문:",
      data.feature_names[weak.tree_.feature[0]], "<=", round(weak.tree_.threshold[0], 4))


# %% [Block 2] 직관적 의미 — 왜 부스팅은 편향을 줄이나 — 트리를 하나씩 더할 때 훈련/시험 오차가 어떻게 변하는지 직접 봅니다.
# ─────────────────────────────────────────────────────────────
# 트리를 하나씩 더할 때 훈련/시험 오차가 어떻게 변하는지 직접 봅니다.
# ─────────────────────────────────────────────────────────────
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# ① 인공 데이터: y = 3x^2 + 잡음
np.random.seed(42)                 # 난수 씨앗 고정 → 매번 같은 데이터
X = np.random.rand(200, 1) - 0.5   # -0.5 ~ 0.5 사이의 값 200개
y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(200)   # randn = 표준정규분포 잡음
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
# ② 부스팅: 트리를 1개, 5개, 20개, 100개 더해 가며 오차 관찰
print("[부스팅] 트리를 더할수록 편향이 줄어드는가?")
print("  트리수 |  훈련 MSE  |  시험 MSE")
for n in [1, 5, 20, 100, 500]:
    m = GradientBoostingRegressor(max_depth=2, n_estimators=n,
                                  learning_rate=0.1, random_state=42)
    m.fit(Xtr, ytr)
    tr = mean_squared_error(ytr, m.predict(Xtr))   # 훈련 오차
    te = mean_squared_error(yte, m.predict(Xte))   # 시험 오차
    print("  %6d | %10.5f | %10.5f" % (n, tr, te))
# ③ 배깅(랜덤 포레스트): 나무를 더해도 시험 오차가 크게 안 변한다
print("\n[배깅] 나무를 더해도 오차가 요동치지 않는가?")
print("  나무수 |  훈련 MSE  |  시험 MSE")
for n in [1, 5, 20, 100, 500]:
    m = RandomForestRegressor(n_estimators=n, random_state=42)
    m.fit(Xtr, ytr)
    print("  %6d | %10.5f | %10.5f"
          % (n, mean_squared_error(ytr, m.predict(Xtr)),
             mean_squared_error(yte, m.predict(Xte))))


# %% [Block 3] 수학적 의미 — 라이브러리 없이 만드는 미니 그레이디언트 부스팅 (트리 3그루)
# ─────────────────────────────────────────────────────────────
# 라이브러리 없이 만드는 미니 그레이디언트 부스팅 (트리 3그루)
# 강의 자료 7페이지 그림을 코드로 재현합니다.
# ─────────────────────────────────────────────────────────────
import numpy as np
from sklearn.tree import DecisionTreeRegressor   # 회귀용 결정 트리
# ① 데이터 만들기: y = 3x^2 + 약간의 잡음
np.random.seed(42)                  # 난수 고정
X = np.random.rand(100, 1) - 0.5    # 입력: -0.5 ~ 0.5 사이 100개
                                    #   rand(100,1) → 0~1 사이 난수를 100행 1열로
                                    #   -0.5 를 빼서 중심을 0으로 옮깁니다.
y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)
#   X[:, 0]  → 2차원 배열에서 첫 번째 열만 꺼내 1차원으로 만듭니다.
#   ** 2     → 제곱 연산자
#   randn    → 평균 0, 표준편차 1인 정규분포 난수(=자연스러운 잡음)
# ② 1번 주자: 원래 정답 y 를 보고 학습
tree1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree1.fit(X, y)
# ③ 잔차 계산: "정답 - 1번 주자가 맞힌 것" = 아직 남은 거리
resid2 = y - tree1.predict(X)
# ④ 2번 주자: y 가 아니라 resid2(남은 거리)를 정답으로 학습 ★핵심★
tree2 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree2.fit(X, resid2)
# ⑤ 다시 남은 거리
resid3 = resid2 - tree2.predict(X)
# ⑥ 3번 주자
tree3 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree3.fit(X, resid3)
# ⑦ 단계별로 "남은 오차의 총량(제곱합)"이 얼마나 줄었는지 확인
print("단계별 남은 오차의 제곱합(작을수록 좋음)")
print("  시작(아무것도 안 함) : %8.4f" % np.sum(y ** 2))
print("  트리 1 학습 후        : %8.4f" % np.sum(resid2 ** 2))
print("  트리 2 학습 후        : %8.4f" % np.sum(resid3 ** 2))
print("  트리 3 학습 후        : %8.4f"
      % np.sum((resid3 - tree3.predict(X)) ** 2))
# ⑧ 새로운 입력 x=0.8 에 대한 최종 예측 = 세 트리의 예측을 그냥 더한다
X_new = np.array([[0.8]])
each = [t.predict(X_new)[0] for t in (tree1, tree2, tree3)]
#   리스트 컴프리헨션: (tree1, tree2, tree3)을 하나씩 t에 넣어 예측값을 모읍니다.
print("\n각 트리의 예측: %.4f + %.4f + %.4f" % tuple(each))
print("최종 예측 (합) : %.4f" % sum(each))


# %% [Block 4] 검증 — 사이킷런 결과와 같은가? — 우리가 손으로 만든 3그루 부스팅과
# 우리가 손으로 만든 3그루 부스팅과
# 사이킷런의 GradientBoostingRegressor(트리 3개, 학습률 1.0)가 같은 답을 내는지 확인합니다.
from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=2,          # 얕은 트리(약한 학습기)
                                 n_estimators=3,       # 트리 3그루
                                 learning_rate=1.0,    # 각 트리를 100% 반영
                                 random_state=42)
gbrt.fit(X, y)
manual = sum(t.predict(X_new)[0] for t in (tree1, tree2, tree3))
library = gbrt.predict(X_new)[0]
print("손으로 만든 부스팅  : %.10f" % manual)
print("사이킷런 GBRT       : %.10f" % library)
print("두 값의 차이        : %.2e" % abs(manual - library))


# %% [Block 5] 실습 — [실습 2 정답 예시] 30그루까지 늘려 남은 오차를 추적합니다.
# [실습 2 정답 예시] 30그루까지 늘려 남은 오차를 추적합니다.
resid = y.copy()          # copy(): 원본 y를 건드리지 않으려고 복사본을 씁니다.
trees, history = [], []
for m in range(30):                                  # 30번 반복
    t = DecisionTreeRegressor(max_depth=2, random_state=42)
    t.fit(X, resid)                                  # 남은 오차를 정답으로 학습
    resid = resid - t.predict(X)                     # 새 잔차로 갱신
    trees.append(t)
    history.append(np.sum(resid ** 2))               # 남은 오차의 제곱합 기록
print("트리 수별 남은 오차 제곱합")
for m in [0, 1, 2, 4, 9, 19, 29]:
    print("  %2d그루 후 : %9.5f" % (m + 1, history[m]))
print("\n→ 계속 줄지만 줄어드는 폭이 급격히 작아집니다.")
print("   이것이 '트리를 무한정 늘려도 이득이 없다'는 신호이고,")
print("   조기 종료(early stopping)가 필요한 이유입니다.")
