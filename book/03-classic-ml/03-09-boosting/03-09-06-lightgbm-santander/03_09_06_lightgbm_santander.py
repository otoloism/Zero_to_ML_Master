# -*- coding: utf-8 -*-
"""
03-09-06 LightGBM과 Santander 종합 실습

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-09장 - 부스팅 알고리즘/03-09-06 LightGBM과 Santander 종합 실습.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 왜 같은 잎 개수로 손실이 더 낮은가 — 두 성장 방식이 같은 잎 개수로 얼마나 다른 손실에 도달하는지
# ─────────────────────────────────────────────────────────────
# 두 성장 방식이 같은 잎 개수로 얼마나 다른 손실에 도달하는지
# 실제 라이브러리로 확인합니다.
# ─────────────────────────────────────────────────────────────
import warnings, time
warnings.filterwarnings('ignore')
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
X, y = make_classification(n_samples=30000, n_features=40, n_informative=15,
                           n_redundant=10, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
print('  모델                        | 학습 시간 | 시험 AUC')
print('  ----------------------------+-----------+----------')
t0 = time.time()
xm = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                   random_state=0, verbosity=0).fit(Xtr, ytr)
print('  XGBoost (level-wise, d=6)   | %7.2f초 | %8.4f'
      % (time.time() - t0, roc_auc_score(yte, xm.predict_proba(Xte)[:, 1])))
t0 = time.time()
lm = LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.1,
                    random_state=0, verbose=-1).fit(Xtr, ytr)
#   verbose=-1 → LightGBM 의 안내 메시지를 끕니다.
print('  LightGBM (leaf-wise, 31잎)  | %7.2f초 | %8.4f'
      % (time.time() - t0, roc_auc_score(yte, lm.predict_proba(Xte)[:, 1])))
print()
print('※ 깊이 6 짜리 완전 이진 트리의 잎 개수는 2^6 = %d 개입니다.' % (2 ** 6))
print('   LightGBM 의 num_leaves 기본값 31 은 그보다 적은데도 비슷하거나 나은 성능을 냅니다.')
print('   "필요한 곳에만 잎을 만들었기 때문" 입니다.')


# %% [Block 2] 3단계 — 유방암 데이터 실습 — 강의 자료 52~55페이지 — 강의 자료 52페이지 코드
# ─────────────────────────────────────────────────────────────
# 강의 자료 52페이지 코드
# ─────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')
from lightgbm import LGBMClassifier      # LightGBM 불러오기
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                             precision_score, roc_auc_score, confusion_matrix)
print('LightGBM 버전 :', lgb.__version__)
dataset = load_breast_cancer()
ftr = dataset.data
target = dataset.target
# 예제 데이터 세트 중 80%를 학습, 20%를 테스트 데이터셋으로 분할
X_train, X_test, y_train, y_test = train_test_split(
    ftr, target, test_size=0.2, random_state=156)
# 트리개수는 400개로 지정
lgbm_wrapper = LGBMClassifier(n_estimators=400, random_state=0, verbose=-1)
# LGBM도 XGBoost처럼 early stopping 가능
# ※ 최신 LightGBM(4.x)에서는 fit()의 early_stopping_rounds 인자가 제거되고
#    callbacks 방식으로 바뀌었습니다.
evals = [(X_test, y_test)]
lgbm_wrapper.fit(X_train, y_train,
                 eval_set=evals, eval_metric='logloss',
                 callbacks=[lgb.early_stopping(100, verbose=False),
                            lgb.log_evaluation(0)])
preds = lgbm_wrapper.predict(X_test)
print('설정한 최대 트리 수 : 400')
print('조기 종료로 실제 사용된 트리 수 : %d' % lgbm_wrapper.best_iteration_)
print('그때의 검증 logloss : %.6f'
      % lgbm_wrapper.best_score_['valid_0']['binary_logloss'])


# %% [Block 3] 3단계 — 유방암 데이터 실습 — 강의 자료 52~55페이지 — 강의 자료 54페이지 — 평가 함수 (05절과 동일)
# ─────────────────────────────────────────────────────────────
# 강의 자료 54페이지 — 평가 함수 (05절과 동일)
# ─────────────────────────────────────────────────────────────
def get_clf_eval(y_test, y_pred):
    """혼동행렬, 정확도, 정밀도, 재현율, F1, AUC 를 한 번에 출력"""
    confusion = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    print('오차행렬:\n', confusion)
    print('\n정확도: {:.4f}'.format(accuracy))
    print('정밀도: {:.4f}'.format(precision))
    print('재현율: {:.4f}'.format(recall))
    print('F1: {:.4f}'.format(F1))
    print('AUC: {:.4f}'.format(AUC))
get_clf_eval(y_test, preds)


# %% [Block 4] 3단계 — 유방암 데이터 실습 — 강의 자료 52~55페이지 — 강의 자료 55페이지 — 피처 중요도
# ─────────────────────────────────────────────────────────────
# 강의 자료 55페이지 — 피처 중요도
# plot_importance( )를 이용해 피처 중요도 시각화
# ─────────────────────────────────────────────────────────────
# from lightgbm import plot_importance
# import matplotlib.pyplot as plt
# %matplotlib inline
# f, ax = plt.subplots(figsize=(6, 6))
# plot_importance(lgbm_wrapper, max_num_features=15, ax=ax)   # 상위 15개만 조회
# 그래프 대신 표로 출력합니다.
imp = sorted(zip(dataset.feature_names, lgbm_wrapper.feature_importances_),
             key=lambda kv: -kv[1])[:15]
print('상위 15개 피처 중요도 (분할에 쓰인 횟수 기준)')
for name, val in imp:
    bar = '█' * int(val / 3)
    print('  %-24s %4d %s' % (name, val, bar))
print()
print('※ LightGBM 의 feature_importances_ 기본값은 XGBoost 와 달리')
print('   gain 이 아니라 split(분할 횟수) 기준입니다.')
print('   gain 기준을 보려면 importance_type="gain" 으로 모델을 만드세요.')


# %% [Block 5] 4단계 — Santander 데이터 — 실전 데이터의 얼굴 — Santander 와 같은 성질을 가진 데이터를 합성합니다.
# ─────────────────────────────────────────────────────────────
# Santander 와 같은 성질을 가진 데이터를 합성합니다.
# (원본: https://www.kaggle.com/c/santander-customer-satisfaction)
# ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=20000, n_features=60,
                           n_informative=12, n_redundant=20,
                           weights=[0.96, 0.04],   # ★ 96% : 4% 불균형
                           flip_y=0.02, class_sep=0.8, random_state=0)
cols = ['var%d' % i for i in range(60)]
cust_df = pd.DataFrame(X, columns=cols)
cust_df['TARGET'] = y
# ★ var3 에 -999999 를 심습니다 (원본 데이터의 함정 재현)
rng = np.random.default_rng(0)
idx = rng.choice(len(cust_df), 120, replace=False)
cust_df.loc[idx, 'var3'] = -999999.0
cust_df.insert(0, 'ID', np.arange(1, len(cust_df) + 1))   # ID 컬럼 추가
print('데이터셋 형태: ', cust_df.shape)
print()
print(cust_df[['ID', 'var3', 'var15', 'var16', 'TARGET']].head(3).to_string())


# %% [Block 6] 4단계 — Santander 데이터 — 실전 데이터의 얼굴 — 강의 자료 57~59페이지 — 데이터 살펴보기
# ─────────────────────────────────────────────────────────────
# 강의 자료 57~59페이지 — 데이터 살펴보기
# ─────────────────────────────────────────────────────────────
print('[1] 결측치 개수 :', cust_df.isnull().sum().sum())
print('    → 0 입니다. 하지만 "결측치가 없다"고 결론내면 안 됩니다!')
print()
print('[2] TARGET 분포')
print(cust_df['TARGET'].value_counts().to_string())
unsatisfied_cnt = cust_df[cust_df['TARGET'] == 1]['TARGET'].count()
total_cnt = cust_df['TARGET'].count()
print('unsatisfied한 비율은 {:.2f}%'.format(unsatisfied_cnt / total_cnt * 100))
print()
print('[3] var3 의 기술통계 — 여기서 이상함을 발견합니다')
print(cust_df['var3'].describe().to_string())
print()
print('[4] var3 값 확인 (강의 자료 61페이지)')
print(cust_df['var3'].value_counts(ascending=False).head(5).to_string())
print()
print('→ 대부분의 값은 정상 범위인데 -999999 만 %d건 튀어나옵니다.'
      % (cust_df['var3'] == -999999).sum())
print('   평균이 %.1f 로 왜곡되어 있는 것이 그 증거입니다.'
      % cust_df['var3'].mean())


# %% [Block 7] 4단계 — Santander 데이터 — 실전 데이터의 얼굴 — 강의 자료 62페이지 — 전처리
# ─────────────────────────────────────────────────────────────
# 강의 자료 62페이지 — 전처리
# ─────────────────────────────────────────────────────────────
# -999999 를 최빈값(또는 중앙값)으로 바꿔 줍니다.
# 강의 자료는 2로 바꿨습니다(원본 데이터의 최빈값이 2였기 때문).
normal = cust_df.loc[cust_df['var3'] != -999999, 'var3']
fill_value = normal.median()          # 여기서는 중앙값을 씁니다
cust_df['var3'] = cust_df['var3'].replace(-999999, fill_value)
print('-999999 를 %.4f (중앙값) 로 대체했습니다.' % fill_value)
print('대체 후 var3 평균 : %.4f' % cust_df['var3'].mean())
# ID 는 예측에 아무 도움이 안 되므로 버립니다 (오히려 과적합의 원인)
cust_df = cust_df.drop(['ID'], axis=1)
#   axis=1 → 열(column)을 지운다는 뜻. axis=0 이면 행을 지웁니다.
# 피처세트와 레이블 세트를 분리
X_features = cust_df.iloc[:, :-1]     # 마지막 열(TARGET) 앞까지가 피처
y_labels = cust_df.iloc[:, -1]        # 마지막 열이 정답
#   iloc[행, 열] → 위치(정수)로 잘라내기. -1 은 마지막을 뜻합니다.
print('피처 데이터 세트 형태: {}'.format(X_features.shape))


# %% [Block 8] 4단계 — Santander 데이터 — 실전 데이터의 얼굴 — 강의 자료 63페이지 — 학습/테스트 분할과 분포 확인
# ─────────────────────────────────────────────────────────────
# 강의 자료 63페이지 — 학습/테스트 분할과 분포 확인
# ─────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_labels, test_size=0.2, random_state=0)
train_cnt = y_train.count()
test_cnt = y_test.count()
print('학습 세트 형태: {}, 테스트 세트 형태: {}'
      .format(X_train.shape, X_test.shape))
print('\n학습 세트 레이블 값 분포 비율: ')
print(y_train.value_counts() / train_cnt)
print('\n테스트 세트 레이블 값 분포 비율: ')
print(y_test.value_counts() / test_cnt)
print()
print('→ 두 세트의 1 비율이 비슷해야 합니다. 크게 어긋나면')
print('   train_test_split 에 stratify=y_labels 를 주어 계층적 분할을 하세요.')


# %% [Block 9] ① 기준선 — 강의 자료 64페이지 — 강의 자료 64페이지 — 기본 설정으로 기준선을 만듭니다.
# ─────────────────────────────────────────────────────────────
# 강의 자료 64페이지 — 기본 설정으로 기준선을 만듭니다.
# ─────────────────────────────────────────────────────────────
import warnings, time
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
# n_estimators = 500, random_state = 156
# 성능평가 지표를 'auc', 조기중단은 100회로 설정
# ※ 최신 XGBoost 에서는 early_stopping_rounds/eval_metric 을 생성자에 넣습니다.
t0 = time.time()
xgb_clf = XGBClassifier(n_estimators=500, random_state=156,
                        early_stopping_rounds=100, eval_metric='auc')
xgb_clf.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
xgb_auc_score = roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:, 1],
                              average='macro')
#   predict_proba(X)[:, 1] → '1일 확률'만 꺼냅니다. AUC 는 확률이 필요합니다.
print('조기 중단 지점 : %d 라운드' % xgb_clf.best_iteration)
print('ROC AUC: {:.4f}'.format(xgb_auc_score))
print('학습 시간: %.1f초' % (time.time() - t0))
print()
print('※ 참고 — 정확도로 재면 얼마나 나올까요?')
from sklearn.metrics import accuracy_score
print('   정확도 : %.4f' % accuracy_score(y_test, xgb_clf.predict(X_test)))
print('   "전부 0으로 찍기" 정확도 : %.4f' % (1 - y_test.mean()))
print('   → 정확도로는 두 모델이 거의 구분되지 않습니다. 그래서 AUC 를 씁니다.')


# %% [Block 10] ② 구조 파라미터 탐색 — 강의 자료 65~66페이지 — 강의 자료 65페이지 — GridSearchCV
# ─────────────────────────────────────────────────────────────
# 강의 자료 65페이지 — GridSearchCV
# ─────────────────────────────────────────────────────────────
import time
from sklearn.model_selection import GridSearchCV
# 하이퍼 파라미터의 테스트 수행속도를 높이기 위해 n_estimators를 100으로 감소
xgb_clf = XGBClassifier(n_estimators=100, random_state=156)
params = {'max_depth': [5, 7],
          'min_child_weight': [1, 3],
          'colsample_bytree': [0.5, 0.75]}
#   2 × 2 × 2 = 8가지 조합, cv=3 이므로 총 24번 학습합니다.
t0 = time.time()
gridcv = GridSearchCV(xgb_clf, param_grid=params, cv=3, scoring='roc_auc')
gridcv.fit(X_train, y_train)
print('Grid Search 최적 파라미터: ', gridcv.best_params_)
print('교차검증 최고 ROC AUC : {:.4f}'.format(gridcv.best_score_))
xgb_auc_score = roc_auc_score(y_test, gridcv.predict_proba(X_test)[:, 1],
                              average='macro')
print('테스트 ROC AUC:{:.4f}'.format(xgb_auc_score))
print('탐색 시간: %.1f초' % (time.time() - t0))


# %% [Block 11] ③ 학습률 낮추고 규제 추가 — 강의 자료 67페이지 — 강의 자료 67페이지 — 최종 모델
# ─────────────────────────────────────────────────────────────
# 강의 자료 67페이지 — 최종 모델
# n_estimators = 1000, learning_rate = 0.02, reg_alpha = 0.03
# ─────────────────────────────────────────────────────────────
import time
best = gridcv.best_params_
t0 = time.time()
xgb_clf = XGBClassifier(n_estimators=1000,
                        colsample_bytree=best['colsample_bytree'],
                        max_depth=best['max_depth'],
                        min_child_weight=best['min_child_weight'],
                        learning_rate=0.02,       # ★ 보폭을 크게 줄이고
                        reg_alpha=0.03,           # ★ L1 규제 추가
                        random_state=152,
                        early_stopping_rounds=200,  # ★ 인내심도 크게
                        eval_metric='auc')
# 성능지표를 auc로, 조기중단 파라미터 값은 200으로 설정하고 학습수행
xgb_clf.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
xgb_auc_score = roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:, 1],
                              average='macro')
print('조기 중단 지점 : %d 라운드' % xgb_clf.best_iteration)
print('ROC AUC: {:.4f}'.format(xgb_auc_score))
print('학습 시간: %.1f초' % (time.time() - t0))
print()
print('→ 학습률을 0.1 → 0.02 로 낮추자 필요한 트리 수가 크게 늘었습니다.')
print('   04절에서 배운 "η × M ≈ 일정" 이 실전 데이터에서도 그대로 나타납니다.')


# %% [Block 12] ③ 학습률 낮추고 규제 추가 — 강의 자료 67페이지 — 강의 자료 68페이지 — 피처 중요도 상위 20개
# ─────────────────────────────────────────────────────────────
# 강의 자료 68페이지 — 피처 중요도 상위 20개
# ─────────────────────────────────────────────────────────────
# from xgboost import plot_importance
# import matplotlib.pyplot as plt
# %matplotlib inline
# f, ax = plt.subplots(figsize=(6, 6))
# plot_importance(xgb_clf, max_num_features=20, ax=ax)   # 상위 20개만 조회
imp = sorted(zip(X_features.columns, xgb_clf.feature_importances_),
             key=lambda kv: -kv[1])[:20]
print('상위 20개 피처 중요도')
for name, val in imp:
    bar = '█' * int(val * 300)
    print('  %-10s %.4f %s' % (name, val, bar))
print()
print('※ 원본 Santander 데이터에서는 var38(잔고 추정치)과 var15(나이 추정치)가')
print('   압도적인 1·2위였습니다. 컬럼명이 익명화되어 있어도, 중요도 상위 피처를')
print('   따로 뜯어 보면 그 의미를 역추적할 수 있는 경우가 많습니다.')


# %% [Block 13] 6단계 — 세 모델 총정리 — 같은 데이터에서 나란히 재기 — GBM vs XGBoost vs LightGBM — 같은 데이터, 같은 조건으로 비교
# ─────────────────────────────────────────────────────────────
# GBM vs XGBoost vs LightGBM — 같은 데이터, 같은 조건으로 비교
# ─────────────────────────────────────────────────────────────
import warnings, time
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
results = []
def bench(name, model):
    """모델을 학습시키고 (이름, 학습시간, AUC) 를 기록합니다."""
    t0 = time.time()
    model.fit(X_train, y_train)
    dt = time.time() - t0
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    results.append((name, dt, auc))
    return dt, auc
bench('GradientBoosting (sklearn)',
      GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                 max_depth=3, random_state=0))
bench('HistGradientBoosting',
      HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                     max_depth=3, random_state=0))
bench('XGBoost',
      XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=3,
                    random_state=0, verbosity=0))
bench('LightGBM',
      LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=3,
                     random_state=0, verbose=-1))
print('  모델                        | 학습 시간 |  ROC AUC | 상대 속도')
print('  ----------------------------+-----------+----------+----------')
base = results[0][1]
for name, dt, auc in results:
    print('  %-27s | %7.2f초 | %8.4f | %7.1f배'
          % (name, dt, auc, base / dt))
print()
print('→ AUC 는 비슷한데 학습 시간은 크게 다릅니다.')
print('   "성능이 아니라 속도로 고르는 것" 이 부스팅 라이브러리 선택의 현실입니다.')


# %% [Block 14] 실습 — [실습 1 정답 예시] num_leaves 와 과대적합
# [실습 1 정답 예시] num_leaves 와 과대적합
import warnings
warnings.filterwarnings('ignore')
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
print('  num_leaves | 훈련 AUC | 시험 AUC | 격차')
print('  -----------+----------+----------+--------')
for nl in [15, 31, 63, 127, 255]:
    m = LGBMClassifier(n_estimators=200, num_leaves=nl, learning_rate=0.05,
                       random_state=0, verbose=-1).fit(X_train, y_train)
    tr = roc_auc_score(y_train, m.predict_proba(X_train)[:, 1])
    te = roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
    print('  %10d | %8.4f | %8.4f | %6.4f' % (nl, tr, te, tr - te))
print()
print('→ 읽는 법 두 가지.')
print('  ① 훈련 AUC 가 1.0000 에 붙어 버렸다면 그 지점부터 모델은 훈련 데이터를 "외운" 것입니다.')
print('     여기서는 num_leaves=31 부터 이미 1.0 입니다.')
print('  ② 그런데 시험 AUC 는 단조롭게 떨어지지 않습니다. 오르내리며 흔들립니다.')
print('     "num_leaves 를 키우면 무조건 나빠진다" 가 아니라')
print('     "훈련 AUC 가 1.0 이 된 뒤로는 더 키워 봐야 얻을 게 없다" 가 정확한 결론입니다.')
print('  → 그래서 판단 기준은 시험 AUC 하나가 아니라 훈련-시험 "격차" 입니다.')
print('     격차가 0.10 이상으로 벌어져 있다면 num_leaves 를 줄이거나')
print('     min_child_samples / reg_lambda 로 제동을 걸어야 합니다.')


# %% [Block 15] 실습 — [실습 3 정답 예시] -999999 를 그대로 두면 어떻게 될까?
# [실습 3 정답 예시] -999999 를 그대로 두면 어떻게 될까?
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
# (가) 원래 데이터를 다시 만들어 -999999 를 살려 둡니다
raw = cust_df.copy()
rng2 = np.random.default_rng(0)
bad_idx = rng2.choice(len(raw), 120, replace=False)
raw.loc[raw.index[bad_idx], 'var3'] = -999999.0
Xa = raw.iloc[:, :-1]; ya = raw.iloc[:, -1]
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(Xa, ya, test_size=0.2, random_state=0)
ma = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                   random_state=0, verbosity=0).fit(Xa_tr, ya_tr)
auc_raw = roc_auc_score(ya_te, ma.predict_proba(Xa_te)[:, 1])
# (나) 대체한 데이터 (이미 전처리된 X_features)
mb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                   random_state=0, verbosity=0).fit(X_train, y_train)
auc_fix = roc_auc_score(y_test, mb.predict_proba(X_test)[:, 1])
print('-999999 를 그대로 둔 경우 : ROC AUC %.4f' % auc_raw)
print('중앙값으로 대체한 경우    : ROC AUC %.4f' % auc_fix)
print('차이                      : %+.4f' % (auc_fix - auc_raw))
print()
print('→ 차이가 크지 않을 수 있습니다. 트리는 "값의 크기"가 아니라 "순서"만 보고,')
print('   -999999 는 어차피 맨 왼쪽 구간으로 몰리기 때문입니다.')
print('   하지만 ① describe() 같은 통계가 오염되고 ② 선형 모델·거리 기반 모델에서는')
print('   치명적이며 ③ 스케일링을 하면 다른 값들이 전부 뭉개집니다.')
print('   그러므로 "트리를 쓰니까 괜찮다"가 아니라 "발견하면 반드시 고친다"가 맞습니다.')
