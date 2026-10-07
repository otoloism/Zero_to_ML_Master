# -*- coding: utf-8 -*-
"""
03-09-05 XGBoost — 복잡도까지 값을 매기는 부스팅

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-09장 - 부스팅 알고리즘/03-09-05 XGBoost — 복잡도까지 값을 매기는 부스팅.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] ④ Gain 공식 — 벽을 틀 것인가 — Gain 공식을 손으로 계산해 봅니다. 숫자가 작으면 수식이 투명해집니다.
# ─────────────────────────────────────────────────────────────
# Gain 공식을 손으로 계산해 봅니다. 숫자가 작으면 수식이 투명해집니다.
# 상황: 어떤 잎에 샘플 6개가 있고, 이것을 3+3 으로 나눌지 결정합니다.
# ─────────────────────────────────────────────────────────────
import numpy as np
# MSE 손실을 쓴다고 가정 → g_i = 예측 - 정답 = -잔차,  h_i = 1
residual = np.array([2.0, 3.0, 2.5, -1.0, -2.0, -1.5])   # 각 샘플의 잔차
g = -residual                # 1차 미분 (부호 주의: g = -잔차)
h = np.ones(6)               # 2차 미분 (MSE 는 항상 1)
lam = 1.0                    # reg_lambda
gamma = 0.5                  # 분할 비용
def leaf_score(gs, hs, lam):
    """잎 하나의 '구조 점수' = G²/(H+λ). 클수록 그 잎이 손실을 많이 줄인다."""
    G, H = gs.sum(), hs.sum()
    return G ** 2 / (H + lam)
def leaf_weight(gs, hs, lam):
    """잎이 내놓을 최적 예측값 w* = -G/(H+λ)"""
    return -gs.sum() / (hs.sum() + lam)
# ① 나누지 않은 경우
score_all = leaf_score(g, h, lam)
print('나누기 전 : 잎 1개')
print('  G = %.2f, H = %.2f' % (g.sum(), h.sum()))
print('  구조 점수 G²/(H+λ) = %.4f' % score_all)
print('  잎의 예측값 w*      = %.4f' % leaf_weight(g, h, lam))
# ② 앞 3개 / 뒤 3개로 나눈 경우
gL, hL = g[:3], h[:3]
gR, hR = g[3:], h[3:]
score_L = leaf_score(gL, hL, lam)
score_R = leaf_score(gR, hR, lam)
print('\n나눈 후 : 잎 2개')
print('  왼쪽  G=%.2f H=%.2f → 점수 %.4f, w*=%.4f'
      % (gL.sum(), hL.sum(), score_L, leaf_weight(gL, hL, lam)))
print('  오른쪽 G=%.2f H=%.2f → 점수 %.4f, w*=%.4f'
      % (gR.sum(), hR.sum(), score_R, leaf_weight(gR, hR, lam)))
# ③ Gain 계산
gain = 0.5 * (score_L + score_R - score_all) - gamma
print('\nGain = 0.5 × (%.4f + %.4f − %.4f) − γ(%.2f) = %.4f'
      % (score_L, score_R, score_all, gamma, gain))
print('판정 : %s' % ('분할한다 (Gain > 0)' if gain > 0 else '분할하지 않는다 (Gain ≤ 0)'))
# ④ gamma 를 키우면 언제 분할을 포기할까?
print('\n  gamma |   Gain   | 판정')
print('  ------+----------+---------')
for gm in [0.0, 0.5, 2.0, 5.0, 10.0]:
    gn = 0.5 * (score_L + score_R - score_all) - gm
    print('  %5.1f | %8.4f | %s' % (gm, gn, '분할' if gn > 0 else '중단'))
print('\n→ gamma 를 키우면 어느 순간부터 트리가 자라기를 멈춥니다.')
print('   이것이 "복잡도를 비용함수에 넣는다"의 실제 작동 방식입니다.')


# %% [Block 2] λ의 효과도 숫자로 — reg_lambda 를 키우면 잎의 예측값이 정말 0 쪽으로 끌려갈까요?
# reg_lambda 를 키우면 잎의 예측값이 정말 0 쪽으로 끌려갈까요?
print('  lambda | 왼쪽 잎 w* | 오른쪽 잎 w* | Gain(γ=0)')
print('  -------+------------+--------------+----------')
for lm in [0.0, 1.0, 5.0, 20.0, 100.0]:
    wl = leaf_weight(gL, hL, lm)
    wr = leaf_weight(gR, hR, lm)
    gn = 0.5 * (leaf_score(gL, hL, lm) + leaf_score(gR, hR, lm)
                - leaf_score(g, h, lm))
    print('  %6.1f | %10.4f | %12.4f | %8.4f' % (lm, wl, wr, gn))
print()
print('→ λ가 커질수록 w* 가 0 쪽으로 수축하고, Gain 도 작아져 분할이 덜 일어납니다.')
print('   λ는 "잎 값의 크기"를, γ는 "잎의 개수"를 각각 억제합니다.')


# %% [Block 3] ① 데이터 준비 — 강의 자료 34페이지 코드
# ─────────────────────────────────────────────────────────────
# 강의 자료 34페이지 코드
# ─────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb                       # XGBoost 불러오기
from xgboost import plot_importance          # Feature Importance를 불러오기 위함
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score,
                             f1_score, roc_auc_score)
print('XGBoost 버전 :', xgb.__version__)
dataset = load_breast_cancer()
X_features = dataset.data        # 특징(피처) 30개
y_label = dataset.target         # 정답: 0=악성(malignant), 1=양성(benign)
# DataFrame 으로 만들면 표처럼 보기 편합니다.
cancer_df = pd.DataFrame(data=X_features, columns=dataset.feature_names)
cancer_df['target'] = y_label
print('데이터 형태 :', cancer_df.shape)
print('\n앞 3개 행의 일부 컬럼:')
print(cancer_df[['mean radius', 'mean texture', 'mean perimeter',
                 'mean area', 'target']].head(3).to_string())


# %% [Block 4] ① 데이터 준비 — 강의 자료 36페이지 — 레이블 분포 확인
# 강의 자료 36페이지 — 레이블 분포 확인
print(dataset.target_names)              # ['malignant' 'benign']
print(cancer_df['target'].value_counts())
print()
print('악성(0) : %d개  /  양성(1) : %d개'
      % ((y_label == 0).sum(), (y_label == 1).sum()))
print('양성 비율 : %.1f%%' % (y_label.mean() * 100))
print('→ 심하게 불균형하지는 않으므로 scale_pos_weight 는 굳이 쓰지 않아도 됩니다.')


# %% [Block 5] ① 데이터 준비 — 강의 자료 37페이지 — 학습/테스트 분할
# 강의 자료 37페이지 — 학습/테스트 분할
# 전체 데이터셋을 학습용 80%, 테스트용 20%로 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_label, test_size=0.2, random_state=156)
print(X_train.shape, X_test.shape)


# %% [Block 6] ② 파이썬 래퍼 방식 — DMatrix와 xgb.train — 강의 자료 38페이지
# ─────────────────────────────────────────────────────────────
# 강의 자료 38페이지
# 넘파이 형태의 학습 데이터 세트와 테스트 데이터를 DMatrix로 변환하는 예제
# ─────────────────────────────────────────────────────────────
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test, label=y_test)
#   DMatrix: XGBoost 전용 자료형입니다.
#   내부적으로 값을 미리 정렬·압축해 두어 학습이 빨라집니다.
# max_depth = 3, 학습률은 0.1, 예제가 이진분류이므로
# 목적함수(objective)는 binary:logistic(이진 로지스틱)
# 오류함수의 평가성능지표는 logloss
# 부스팅 반복횟수는 400
params = {'max_depth': 3,
          'eta': 0.1,                       # 파이썬 래퍼에서는 learning_rate 가 아니라 eta
          'objective': 'binary:logistic',
          'eval_metric': 'logloss'}
num_rounds = 400
print('설정한 파라미터:')
for k, v in params.items():
    print('  %-14s : %s' % (k, v))
print('  %-14s : %s' % ('num_boost_round', num_rounds))


# %% [Block 7] ② 파이썬 래퍼 방식 — DMatrix와 xgb.train — 강의 자료 39페이지 — 학습 수행
# ─────────────────────────────────────────────────────────────
# 강의 자료 39페이지 — 학습 수행
# train 데이터 세트는 'train', evaluation(test) 데이터 세트는 'eval' 로 명기
# ─────────────────────────────────────────────────────────────
wlist = [(dtrain, 'train'), (dtest, 'eval')]
# 하이퍼 파라미터와 early stopping 파라미터를 train() 함수의 파라미터로 전달
evals_result = {}                            # 학습 곡선을 담을 딕셔너리
xgb_model = xgb.train(params=params, dtrain=dtrain,
                      num_boost_round=num_rounds, evals=wlist,
                      evals_result=evals_result, verbose_eval=False)
#   verbose_eval=False → 400줄이 쏟아지지 않게 막고, 아래에서 골라 출력합니다.
tr = evals_result['train']['logloss']
ev = evals_result['eval']['logloss']
for i in [0, 1, 2, 3, 4, 5]:
    print('[%d]\ttrain-logloss:%.5f\teval-logloss:%.5f' % (i, tr[i], ev[i]))
print('\t\t...')
for i in [395, 396, 397, 398, 399]:
    print('[%d]\ttrain-logloss:%.5f\teval-logloss:%.5f' % (i, tr[i], ev[i]))
print()
print('훈련 logloss 는 %.5f 까지 떨어졌지만, 평가 logloss 는 %.5f 에서 멈췄습니다.'
      % (tr[-1], ev[-1]))
print('평가 logloss 의 최솟값은 %d번째 라운드의 %.5f 였습니다.'
      % (int(np.argmin(ev)), min(ev)))
print('→ 그 뒤로는 훈련 데이터만 더 외운 것입니다. (조기 종료가 필요한 이유)')


# %% [Block 8] ② 파이썬 래퍼 방식 — DMatrix와 xgb.train — 강의 자료 40페이지 — 예측
# ─────────────────────────────────────────────────────────────
# 강의 자료 40페이지 — 예측
# 파이썬 래퍼의 predict() 는 '확률'을 돌려줍니다 (사이킷런과 다른 점!)
# ─────────────────────────────────────────────────────────────
pred_probs = xgb_model.predict(dtest)
print('predict() 수행 결과값을 10개만 표시, 예측 확률 값으로 표시됨')
print(np.round(pred_probs[:10], 3))
# 예측 확률이 0.5보다 크면 1, 그렇지 않으면 0으로 예측값 결정해 리스트 객체인 preds에 저장
preds = [1 if x > 0.5 else 0 for x in pred_probs]
#   리스트 컴프리헨션: for 문을 한 줄로 쓴 것
#   [결과 for 변수 in 반복대상] 형태이고, if 를 붙여 삼항 연산자로 씁니다.
print('예측값 10개만 표시: ', preds[:10])


# %% [Block 9] ② 파이썬 래퍼 방식 — DMatrix와 xgb.train — 강의 자료 41페이지 — 평가 지표 함수
# ─────────────────────────────────────────────────────────────
# 강의 자료 41페이지 — 평가 지표 함수
# 혼동행렬, 정확도, 정밀도, 재현율, F1, AUC 불러오기
# ─────────────────────────────────────────────────────────────
def get_clf_eval(y_test, y_pred):
    """분류 성능 지표 6종을 한 번에 출력하는 도우미 함수."""
    confusion = confusion_matrix(y_test, y_pred)   # 오차행렬(혼동행렬)
    accuracy = accuracy_score(y_test, y_pred)      # 정확도 = 맞힌 비율
    precision = precision_score(y_test, y_pred)    # 정밀도 = 1이라 한 것 중 진짜 1
    recall = recall_score(y_test, y_pred)          # 재현율 = 진짜 1 중 찾아낸 비율
    F1 = f1_score(y_test, y_pred)                  # 정밀도와 재현율의 조화평균
    AUC = roc_auc_score(y_test, y_pred)            # ROC 곡선 아래 면적
    print('오차행렬:\n', confusion)
    print('\n정확도: {:.4f}'.format(accuracy))
    print('정밀도: {:.4f}'.format(precision))
    print('재현율: {:.4f}'.format(recall))
    print('F1: {:.4f}'.format(F1))
    print('AUC: {:.4f}'.format(AUC))
get_clf_eval(y_test, preds)


# %% [Block 10] ③ 사이킷런 래퍼 방식 — 강의 자료 44페이지 — 사이킷런 래퍼는 fit/predict 로 끝납니다.
# ─────────────────────────────────────────────────────────────
# 강의 자료 44페이지 — 사이킷런 래퍼는 fit/predict 로 끝납니다.
# max_depth = 3, 학습률은 0.1, 부스팅 반복횟수는 400
# ─────────────────────────────────────────────────────────────
from xgboost import XGBClassifier
xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
xgb_wrapper.fit(X_train, y_train)             # ← DMatrix 변환 불필요!
w_preds = xgb_wrapper.predict(X_test)         # ← 확률이 아니라 레이블이 바로 나옴
# 강의 자료 45페이지 — 예측 결과 확인
get_clf_eval(y_test, w_preds)
print()
print('→ 파이썬 래퍼와 완전히 같은 결과입니다. 같은 알고리즘이니 당연합니다.')
print('   확률이 필요하면 predict_proba() 를 쓰면 됩니다:')
print('  ', np.round(xgb_wrapper.predict_proba(X_test)[:3, 1], 4))


# %% [Block 11] ④ 피처 중요도 시각화 — 강의 자료 42·49페이지 — plot_importance
# ─────────────────────────────────────────────────────────────
# 강의 자료 42·49페이지 — plot_importance
# (여기서는 그래프 대신 표로 출력합니다. 코드는 아래 주석 참고)
# ─────────────────────────────────────────────────────────────
# from xgboost import plot_importance
# import matplotlib.pyplot as plt
# %matplotlib inline
# fig, ax = plt.subplots(figsize=(10, 12))
# plot_importance(xgb_model, ax=ax)        # 파이썬 래퍼 모델
# plot_importance(xgb_wrapper, ax=ax)      # 사이킷런 래퍼 모델
# 그래프 대신 숫자로 확인 — 파이썬 래퍼는 'f0, f1, ...' 이름을 씁니다.
score = xgb_model.get_score(importance_type='weight')
#   importance_type='weight' → 그 피처가 분할에 쓰인 '횟수' (강의 그래프의 F score)
top = sorted(score.items(), key=lambda kv: -kv[1])[:10]
#   sorted(..., key=lambda kv: -kv[1]) → 값 기준 내림차순 정렬
print('[파이썬 래퍼] 분할에 많이 쓰인 피처 상위 10개 (F score)')
for name, val in top:
    idx = int(name[1:])                     # 'f13' → 13
    print('  %-4s (%-24s) : %.0f' % (name, dataset.feature_names[idx], val))
print()
print('[사이킷런 래퍼] feature_importances_ 상위 10개 (gain 기준 비율)')
imp = sorted(zip(dataset.feature_names, xgb_wrapper.feature_importances_),
             key=lambda kv: -kv[1])[:10]
for name, val in imp:
    bar = '█' * int(val * 60)               # 간이 막대그래프
    print('  %-24s %.4f %s' % (name, val, bar))


# %% [Block 12] ⑤ 교차 검증 — xgb.cv — 강의 자료 43페이지 — xgb.cv (파이썬 래퍼에서만 제공)
# ─────────────────────────────────────────────────────────────
# 강의 자료 43페이지 — xgb.cv (파이썬 래퍼에서만 제공)
# ─────────────────────────────────────────────────────────────
cv_result = xgb.cv(params, dtrain, num_boost_round=10, nfold=3,
                   stratified=False, folds=None, metrics=(), obj=None,
                   maximize=False, early_stopping_rounds=None,
                   as_pandas=True, verbose_eval=None, show_stdv=True,
                   seed=0, callbacks=None, shuffle=True)
#   nfold=3      → 데이터를 3등분해 돌아가며 검증
#   as_pandas    → 결과를 DataFrame 으로 받기
#   show_stdv    → 표준편차도 함께 보여 주기
print(cv_result.round(6).to_string())
print()
print('→ test-logloss-mean 이 계속 줄고 있으므로 10라운드로는 부족합니다.')
print('   num_boost_round 를 늘리면서 이 값이 최소가 되는 지점을 찾으면 됩니다.')
print('   std(표준편차)가 커진다는 것은 폴드마다 성능 편차가 벌어진다는 뜻입니다.')


# %% [Block 13] 6단계 — 조기 종료 — 양날의 검 — 강의 자료 46·48페이지의 코드를 실행하되,
# ─────────────────────────────────────────────────────────────
# 강의 자료 46·48페이지의 코드를 실행하되,
# "테스트 데이터로 조기 종료" vs "별도 검증셋으로 조기 종료" 를 비교합니다.
# ─────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
# ── (A) 강의 자료 방식: 테스트 세트를 eval_set 으로 사용 (권장하지 않음) ──
xgb_a = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3,
                      early_stopping_rounds=100, eval_metric="logloss")
xgb_a.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print('[A] 테스트 세트로 조기 종료 (강의 자료 46p 방식)')
print('    멈춘 라운드 : %d' % xgb_a.best_iteration)
print('    최고 점수   : %.5f' % xgb_a.best_score)
print('    테스트 정확도: %.4f' % accuracy_score(y_test, xgb_a.predict(X_test)))
# ── (B) early_stopping_rounds = 10 (강의 자료 48p) ─────────────
xgb_b = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3,
                      early_stopping_rounds=10, eval_metric="logloss")
xgb_b.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print()
print('[B] early_stopping_rounds = 10 (강의 자료 48p 방식)')
print('    멈춘 라운드 : %d' % xgb_b.best_iteration)
get_clf_eval(y_test, xgb_b.predict(X_test))


# %% [Block 14] 6단계 — 조기 종료 — 양날의 검 — 올바른 방법: 훈련 / 검증 / 테스트 3분할
# ─────────────────────────────────────────────────────────────
# 올바른 방법: 훈련 / 검증 / 테스트 3분할
# ─────────────────────────────────────────────────────────────
# 훈련 세트를 다시 쪼개서 검증 세트를 만듭니다 (테스트는 건드리지 않음)
X_tr2, X_val, y_tr2, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=156)
print('훈련 %d개 / 검증 %d개 / 테스트 %d개'
      % (len(X_tr2), len(X_val), len(X_test)))
print()
print('  patience | 멈춘 라운드 | 검증 logloss | 테스트 정확도')
print('  ---------+-------------+--------------+---------------')
for patience in [10, 30, 50, 100]:
    m = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3,
                      early_stopping_rounds=patience, eval_metric='logloss',
                      random_state=0)
    m.fit(X_tr2, y_tr2, eval_set=[(X_val, y_val)], verbose=False)
    print('  %8d | %11d | %12.5f | %13.4f'
          % (patience, m.best_iteration, m.best_score,
             accuracy_score(y_test, m.predict(X_test))))
print()
print('→ 이 데이터에서는 44라운드 이후 검증 logloss 가 확실히 나빠져서,')
print('   인내심을 아무리 늘려도 같은 지점에서 멈춥니다. 안정적이라는 뜻입니다.')
print('→ 무엇보다 이 방식이 정직합니다. 테스트 정확도는 "예상 실전 성능"으로 믿을 수 있습니다.')
print('   (A) 방식은 테스트 세트를 이미 훔쳐봤기 때문에 그 숫자를 믿을 수 없습니다.')


# %% [Block 15] 실습 — [실습 1 정답 예시] gamma 가 트리 크기를 어떻게 줄이는가
# [실습 1 정답 예시] gamma 가 트리 크기를 어떻게 줄이는가
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
print('  gamma | 전체 노드 수 | 잎(leaf) 수 | 테스트 정확도')
print('  ------+--------------+-------------+---------------')
for gm in [0.0, 0.5, 2.0, 10.0]:
    p = {'max_depth': 3, 'eta': 0.1, 'objective': 'binary:logistic',
         'eval_metric': 'logloss', 'gamma': gm}
    m = xgb.train(p, dtrain, num_boost_round=50, verbose_eval=False)
    df = m.trees_to_dataframe()               # 트리 구조를 표로 뽑아 줍니다
    n_leaf = (df['Feature'] == 'Leaf').sum()  # Feature 열이 'Leaf' 면 잎 노드
    pr = [1 if x > 0.5 else 0 for x in m.predict(dtest)]
    print('  %5.1f | %12d | %11d | %13.4f'
          % (gm, len(df), n_leaf,
             sum(a == b for a, b in zip(pr, y_test)) / len(y_test)))
print()
print('→ gamma 를 키우면 노드 수가 줄어듭니다. max_depth 를 건드리지 않았는데도요.')
print('   "이득이 비용보다 작은 분할은 아예 하지 않는다"가 실제로 작동한 결과입니다.')


# %% [Block 16] 실습 — [실습 3 정답 예시] 결측치가 있어도 XGBoost 는 학습된다
# [실습 3 정답 예시] 결측치가 있어도 XGBoost 는 학습된다
import numpy as np, warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
rng = np.random.default_rng(0)
X_nan = X_train.copy()
mask = rng.random(X_nan.shape) < 0.10        # 10% 위치를 무작위로 고름
X_nan[mask] = np.nan                          # 그 자리를 결측치로 만듦
X_test_nan = X_test.copy()
X_test_nan[rng.random(X_test_nan.shape) < 0.10] = np.nan
print('훈련 데이터의 결측치 개수 :', int(np.isnan(X_nan).sum()))
# XGBoost — 그냥 학습됩니다
m = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=3,
                  random_state=0).fit(X_nan, y_train)
print('XGBoost 정확도 (결측치 10%%) : %.4f'
      % accuracy_score(y_test, m.predict(X_test_nan)))
# 사이킷런 GBM — 에러가 납니다
try:
    GradientBoostingClassifier(random_state=0).fit(X_nan, y_train)
    print('사이킷런 GBM : 학습 성공')
except Exception as e:
    print('사이킷런 GBM : 에러 발생 →', type(e).__name__)
    print('   ', str(e).split(chr(10))[0][:90])
print()
print('→ XGBoost 는 결측치를 만나면 "왼쪽으로 보낼까 오른쪽으로 보낼까"를')
print('   학습 중에 스스로 정합니다(default direction). 전처리 부담이 크게 줄어듭니다.')
