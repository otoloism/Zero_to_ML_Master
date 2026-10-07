# -*- coding: utf-8 -*-
"""
교차 검증:1

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-10장 - 이상탐지·교차검증 실습/03-10-05 K-Fold와 Stratified K-Fold.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] ▸ 학습 목표
import sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ── 붓꽃(iris) 데이터 ─────────────────────────────────────
#   150송이의 붓꽃, 특성 4개(꽃받침 길이·너비, 꽃잎 길이·너비)
#   품종 3종(setosa, versicolor, virginica) 각 50송이
iris = load_iris()

# train_test_split : 데이터를 훈련용과 시험용으로 나눕니다.
#   test_size=0.3   → 30%를 시험용으로 (150 × 0.3 = 45송이)
#   random_state=1  → 나누는 방식을 고정 (매번 같은 분할)
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=1)

# DecisionTreeClassifier : 의사결정 트리 분류기
dt_clf = DecisionTreeClassifier(random_state=1)
dt_clf.fit(X_train, y_train)        # 학습
pred = dt_clf.predict(X_test)       # 예측

# accuracy_score(정답, 예측) : 맞힌 비율
# round(값, 4) : 소수점 넷째 자리까지 반올림
print("정확도 :", round(accuracy_score(y_test, pred), 4))


# %% [Block 2] ▸ 학습 목표
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# random_state만 바꿔 가며 같은 실험을 20번 반복합니다.
#   모델도, 데이터도, 코드도 전부 똑같습니다.
#   달라지는 건 '어떤 45송이를 시험에 쓰느냐' 하나뿐입니다.
scores = []
for seed in range(20):
    Xtr, Xte, ytr, yte = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=seed)
    m = DecisionTreeClassifier(random_state=1).fit(Xtr, ytr)
    scores.append(round(accuracy_score(yte, m.predict(Xte)), 4))

print("20번의 정확도 :")
print(scores)
print()
print("최솟값 :", min(scores))
print("최댓값 :", max(scores))
print("평균   :", round(float(np.mean(scores)), 4))
print("표준편차:", round(float(np.std(scores)), 4))
print()
print("★ 같은 모델·같은 데이터인데 정확도가", min(scores), "~", max(scores), "사이를 오갑니다.")
print("  운 좋은 분할 하나를 골라 보고하면 성능을 부풀리는 셈입니다.")


# %% [Block 3] ▸ 직관 — 왜 표준편차까지 봐야 하나
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# KFold(n_splits=5) : 데이터를 5조각으로 나누는 '분할 계획표'를 만듭니다.
#   ★ 아직 데이터를 나눈 게 아닙니다. 나누는 방법만 정한 것입니다.
kfold = KFold(n_splits=5)

iris = load_iris()
feature = iris.data     # 특성 4개 × 150송이
label = iris.target     # 정답 품종 (0, 1, 2)

cv_accuracy = []        # 폴드별 정확도를 담을 빈 리스트
n_iter = 0              # 지금 몇 번째 폴드인지 세는 변수
dt_clf = DecisionTreeClassifier(random_state=1)

print("feature.shape :", feature.shape)
print("label 종류    :", set(label.tolist()))


# %% [Block 4] ▸ 직관 — 왜 표준편차까지 봐야 하나
import numpy as np
from sklearn.metrics import accuracy_score

n_iter = 0
cv_accuracy = []

# kfold.split(feature) : 5번 반복하며 (훈련 인덱스, 검증 인덱스) 쌍을 내놓습니다.
#   ★ 데이터 자체가 아니라 '몇 번째 줄을 쓸지'라는 번호표를 줍니다.
for train_index, test_index in kfold.split(feature):

    # 번호표로 실제 데이터를 꺼냅니다.
    #   feature[train_index] : train_index에 적힌 줄들만 뽑기 (팬시 인덱싱)
    x_train, x_test = feature[train_index], feature[test_index]
    y_train, y_test = label[train_index], label[test_index]

    dt_clf.fit(x_train, y_train)       # 학습
    pred = dt_clf.predict(x_test)      # 예측
    n_iter += 1                        # 폴드 번호 +1

    # np.round(값, 4) : 소수 넷째 자리까지
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = x_train.shape[0]      # 훈련에 쓴 개수
    test_size = x_test.shape[0]        # 검증에 쓴 개수

    # '{0}'.format(...) : 중괄호 자리에 값을 끼워 넣는 옛 방식 문자열 포맷
    #   \n 은 줄바꿈입니다.
    print('\n#{0} 교차 검증 정확도 : {1}, 학습 데이터 크기 : {2}, 검증 데이터 크기 : {3}'
          .format(n_iter, accuracy, train_size, test_size))
    print('#{0} 검증 세트 인덱스 : {1}'.format(n_iter, test_index[:12]), "...")

    cv_accuracy.append(accuracy)

print()
print('## 평균 검증 정확도:', np.mean(cv_accuracy))


# %% [Block 5] ▸ 표준편차까지 함께 보기
import numpy as np

print("폴드별 정확도 :", [float(v) for v in cv_accuracy])
print()
print("평균     :", round(float(np.mean(cv_accuracy)), 4))
print("표준편차 :", round(float(np.std(cv_accuracy)), 4))
print("최솟값   :", min(cv_accuracy))
print("최댓값   :", max(cv_accuracy))
print()
print("보고 형식 : 정확도 {:.4f} ± {:.4f}".format(
    float(np.mean(cv_accuracy)), float(np.std(cv_accuracy))))
print()
print("★ 표준편차가 0.098이나 됩니다. 매우 불안정하다는 신호입니다.")
print("  마지막 폴드가 0.7333으로 혼자 뚝 떨어졌기 때문입니다.")


# %% [Block 6] ▸ 표준편차까지 함께 보기
import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

kfold = KFold(n_splits=3)     # ← 5를 3으로 바꿨을 뿐입니다
n_iter = 0
dt_clf = DecisionTreeClassifier(random_state=1)
cv_accuracy_3 = []

for train_index, test_index in kfold.split(feature):
    x_train, x_test = feature[train_index], feature[test_index]
    y_train, y_test = label[train_index], label[test_index]

    dt_clf.fit(x_train, y_train)
    pred = dt_clf.predict(x_test)
    n_iter += 1

    accuracy = np.round(accuracy_score(y_test, pred), 4)
    print('#{0} 교차 검증 정확도 : {1}, 학습 데이터 크기 : {2}, 검증 데이터 크기 : {3}'
          .format(n_iter, accuracy, x_train.shape[0], x_test.shape[0]))
    cv_accuracy_3.append(accuracy)

print()
print('## 평균 검증 정확도:', np.mean(cv_accuracy_3))


# %% [Block 7] ▸ 범인 찾기 — 레이블 분포를 들여다보기
import pandas as pd
from sklearn.model_selection import KFold

# 붓꽃 데이터를 DataFrame으로 만들어 레이블 분포를 보기 쉽게 합니다.
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label'] = iris.target

print("전체 레이블 분포")
print(iris_df['label'].value_counts().sort_index())
print()
print("→ 품종 0, 1, 2가 각각 50송이씩. 완벽히 균형 잡힌 데이터입니다.")
print()

kfold_1 = KFold(n_splits=3)
n_iter_ = 0

for train_index, test_index in kfold_1.split(iris_df):
    n_iter_ += 1
    # .iloc[번호목록] : DataFrame에서 번호로 행을 꺼냅니다.
    train = iris_df['label'].iloc[train_index]
    test = iris_df['label'].iloc[test_index]

    print('# 교차 검증:{}'.format(n_iter_))
    # value_counts() : 각 값이 몇 번 나오는지 셉니다.
    print('  학습 레이블 데이터 :',
          {int(k): int(v) for k, v in train.value_counts().sort_index().items()})
    print('  검증 레이블 데이터 :',
          {int(k): int(v) for k, v in test.value_counts().sort_index().items()})
    print()


# %% [Block 8] ▸ 왜 5-폴드에서는 안 터졌을까
import numpy as np
from sklearn.model_selection import KFold

print("5-폴드일 때 각 폴드의 검증 세트 레이블 구성")
kf5 = KFold(n_splits=5)
for i, (tr, te) in enumerate(kf5.split(feature), 1):
    # np.bincount : 0,1,2가 각각 몇 개인지 셉니다.
    #   minlength=3 → 없는 값도 0으로 표시
    cnt = np.bincount(label[te], minlength=3)
    print(f"  #{i} 검증 세트 → 품종0:{cnt[0]:>2}  품종1:{cnt[1]:>2}  품종2:{cnt[2]:>2}")

print()
print("★ 3번째 폴드(#3)를 보세요. 품종 1이 20송이, 품종 2가 10송이로 섞였습니다.")
print("  그래서 완전한 0.0은 면했지만, 여전히 분포가 심하게 왜곡되어 있습니다.")
print("  이것이 폴드별 점수가 1.0 ~ 0.7333 으로 널뛴 이유입니다.")


# %% [Block 9] ▸ 시각화 ② — 계층적 분할이 하는 일
import pandas as pd
from sklearn.model_selection import StratifiedKFold

skfold_1 = StratifiedKFold(n_splits=3)
n_iter_ = 0

# ★★★ 결정적 차이 ★★★
#   KFold        : split(데이터)          ← 데이터만 넘김
#   StratifiedKFold : split(데이터, 레이블) ← 레이블도 반드시 넘겨야 함!
#   레이블을 봐야 '비율을 맞춰서' 나눌 수 있기 때문입니다.
for train_index, test_index in skfold_1.split(iris_df, iris_df['label']):
    n_iter_ += 1
    train = iris_df['label'].iloc[train_index]
    test = iris_df['label'].iloc[test_index]

    print('# 교차 검증:{}'.format(n_iter_))
    print('  학습 레이블 데이터 :',
          {int(k): int(v) for k, v in train.value_counts().sort_index().items()})
    print('  검증 레이블 데이터 :',
          {int(k): int(v) for k, v in test.value_counts().sort_index().items()})
    print()


# %% [Block 10] ▸ 성능 재측정 — 강의 자료 31p
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

S_kfold = StratifiedKFold(n_splits=5)

cv_accuracy = []
n_iter = 0
dt_clf = DecisionTreeClassifier(random_state=1)

# ★ split(feature, label) — 두 번째 인자가 핵심입니다
for train_index, test_index in S_kfold.split(feature, label):
    x_train, x_test = feature[train_index], feature[test_index]
    y_train, y_test = label[train_index], label[test_index]

    dt_clf.fit(x_train, y_train)
    pred = dt_clf.predict(x_test)
    n_iter += 1

    accuracy = np.round(accuracy_score(y_test, pred), 4)
    print('#{0} 교차 검증 정확도 : {1}, 학습 데이터 크기 : {2}, 검증 데이터 크기 : {3}'
          .format(n_iter, accuracy, x_train.shape[0], x_test.shape[0]))
    cv_accuracy.append(accuracy)

print()
print('## 평균 검증 정확도:', np.mean(cv_accuracy))
print('## 표준편차       :', round(float(np.std(cv_accuracy)), 4))


# %% [Block 11] ▸ KFold vs StratifiedKFold 최종 비교
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def run_cv(splitter, use_label):
    """분할기를 받아 교차검증을 돌리고 점수 리스트를 돌려줍니다."""
    scores = []
    m = DecisionTreeClassifier(random_state=1)
    # use_label이 True면 split에 label도 넘깁니다.
    gen = (splitter.split(feature, label) if use_label
           else splitter.split(feature))
    for tr, te in gen:
        m.fit(feature[tr], label[tr])
        scores.append(round(float(accuracy_score(label[te], m.predict(feature[te]))), 4))
    return scores

print(f"{'분할 방식':>26} | {'폴드별 점수':>38} | {'평균':>8} | {'표준편차':>8}")
print("-" * 96)
for name, sp, ul in [
        ("KFold(3)", KFold(n_splits=3), False),
        ("StratifiedKFold(3)", StratifiedKFold(n_splits=3), True),
        ("KFold(5)", KFold(n_splits=5), False),
        ("StratifiedKFold(5)", StratifiedKFold(n_splits=5), True)]:
    s = run_cv(sp, ul)
    print(f"{name:>26} | {str(s):>38} | {np.mean(s):>8.4f} | {np.std(s):>8.4f}")

print()
print("★ 3-폴드 : 0.0000 → 0.9733   (완전한 실패에서 최고 점수로)")
print("★ 5-폴드 : 0.9133 → 0.9667   (평균 상승 + 표준편차 0.0980 → 0.0365로 3분의 1)")
print()
print("표준편차가 줄었다는 게 특히 중요합니다.")
print("측정이 안정되었다는 뜻이고, 그래야 모델 비교를 믿을 수 있습니다.")


# %% [Block 12] ▸ KFold vs StratifiedKFold 최종 비교
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time

print(f"{'K':>4} | {'훈련 데이터':>10} | {'검증 데이터':>10} | {'평균':>8} | "
      f"{'표준편차':>9} | {'걸린 시간':>10}")
print("-" * 74)

for K in [2, 3, 5, 10, 20]:
    t0 = time.time()
    sk = StratifiedKFold(n_splits=K)
    scores = []
    tr_size = te_size = 0
    for tr, te in sk.split(feature, label):
        m = DecisionTreeClassifier(random_state=1).fit(feature[tr], label[tr])
        scores.append(accuracy_score(label[te], m.predict(feature[te])))
        tr_size, te_size = len(tr), len(te)
    dt = time.time() - t0
    print(f"{K:>4} | {tr_size:>10} | {te_size:>10} | {np.mean(scores):>8.4f} | "
          f"{np.std(scores):>9.4f} | {dt:>9.3f}s")


# %% [Block 13] ▸ 프레임워크 비교
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

# 극단적 불균형 : 1000개 중 이상이 5개 (0.5%)
y_imb = np.array([0] * 995 + [1] * 5)
X_imb = np.arange(1000).reshape(-1, 1)

print("KFold(shuffle=True) — 각 폴드 검증 세트의 이상 개수")
for seed in range(5):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    counts = [int((y_imb[te] == 1).sum()) for _, te in kf.split(X_imb)]
    zero = counts.count(0)
    print(f"  seed={seed} → {counts}   이상이 0개인 폴드 : {zero}개")

print()
print("StratifiedKFold — 각 폴드 검증 세트의 이상 개수")
sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
counts = [int((y_imb[te] == 1).sum()) for _, te in sk.split(X_imb, y_imb)]
print(f"  → {counts}   이상이 0개인 폴드 : {counts.count(0)}개")
print()
print("★ 섞기만 하면 '이상이 하나도 없는 폴드'가 생깁니다.")
print("  그 폴드에서는 재현율을 계산할 수조차 없습니다.")
print("  StratifiedKFold는 5개를 1개씩 정확히 배분합니다.")


# %% [Block 14] ▸ 프레임워크 비교
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ── 실습 3 + 4 정답 : 최적 max_depth 찾기 + 과대적합 진단 ──────
print(f"{'max_depth':>10} | {'훈련 정확도':>11} | {'검증 정확도':>11} | "
      f"{'표준편차':>9} | {'격차':>7}")
print("-" * 62)

best_depth, best_score = None, -1
for depth in range(1, 11):
    sk = StratifiedKFold(n_splits=5)
    tr_scores, te_scores = [], []
    for tr, te in sk.split(feature, label):
        m = DecisionTreeClassifier(max_depth=depth,
                                   random_state=1).fit(feature[tr], label[tr])
        tr_scores.append(accuracy_score(label[tr], m.predict(feature[tr])))
        te_scores.append(accuracy_score(label[te], m.predict(feature[te])))

    tr_m, te_m = np.mean(tr_scores), np.mean(te_scores)
    gap = tr_m - te_m
    if te_m > best_score:
        best_score, best_depth = te_m, depth

    print(f"{depth:>10} | {tr_m:>11.4f} | {te_m:>11.4f} | "
          f"{np.std(te_scores):>9.4f} | {gap:>7.4f}")

print()
print(f"★ 최적 max_depth = {best_depth} (검증 정확도 {best_score:.4f})")
print()
print("깊이를 계속 키워도 검증 정확도는 더 오르지 않는데")
print("훈련 정확도만 1.0을 향해 올라갑니다 → 전형적인 과대적합 패턴입니다.")
