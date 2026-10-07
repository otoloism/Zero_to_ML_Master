# -*- coding: utf-8 -*-
"""
03-10-06 cross_val_score부터 LOOCV까지

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-10장 - 이상탐지·교차검증 실습/03-10-06 cross_val_score부터 LOOCV까지.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] ▸ 학습 목표
import sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ★ 오늘의 주인공
from sklearn.model_selection import cross_val_score

iris = load_iris()
feature = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=1)

# ── 이 한 줄이 전부입니다 ─────────────────────────────────
#   cross_val_score(모델, 특성, 정답)
#   → 폴드별 점수가 담긴 넘파이 배열을 돌려줍니다.
basic_scores = cross_val_score(dt_clf, feature, label)

print('\n##cross_val_score를 통한 성능 평가 : ', basic_scores)


# %% [Block 2] ▸ 학습 목표
print('\n## 평균 검증 정확도:', round(float(np.mean(basic_scores)), 4))


# %% [Block 3] ▸ 이 한 줄이 내부에서 하는 일
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

fresh = DecisionTreeClassifier(random_state=1)
scores = cross_val_score(fresh, feature, label)   # 교차검증 실행

print("교차검증 점수 :", np.round(scores, 4))
print()
try:
    fresh.predict(feature[:3])
except NotFittedError as e:
    print("교차검증 후 바로 predict를 부르면 :")
    print("  ", type(e).__name__)
    print("  ", str(e)[:90], "...")
print()
print("→ 최종 모델을 쓰려면 전체 데이터로 다시 학습해야 합니다.")
fresh.fit(feature, label)
print("   fit 후 예측 :", fresh.predict(feature[:3]))


# %% [Block 4] ▸ 이 한 줄이 내부에서 하는 일
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier(random_state=1)

a = cross_val_score(dt_clf, feature, label)                          # 기본값
b = cross_val_score(dt_clf, feature, label, cv=StratifiedKFold(5))   # 계층적
c = cross_val_score(dt_clf, feature, label, cv=KFold(5))             # 그냥 KFold

print("cross_val_score 기본값      :", np.round(a, 4))
print("StratifiedKFold(5) 명시     :", np.round(b, 4))
print("KFold(5) 명시               :", np.round(c, 4))
print()
print("기본값 == StratifiedKFold ? :", np.allclose(a, b))
print("기본값 == KFold          ? :", np.allclose(a, c))
print()
print("★ 결론 : cross_val_score는 분류기를 주면 StratifiedKFold를 몰래 씁니다.")


# %% [Block 5] ▸ 이 한 줄이 내부에서 하는 일
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes

# 회귀 데이터로 확인해 봅시다.
d = load_diabetes()

x = cross_val_score(LinearRegression(), d.data, d.target)              # 기본값
y_ = cross_val_score(LinearRegression(), d.data, d.target, cv=KFold(5))  # KFold 명시

print("회귀 - 기본값   :", np.round(x, 4))
print("회귀 - KFold(5) :", np.round(y_, 4))
print("두 결과가 같은가 :", np.allclose(x, y_))
print()
print("★ 회귀에서는 KFold가 자동 선택됩니다. (계층화가 불가능하니까요)")
print()
print("참고 : 회귀의 기본 점수는 정확도가 아니라 R² (결정계수)입니다.")
print("       모델마다 score() 메서드가 다른 지표를 쓰기 때문입니다.")


# %% [Block 6] ▸ 직접 분할기를 넘겨 앞 절 결과 재현하기
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier(random_state=1)

print(f"{'cv 인자':>34} | {'평균':>8} | {'표준편차':>9}")
print("-" * 58)
for name, cv in [
        ("(생략)  ← 자동 선택", None),
        ("5  ← 정수만 (역시 자동 선택)", 5),
        ("KFold(n_splits=3)", KFold(3)),
        ("KFold(3, shuffle=True, random_state=1)", KFold(3, shuffle=True, random_state=1)),
        ("StratifiedKFold(n_splits=3)", StratifiedKFold(3)),
        ("StratifiedKFold(n_splits=10)", StratifiedKFold(10))]:
    s = (cross_val_score(dt_clf, feature, label) if cv is None
         else cross_val_score(dt_clf, feature, label, cv=cv))
    print(f"{name:>34} | {np.mean(s):>8.4f} | {np.std(s):>9.4f}")

print()
print("★ KFold(3)만 0.0입니다. 나머지는 모두 정상입니다.")
print("  shuffle=True 로도 살아나지만, 계층화가 더 안전합니다.")


# %% [Block 7] ▸ 직접 분할기를 넘겨 앞 절 결과 재현하기
import pandas as pd
from sklearn.model_selection import cross_validate

c_val = cross_validate(dt_clf, feature, label)

# 반환값은 딕셔너리입니다. DataFrame으로 감싸면 표로 예쁘게 보입니다.
#   index=[...] : 각 행에 이름 붙이기
df = pd.DataFrame(c_val, index=['set1', 'set2', 'set3', 'set4', 'set5'])
print(df.round(6))


# %% [Block 8] ▸ 직접 분할기를 넘겨 앞 절 결과 재현하기 — .describe() : 평균·표준편차·최솟값 등 요약 통계를 한 번에 계산합니다.
# .describe() : 평균·표준편차·최솟값 등 요약 통계를 한 번에 계산합니다.
# .loc[['mean'], :] : 그중 'mean' 행만, 모든 열을 꺼냅니다.
print(df.describe().loc[['mean'], :].round(6))


# %% [Block 9] ▸ 여러 지표를 한 번에 재기
import pandas as pd
from sklearn.model_selection import cross_validate

# scoring에 리스트를 주면 여러 지표를 한 번에 잽니다.
#   accuracy  : 정확도
#   f1_macro  : 클래스별 F1을 단순 평균 (🔗 03-10-07절)
#   precision_macro / recall_macro : 정밀도·재현율의 클래스 평균
# return_train_score=True : 훈련 점수도 같이 돌려줍니다 (과대적합 진단용)
res = cross_validate(
    dt_clf, feature, label,
    scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
    return_train_score=True)

out = pd.DataFrame(res)
print("사용 가능한 열 :")
for k in out.columns:
    print("  -", k)
print()
print("각 지표의 평균")
print(out.mean().round(4).to_string())
print()
gap = out['train_accuracy'].mean() - out['test_accuracy'].mean()
print("훈련 정확도 - 검증 정확도 =", round(float(gap), 4))
print("→ 이 격차가 크면 과대적합입니다. 🔗 03-08-03 학습 곡선")


# %% [Block 10] ▸ 시각화 ① — 강의 자료 39p
import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
feature = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=1)

# LeaveOneOut() : 인자가 아예 없습니다. K가 데이터 개수로 자동 결정되니까요.
loocv = LeaveOneOut()
score = cross_val_score(dt_clf, feature, label, cv=loocv)

print("\n## 교차 검증 횟수 : ", len(score))
print("\n## 평균 검증 정확도:", round(float(np.mean(score)), 4))


# %% [Block 11] 평균 검증 정확도: 0.94⚠️ 개별 점수를 보면 충격적입니다150개의 점수가 나왔는데, 그 값들이 0.0 아니면 1.0뿐입니다.
import numpy as np

print("점수의 종류 :", sorted(set(score.tolist())))
print("맞힌 개수   :", int(score.sum()), "/ 150")
print("평균        :", round(float(np.mean(score)), 4))
print("표준편차    :", round(float(np.std(score)), 4), " ← 매우 큽니다!")
print()
print("★ 개별 점수의 표준편차가 0.2375로 어마어마합니다.")
print("  하지만 이건 '모델이 불안정하다'는 뜻이 아닙니다.")
print("  검증 데이터가 1개뿐이라 점수가 0/1로만 나오기 때문입니다.")
print()
print("→ LOOCV에서는 개별 점수의 표준편차를 '신뢰도'로 읽으면 안 됩니다.")
print("  평균(=전체 맞힌 비율)만 의미가 있습니다.")


# %% [Block 12] ▸ 시각화 ② — 강의 자료 41p
import numpy as np
from sklearn.model_selection import ShuffleSplit, cross_val_score

# ShuffleSplit(test_size, train_size, n_splits)
#   test_size = .6   → 60%를 검증으로
#   train_size = .4  → 40%를 훈련으로
#   n_splits = 20    → 이 뽑기를 20번 반복
#   ★ test_size + train_size 가 1이 아니어도 됩니다! (남는 건 안 씀)
shuffle_split = ShuffleSplit(test_size=.6, train_size=.4, n_splits=20,
                             random_state=0)
score = cross_val_score(dt_clf, feature, label, cv=shuffle_split)

print("\n## 교차 검증 횟수 : ", len(score))
print("\n## 평균 검증 정확도:", round(float(np.mean(score)), 4))
print("## 표준편차       :", round(float(np.std(score)), 4))


# %% [Block 13] 표준편차 : 0.0232ShuffleSplit vs KFold — 구조적 차이
import numpy as np
from sklearn.model_selection import ShuffleSplit, KFold

# ── K-폴드와 무엇이 다른지 눈으로 확인 ──────────────────────
ss = ShuffleSplit(n_splits=5, test_size=.3, random_state=0)
test_sets = [set(te.tolist()) for _, te in ss.split(feature)]

print("【ShuffleSplit】")
print("  1번 분할과 2번 분할의 검증 세트가 겹치는 개수 :",
      len(test_sets[0] & test_sets[1]), "개")
covered = set().union(*test_sets)
print("  5번 분할로 한 번이라도 검증에 쓰인 데이터 :",
      len(covered), "/ 150 개")
print("  → 한 번도 검증에 안 쓰인 데이터가", 150 - len(covered), "개 있습니다.")
print()

kf = KFold(n_splits=5)
kf_sets = [set(te.tolist()) for _, te in kf.split(feature)]
print("【KFold】")
print("  1번 폴드와 2번 폴드의 검증 세트가 겹치는 개수 :",
      len(kf_sets[0] & kf_sets[1]), "개")
print("  5번 폴드로 한 번이라도 검증에 쓰인 데이터 :",
      len(set().union(*kf_sets)), "/ 150 개")
print("  → 모든 데이터가 정확히 한 번씩 쓰입니다.")


# %% [Block 14] ▸ 계층적 버전도 있습니다
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

# StratifiedShuffleSplit : ShuffleSplit + 클래스 비율 보존
#   무작위로 뽑되, 각 클래스의 비율은 전체와 같게 유지합니다.
sss = StratifiedShuffleSplit(n_splits=20, test_size=.3, random_state=0)
s = cross_val_score(dt_clf, feature, label, cv=sss)

ss_plain = ShuffleSplit(n_splits=20, test_size=.3, random_state=0)
s2 = cross_val_score(dt_clf, feature, label, cv=ss_plain)

print("ShuffleSplit           : 평균", round(float(np.mean(s2)), 4),
      " 표준편차", round(float(np.std(s2)), 4))
print("StratifiedShuffleSplit : 평균", round(float(np.mean(s)), 4),
      " 표준편차", round(float(np.std(s)), 4))
print()
print("★ 분류 문제라면 계층적 버전을 쓰는 것이 안전합니다.")
print("  🔗 03-02절 '계층적 샘플링'과 같은 아이디어입니다.")


# %% [Block 15] ▸ 계층적 버전도 있습니다
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

# RepeatedStratifiedKFold()
#   n_splits=5   : 5-폴드 (기본값)
#   n_repeats=10 : 그 5-폴드를 10번 반복 (기본값)
#   → 총 5 × 10 = 50개의 점수
#   ★ 매 반복마다 폴드를 '다시 섞어' 나눕니다.
rskfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
score = cross_val_score(dt_clf, feature, label, cv=rskfold)

print("\n## 교차 검증 횟수 : ", len(score))
print("\n## 평균 검증 정확도:", round(float(np.mean(score)), 4))
print("## 표준편차       :", round(float(np.std(score)), 4))
print("## 최솟값 ~ 최댓값 :", round(float(score.min()), 4),
      "~", round(float(score.max()), 4))


# %% [Block 16] 최솟값 ~ 최댓값 : 0.8333 ~ 1.0📌 강의 자료와 값이 조금 다른 이유강의 43p는 0.9467, 우리 결과는 0.9413입니다.
import numpy as np
from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold,
                                     cross_val_score)

# ── 반복이 왜 필요한지 실험으로 ──────────────────────────────
# ① 5-폴드를 한 번만 돌리되, 나누는 씨앗을 바꿔 가며 10번
print("【5-폴드 한 번씩, 씨앗만 바꿔 10회】")
single = []
for seed in range(10):
    sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    single.append(round(float(np.mean(
        cross_val_score(dt_clf, feature, label, cv=sk))), 4))
print("  각 회차의 평균 :", single)
print("  회차 간 표준편차 :", round(float(np.std(single)), 4))
print("  → 같은 5-폴드인데도 '어떻게 나눴느냐'에 따라 결과가 흔들립니다.")
print()

# ② RepeatedStratifiedKFold로 50개를 한 번에
rs = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
rep = cross_val_score(dt_clf, feature, label, cv=rs)
print("【RepeatedStratifiedKFold(5, 10)】")
print("  점수 개수 :", len(rep))
print("  전체 평균 :", round(float(np.mean(rep)), 4))
print("  평균의 표준오차 :", round(float(np.std(rep) / np.sqrt(len(rep))), 4))
print("  → 50개를 평균 내므로 '분할 운'의 영향이 크게 줄어듭니다.")


# %% [Block 17] ▸ 교차검증 5형제 총정리
import numpy as np
import time
from sklearn.model_selection import (KFold, StratifiedKFold, LeaveOneOut,
    ShuffleSplit, StratifiedShuffleSplit, RepeatedStratifiedKFold,
    cross_val_score)

setups = [
    ("KFold(5)",                     KFold(5)),
    ("StratifiedKFold(5)",           StratifiedKFold(5)),
    ("LeaveOneOut()",                LeaveOneOut()),
    ("ShuffleSplit(20, .3)",         ShuffleSplit(n_splits=20, test_size=.3,
                                                  random_state=0)),
    ("StratifiedShuffleSplit(20,.3)", StratifiedShuffleSplit(n_splits=20,
                                                             test_size=.3,
                                                             random_state=0)),
    ("RepeatedStratifiedKFold(5,10)", RepeatedStratifiedKFold(n_splits=5,
                                                              n_repeats=10,
                                                              random_state=42)),
]

print(f"{'방식':>30} | {'횟수':>5} | {'평균':>8} | {'표준편차':>9} | {'시간':>8}")
print("-" * 74)
for name, cv in setups:
    t0 = time.time()
    s = cross_val_score(dt_clf, feature, label, cv=cv)
    dt = time.time() - t0
    print(f"{name:>30} | {len(s):>5} | {np.mean(s):>8.4f} | "
          f"{np.std(s):>9.4f} | {dt:>7.3f}s")

print()
print("★ KFold(5)만 0.9133으로 낮습니다 — 계층화를 안 했기 때문입니다.")
print("★ LOOCV의 표준편차 0.2375는 '불안정'이 아니라 '점수가 0/1뿐'이라는 뜻입니다.")
print("★ 횟수가 늘수록 시간도 늘어납니다. 공짜는 없습니다.")


# %% [Block 18] ▸ 프레임워크 비교
import time
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

rs = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

# n_jobs=1  : 한 코어로 순서대로
t0 = time.time(); cross_val_score(dt_clf, feature, label, cv=rs, n_jobs=1)
t1 = time.time() - t0

# n_jobs=-1 : 사용 가능한 모든 코어를 씁니다.
#   폴드끼리 서로 독립적이라 병렬 처리에 딱 맞습니다.
t0 = time.time(); cross_val_score(dt_clf, feature, label, cv=rs, n_jobs=-1)
t2 = time.time() - t0

print("n_jobs=1  :", round(t1, 3), "초")
print("n_jobs=-1 :", round(t2, 3), "초")
print()
print("★ 데이터가 작으면 오히려 병렬화 준비 비용이 더 큽니다.")
print("  모델 학습이 무거울 때(딥러닝·부스팅) n_jobs=-1의 효과가 큽니다.")


# %% [Block 19] ▸ 프레임워크 비교
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

# ── 실습 1 : 지표 바꾸기 ──────────────────────────────────
print("【실습 1】 지표별 결과")
for sc in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro',
           'balanced_accuracy']:
    s = cross_val_score(dt_clf, feature, label, cv=StratifiedKFold(5), scoring=sc)
    print(f"  {sc:>20} : {np.mean(s):.4f}")
print()

# ── 실습 2 : 누수 없는 전처리 ─────────────────────────────
print("【실습 2】 Pipeline으로 스케일링 포함")
pipe = make_pipeline(StandardScaler(),
                     DecisionTreeClassifier(random_state=1))
s = cross_val_score(pipe, feature, label, cv=StratifiedKFold(5))
print("  파이프라인 정확도 :", round(float(np.mean(s)), 4))
print("  → 폴드마다 훈련 부분으로만 스케일러가 학습됩니다.")
print()

# ── 실습 4 : 폴드별 특성 중요도 ──────────────────────────
print("【실습 4】 폴드별 특성 중요도")
res = cross_validate(dt_clf, feature, label, cv=StratifiedKFold(5),
                     return_estimator=True)
imps = np.array([m.feature_importances_ for m in res['estimator']])
for i, name in enumerate(iris.feature_names):
    print(f"  {name:>20} : 평균 {imps[:, i].mean():.4f}  "
          f"편차 {imps[:, i].std():.4f}")
print()
print("  → 편차가 작으면 '어떻게 나눠도 같은 특성이 중요하다'는 뜻이라 안정적입니다.")
