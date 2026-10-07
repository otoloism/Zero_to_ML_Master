# -*- coding: utf-8 -*-
"""
03-10-01 노벨티 탐지와 아웃라이어 탐지

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-10장 - 이상탐지·교차검증 실습/03-10-01 노벨티 탐지와 아웃라이어 탐지.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] ▸ ① 라이브러리 불러오기 — 수치 계산의 기본 도구 — 배열(array)을 다룹니다.
# 수치 계산의 기본 도구 — 배열(array)을 다룹니다.
import numpy as np
# 표 형태 데이터를 다루는 도구 — 엑셀 시트라고 생각하면 됩니다.
import pandas as pd
# 그림을 그리는 도구입니다.
import matplotlib.pyplot as plt
# matplotlib을 예쁘게 감싼 도구입니다(이 장에서는 보조로만 씁니다).
import seaborn as sns

# ── 오늘 쓸 이상탐지 3형제 ─────────────────────────────────
# ① 타원 울타리 방식 (공분산 기반)
from sklearn.covariance import EllipticEnvelope
# ② 동네 밀도 비교 방식 (최근접 이웃 기반)
from sklearn.neighbors import LocalOutlierFactor
# ③ 스무고개 격리 방식 (트리 앙상블 기반)
from sklearn.ensemble import IsolationForest


# %% [Block 2] ▸ ② 데이터 생성
import numpy as np
import pandas as pd

# ── 난수 생성기를 '고정된 씨앗(42)'으로 만듭니다 ──────────────
# RandomState(42) : 42라는 씨앗을 심으면 매번 똑같은 난수가 나옵니다.
#   → 씨앗을 고정해야 오늘 만든 결과를 내일도 그대로 재현할 수 있습니다.
#   → 42는 관습적으로 쓰는 숫자일 뿐, 특별한 의미는 없습니다.
rng = np.random.RandomState(42)

# ── 훈련용 정상 데이터 ────────────────────────────────────
# rng.randn(1000, 2) : 평균 0, 표준편차 1인 정규분포에서 1000행 2열을 뽑습니다.
#   (randn의 n은 normal(정규분포)의 n입니다. rand는 균등분포라 다릅니다!)
# 0.2를 곱하면 표준편차가 0.2로 줄어듭니다 → 점들이 훨씬 촘촘히 뭉칩니다.
X_train = 0.2 * rng.randn(1000, 2)

# np.r_[A, B] : 두 배열을 '위아래로' 이어 붙입니다(row 방향 concat).
#   X_train + 3  → 모든 점을 (3, 3) 방향으로 통째로 밀어 놓은 덩어리
#   X_train      → 원래 (0, 0) 근처 덩어리
#   결과: 서로 떨어진 두 덩어리, 총 2000개
X_train = np.r_[X_train + 3, X_train]

# 넘파이 배열을 표(DataFrame)로 바꿉니다. 열 이름은 x1, x2.
#   → 이렇게 하면 나중에 X_train.x1 처럼 점(.)으로 열을 꺼낼 수 있습니다.
X_train = pd.DataFrame(X_train, columns=['x1', 'x2'])

# ── 검증용 정상 데이터 ('새로 들어온 손님' 역할) ──────────────
# 훈련 데이터와 똑같은 방식으로 만들되 개수만 200 → 총 400개
X_test = 0.2 * rng.randn(200, 2)
X_test = np.r_[X_test + 3, X_test]
X_test = pd.DataFrame(X_test, columns=['x1', 'x2'])

# ── 이상치 데이터 ────────────────────────────────────────
# rng.uniform(low, high, size) : low 이상 high 미만에서 '균등하게' 뽑습니다.
#   정규분포(randn)는 가운데가 두툼하지만, 균등분포(uniform)는 어디든 똑같이 나옵니다.
#   → -1 ~ 5 범위의 넓은 사각형에 50개를 골고루 흩뿌립니다.
outliers = rng.uniform(low=-1, high=5, size=(50, 2))
outliers = pd.DataFrame(outliers, columns=['x1', 'x2'])

# .shape : (행 개수, 열 개수)를 알려 줍니다.
print("훈련 정상 데이터 :", X_train.shape)
print("검증 정상 데이터 :", X_test.shape)
print("이상치 데이터    :", outliers.shape)
print()
print("훈련 데이터 앞 3줄")
print(X_train.head(3).round(4))
print()
print("이상치 데이터 앞 3줄")
print(outliers.head(3).round(4))


# %% [Block 3] ▸ ③ 눈으로 확인하기 — 산점도
import matplotlib.pyplot as plt

# plt.scatter(x좌표, y좌표, ...) : 점을 흩뿌려 그립니다.
#   c        = 점 안쪽 색      ('white' = 흰색)
#   s        = 점 크기         (20*4 = 80. 그냥 80이라고 써도 같습니다)
#   edgecolor= 점 테두리 색    ('k' = black의 약자)
#   label    = 범례에 표시될 이름
plt.scatter(X_train.x1, X_train.x2,
            c='white', s=20*4, edgecolor='k',
            label='training observations')

plt.scatter(outliers.x1, outliers.x2,
            c='red', s=20*4, edgecolor='k',
            label='new abnormal obs.')

# 범례를 오른쪽 위에 붙입니다.
plt.legend(loc='upper right')
# 그림을 화면에 띄웁니다. (주피터에서는 생략해도 나옵니다)
plt.show()


# %% [Block 4] ▸ 직접 세어 보기 — 잡을 수 없는 이상치는 몇 개인가
import numpy as np

# 이상치 50개 각각에 대해, 두 정상 덩어리의 중심 (0,0) / (3,3) 까지의 거리를 잽니다.
centers = np.array([[0.0, 0.0], [3.0, 3.0]])

# outliers.values : DataFrame을 넘파이 배열로 바꿉니다.
# [:, None, :]  → (50, 1, 2) 모양으로 축을 하나 끼워 넣습니다.
# centers[None] → (1, 2, 2)
# 둘을 빼면 브로드캐스팅으로 (50, 2, 2) : 이상치 50개 × 중심 2개 × 좌표 2개
diff = outliers.values[:, None, :] - centers[None, :, :]

# axis=-1 : 마지막 축(좌표 x1,x2)을 따라 제곱합 → 유클리드 거리
dist = np.sqrt((diff ** 2).sum(axis=-1))     # 모양 (50, 2)

# 두 중심 중 '가까운 쪽' 거리만 남깁니다.
nearest = dist.min(axis=1)                   # 모양 (50,)

# 정상 덩어리의 표준편차는 0.2였습니다.
# 통계에서 '평균 ± 3σ' 안이면 사실상 그 분포에 속한다고 봅니다 → 0.6
buried = (nearest < 0.6).sum()

print("이상치 50개 중 정상 덩어리 반경 0.6 안에 파묻힌 개수 :", buried, "개")
print("→ 이론상 어떤 모델도 잡을 수 없는 이상치 비율 :",
      round(buried / 50 * 100, 1), "%")
print()
print("가장 가까운 중심까지의 거리 (작은 순 5개) :",
      np.round(np.sort(nearest)[:5], 3))


# %% [Block 5] ▸ 직접 세어 보기 — 잡을 수 없는 이상치는 몇 개인가 — 어떤 모델이든 이 틀은 똑같습니다
# ── 어떤 모델이든 이 틀은 똑같습니다 ─────────────────────
clf = EllipticEnvelope(contamination=0.1, random_state=42)   # ← 이 줄만 바뀝니다
# clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
# clf = IsolationForest(contamination=0.1, random_state=42)

clf.fit(X_train)                       # 정상 데이터로만 학습 (노벨티 탐지)

y_pred_train    = clf.predict(X_train)     # 훈련 데이터 판정  → +1 / -1
y_pred_test     = clf.predict(X_test)      # 새 정상 손님 판정 → +1이 나와야 정답
y_pred_outliers = clf.predict(outliers)    # 이상치 판정       → -1이 나와야 정답


# %% [Block 6] ▸ 정확도를 세는 방법 — 강의 자료 13p — y_pred_test 안에는 +1과 -1이 섞여 있습니다.
# y_pred_test 안에는 +1과 -1이 섞여 있습니다.
# 정상 데이터를 넣었으니 +1이 나와야 맞습니다 → +1의 비율이 곧 정확도.
#
#  list(...)      : 넘파이 배열을 파이썬 리스트로 바꿉니다.
#                   (넘파이 배열에는 .count() 메서드가 없기 때문입니다)
#  .count(1)      : 리스트 안에 1이 몇 개인지 셉니다.
#  .shape[0]      : 전체 개수(행 수)
print("테스트 데이터셋에서 정확도:",
      list(y_pred_test).count(1) / y_pred_test.shape[0])

# 이상치를 넣었으니 -1이 나와야 맞습니다 → -1의 비율이 곧 정확도.
print("이상치 데이터셋에서 정확도:",
      list(y_pred_outliers).count(-1) / y_pred_outliers.shape[0])


# %% [Block 7] ▸ 실험 — contamination을 바꾸면 두 정확도가 어떻게 움직이나
from sklearn.ensemble import IsolationForest

print(f"{'contamination':>14} | {'정상 통과율':>10} | {'이상치 검출율':>12} | 한 줄 해석")
print("-" * 74)

# 0.01(1%)부터 0.30(30%)까지 의심의 강도를 바꿔 가며 실험합니다.
for nu in [0.01, 0.05, 0.10, 0.20, 0.30]:
    # 같은 모델, 같은 데이터, contamination만 다릅니다.
    clf = IsolationForest(contamination=nu, random_state=42)
    clf.fit(X_train)

    # 정상 데이터를 정상(+1)으로 본 비율
    normal_ok = (clf.predict(X_test) == 1).mean()
    # 이상치를 이상(-1)으로 본 비율
    outlier_ok = (clf.predict(outliers) == -1).mean()

    if nu <= 0.05:
        comment = "너무 관대함 — 이상치를 놓침"
    elif nu <= 0.10:
        comment = "균형점"
    else:
        comment = "너무 깐깐함 — 멀쩡한 데이터를 자꾸 막음"

    print(f"{nu:>14.2f} | {normal_ok:>10.4f} | {outlier_ok:>12.4f} | {comment}")


# %% [Block 8] ▸ 실험 — contamination을 바꾸면 두 정확도가 어떻게 움직이나
from sklearn.ensemble import IsolationForest

clf = IsolationForest(contamination=0.1, random_state=42).fit(X_train)

# 훈련 데이터는 100% 정상인데, 모델은 몇 개를 이상이라 했을까?
pred_train = clf.predict(X_train)
n_wrong = (pred_train == -1).sum()

print("훈련 데이터 개수          :", len(pred_train))
print("모델이 이상이라 판정한 개수:", n_wrong)
print("→ 훈련 정상 통과율        :", round((pred_train == 1).mean(), 4))
print()
print("우리가 준 contamination   : 0.1  (= 10%)")
print("실제로 잘려 나간 비율     :", round(n_wrong / len(pred_train), 4))
print()
print("두 숫자가 거의 같습니다 → contamination이 곧 '자를 비율'이라는 증거입니다.")


# %% [Block 9] ▸ 실험 — contamination을 바꾸면 두 정확도가 어떻게 움직이나
import numpy as np
from sklearn.ensemble import IsolationForest

clf = IsolationForest(contamination=0.1, random_state=42).fit(X_train)

# decision_function : 0보다 크면 정상, 작으면 이상 (5단계 수식 참고)
scores = clf.decision_function(outliers)

print("이상치 50개의 정상다움 점수 (작은 순 5개) :", np.round(np.sort(scores)[:5], 4))
print("이상치 50개의 정상다움 점수 (큰   순 5개) :", np.round(np.sort(scores)[-5:], 4))
print()
print("점수가 0보다 커서 '정상'으로 오판된 이상치 :", (scores > 0).sum(), "개")
print("→ 이상치 검출율 :", round((scores <= 0).mean(), 4))
