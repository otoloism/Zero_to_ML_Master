# -*- coding: utf-8 -*-
"""
그림을 그리려면 아래 코드를 실행하세요

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-10장 - 이상탐지·교차검증 실습/03-10-03 Local Outlier Factor — 우리 동네와 옆 동네의 밀도 비교.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] ▸ 숫자로 증명하기
import numpy as np
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor

rngL = np.random.RandomState(0)

# ── 강의 14p 그림을 숫자로 재현합니다 ──────────────────────
# C1 : 성긴 무리 — 표준편차 1.2로 넓게 퍼져 있음
C1 = rngL.randn(150, 2) * 1.2 + [7, 7]
# C2 : 빽빽한 무리 — 표준편차 0.15로 아주 촘촘함 (C1보다 8배 촘촘)
C2 = rngL.randn(150, 2) * 0.15 + [0, 0]
# o1 : 두 무리 사이의 완전한 외톨이
o1 = np.array([[3.5, 0.0]])
# o2 : 빽빽한 C2 바로 옆에 붙어 있는 점
o2 = np.array([[0.9, 0.9]])

X_lof = np.r_[C1, C2, o1, o2]   # 총 302개

# ── 방법 A : 전역 잣대 (k번째 이웃까지의 거리) ─────────────
#   NearestNeighbors : 가장 가까운 이웃을 찾아 주는 도구입니다.
#   n_neighbors=21 인 이유 : 자기 자신이 0번째 이웃으로 포함되므로 20+1
nn = NearestNeighbors(n_neighbors=21).fit(X_lof)
d, _ = nn.kneighbors(X_lof)   # d[i] = i번 점에서 가까운 순 거리 21개
knn_dist = d[:, -1]           # 마지막 = 20번째 이웃까지의 거리

# ── 방법 B : LOF (국소 밀도 비율) ────────────────────────
lof = LocalOutlierFactor(n_neighbors=20).fit(X_lof)
# negative_outlier_factor_ 는 -LOF 값입니다(사이킷런은 '클수록 정상'으로 통일).
#   앞에 마이너스를 붙여 원래 LOF 점수로 되돌립니다.
lof_score = -lof.negative_outlier_factor_

print("                     전역 kNN 거리 |   LOF 점수")
print("-" * 50)
print(f"C1(성긴 무리) 평균  {knn_dist[:150].mean():>13.3f} | {lof_score[:150].mean():>10.3f}")
print(f"C1 최댓값           {knn_dist[:150].max():>13.3f} | {lof_score[:150].max():>10.3f}")
print(f"C2(빽빽한 무리) 평균{knn_dist[150:300].mean():>13.3f} | {lof_score[150:300].mean():>10.3f}")
print(f"o1(완전 외톨이)     {knn_dist[300]:>13.3f} | {lof_score[300]:>10.3f}")
print(f"o2(빽빽한 곳 옆)    {knn_dist[301]:>13.3f} | {lof_score[301]:>10.3f}")
print()

# '가장 이상해 보이는 10개'를 뽑았을 때 무엇이 뽑히는가?
top_global = np.argsort(knn_dist)[-10:]
top_lof = np.argsort(lof_score)[-10:]
print("전역 거리 상위 10개 중 → C1(정상)이 차지한 자리 :",
      (top_global < 150).sum(), "개")
print("전역 거리 상위 10개에 o2가 들어 있나? :", 301 in top_global)
print()
print("LOF     상위 10개 중 → C1(정상)이 차지한 자리 :",
      (top_lof < 150).sum(), "개")
print("LOF     상위 10개에 o2가 들어 있나? :", 301 in top_lof)


# %% [Block 2] ▸ 전체 흐름 한눈에 보기
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

# ── 예제 데이터 : 정사각형 꼭짓점 4개 + 멀리 떨어진 점 1개 ──────
X_toy = np.array([[0., 0.],    # 0번 점
                  [0., 1.],    # 1번 점
                  [1., 0.],    # 2번 점
                  [1., 1.],    # 3번 점
                  [5., 5.]])   # 4번 점 ← 명백한 이상치
k = 2
n = len(X_toy)

# ── 0) 모든 점 쌍의 거리표 만들기 ─────────────────────────
#   np.linalg.norm(v) : 벡터 v의 길이(유클리드 노름). √(v1² + v2²)
D = np.array([[np.linalg.norm(X_toy[i] - X_toy[j]) for j in range(n)]
              for i in range(n)])

# ── 1층) k-거리와 이웃 집합 ──────────────────────────────
kdist = np.zeros(n)
NB = []
for i in range(n):
    # 자기 자신(거리 0)은 빼고, 가까운 순으로 정렬한 뒤 앞에서 k개만
    order = sorted([(D[i, j], j) for j in range(n) if j != i])[:k]
    kdist[i] = order[-1][0]          # k번째 이웃까지의 거리
    NB.append([j for _, j in order]) # 이웃들의 번호

print("【1층】 k-거리 :", np.round(kdist, 4))
print("        이웃 집합 :", NB)
print()

# ── 2층) 도달 거리 : max(o의 k-거리, 실제 거리) ────────────
def reach(i, j):
    return max(kdist[j], D[i, j])

print("【2층】 도달 거리")
for i in range(n):
    vals = [round(float(reach(i, j)), 4) for j in NB[i]]
    print(f"        점{i} → 이웃{NB[i]} : {vals}")
print()

# ── 3층) lrd = 1 / (도달 거리들의 평균) ───────────────────
lrd = np.array([1 / np.mean([reach(i, j) for j in NB[i]]) for i in range(n)])
print("【3층】 lrd :", np.round(lrd, 4))
print()

# ── 4층) LOF = (이웃들의 lrd 평균) / (내 lrd) ─────────────
lof_manual = np.array([np.mean([lrd[j] for j in NB[i]]) / lrd[i]
                       for i in range(n)])
print("【4층】 LOF (손계산) :", np.round(lof_manual, 4))

# ── 검산 : 사이킷런과 비교 ───────────────────────────────
m = LocalOutlierFactor(n_neighbors=k).fit(X_toy)
lof_sklearn = -m.negative_outlier_factor_
print("        LOF (사이킷런) :", np.round(lof_sklearn, 4))
print()
# np.allclose : 소수점 오차 범위 안에서 두 배열이 같은지 확인
print("★ 완전히 일치하는가? :", np.allclose(lof_manual, lof_sklearn))


# %% [Block 3] ▸ 결과 해석
from sklearn.neighbors import LocalOutlierFactor

X_outliers = outliers.copy()

# ── 모델 만들기 ────────────────────────────────────────────
#  n_neighbors=20   : 이웃 20명을 보고 판단합니다 (k=20)
#  novelty=True     : ★ 가장 중요한 인자!
#                     True  → 노벨티 탐지 모드. predict()를 쓸 수 있습니다.
#                     False → 아웃라이어 탐지 모드(기본값). fit_predict()만 가능.
#  contamination=0.1: 임계값을 하위 10% 지점으로 잡습니다.
clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)

# fit 안에서 벌어지는 일:
#   ① 훈련 데이터끼리 k-거리·도달 거리·lrd를 모두 계산해 저장
#   ② 각 훈련점의 LOF 점수를 구하고, 하위 10% 지점을 임계값으로 저장
clf.fit(X_train.values)

y_pred_train    = clf.predict(X_train.values)
y_pred_test     = clf.predict(X_test.values)
y_pred_outliers = clf.predict(X_outliers.values)

print("검증 데이터 앞 10개 판정 :", y_pred_test[:10])


# %% [Block 4] ▸ 결과 해석
print("테스트 데이터셋에서 정확도:",
      list(y_pred_test).count(1) / y_pred_test.shape[0])
print("이상치 데이터셋에서 정확도:",
      list(y_pred_outliers).count(-1) / y_pred_outliers.shape[0])


# %% [Block 5] ▸ 왜 좋아졌는가 — 그 빈 공간을 다시 물어보기
import numpy as np

# 03-10-02절에서 EllipticEnvelope가 '정상'이라고 우겼던 그 지점
probe = np.array([[1.5, 1.5]])

print("(1.5, 1.5) — 두 덩어리 사이의 텅 빈 공간")
print("  LOF 판정            :", "정상(+1)" if clf.predict(probe)[0] == 1 else "이상(-1)")
print("  decision_function   :", round(clf.decision_function(probe)[0], 4))
print("  (음수 = 이상, 양수 = 정상)")
print()

# score_samples : 원래 LOF 점수에 마이너스를 붙인 값 (클수록 정상)
print("  score_samples       :", round(clf.score_samples(probe)[0], 4))
print("  → 원래 LOF 점수      :", round(-clf.score_samples(probe)[0], 4))
print()
print("LOF 점수가 1보다 훨씬 큽니다 → 이웃들에 비해 나만 유독 헐렁하다는 뜻.")
print("EllipticEnvelope는 '정상'이라 했던 자리를 LOF는 정확히 잡아냅니다.")


# %% [Block 6] ▸ 결과 시각화 — 강의 자료 16p
import matplotlib.pyplot as plt

X_outliers = X_outliers.assign(y=y_pred_outliers)

plt.scatter(X_train.x1, X_train.x2,
            c='white', s=20*4, edgecolor='k',
            label="training observations")

# 이상치로 제대로 잡힌 것 (빨강)
plt.scatter(X_outliers.loc[X_outliers.y == -1, ['x1']],
            X_outliers.loc[X_outliers.y == -1, ['x2']],
            c='red', s=20*4, edgecolor='k',
            label="detected outliers")

# 놓친 것 (초록) — LOF에서는 단 2개만 나옵니다
plt.scatter(X_outliers.loc[X_outliers.y == 1, ['x1']],
            X_outliers.loc[X_outliers.y == 1, ['x2']],
            c='green', s=20*4, edgecolor='k',
            label="detected regular obs")

plt.legend(loc='upper right')
plt.show()


# %% [Block 7] ▸ ① novelty — 하나의 클래스, 두 개의 인격
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

# ── 모드 A : 아웃라이어 탐지 (novelty=False, 기본값) ─────────
# 훈련 데이터에 이상치를 '일부러 섞어서' 넣습니다.
X_mixed = np.r_[X_train.values, outliers.values]   # 2000 + 50 = 2050개

lof_out = LocalOutlierFactor(n_neighbors=20, contamination=0.1)

# fit_predict : 학습과 판정을 동시에. 반환값이 곧 판정 결과입니다.
#   (novelty=False에서는 predict가 아예 존재하지 않습니다)
labels = lof_out.fit_predict(X_mixed)

# 앞 2000개는 정상, 뒤 50개는 이상치라는 사실을 우리는 알고 있습니다.
n_train = len(X_train)
print("【모드 A】 아웃라이어 탐지 (novelty=False)")
print("  섞여 있던 이상치를 잡아낸 비율 :",
      round((labels[n_train:] == -1).mean(), 4))
print("  정상을 정상으로 본 비율        :",
      round((labels[:n_train] == 1).mean(), 4))

# predict를 호출하면 어떻게 될까요?
try:
    lof_out.predict(X_test.values)
except AttributeError as e:
    print("  predict() 호출 시 :", type(e).__name__)
    print("    →", str(e)[:70])
print()

# ── 모드 B : 노벨티 탐지 (novelty=True) ────────────────────
lof_nov = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
lof_nov.fit(X_train.values)      # 정상만 학습
print("【모드 B】 노벨티 탐지 (novelty=True)")
print("  새 이상치를 잡아낸 비율        :",
      round((lof_nov.predict(outliers.values) == -1).mean(), 4))
print("  새 정상을 통과시킨 비율        :",
      round((lof_nov.predict(X_test.values) == 1).mean(), 4))


# %% [Block 8] ▸ ② n_neighbors — 몇 명까지 이웃으로 볼 것인가
from sklearn.neighbors import LocalOutlierFactor

# 1단계에서 만든 C1(성긴 150개) + C2(빽빽한 150개) + o1 + o2 데이터를 씁니다.
#   ★ 한 무리가 150개라는 점을 기억해 두세요.
print(f"{'k':>5} | {'o1 점수':>8} | {'o2 점수':>8} | {'C1 정상 최대':>11} | 한 줄 해석")
print("-" * 78)

for k in [2, 5, 20, 50, 100, 250]:
    sc = -LocalOutlierFactor(n_neighbors=k).fit(X_lof).negative_outlier_factor_

    if k <= 5:
        note = "근시안 — 멀쩡한 C1 주민도 4.6까지 튐"
    elif k <= 100:
        note = "적정 — 이상치만 확실히 튐"
    else:
        note = "원시안 — 한 무리(150개)를 넘어섬!"

    print(f"{k:>5} | {sc[300]:>8.2f} | {sc[301]:>8.2f} | {sc[:150].max():>11.2f} | {note}")

print()
print("★ k=250 을 보세요. o1도 o2도 점수가 1.00 근처로 주저앉았습니다.")
print("   이웃 250명을 보면 한 무리(150개)를 넘어 '전체'를 보는 셈이 되어,")
print("   LOF의 정체성인 '국소성(local)'이 완전히 사라집니다.")
print("   → 두 이상치를 통째로 놓칩니다.")


# %% [Block 9] ▸ 프레임워크 비교
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

# ── 실습 4 정답 : 강의 6p 스타일 — 점수를 원 크기로 ─────────
X_mixed = np.r_[X_train.values, outliers.values]

lof = LocalOutlierFactor(n_neighbors=20).fit(X_mixed)
scores = -lof.negative_outlier_factor_   # 원래 LOF 점수 (클수록 이상)

# argsort()[-20:] : 점수가 가장 큰 20개의 '위치 번호'
top20 = np.argsort(scores)[-20:]

print("LOF 점수 상위 20개")
print("  점수 범위 :", round(scores[top20].min(), 3),
      "~", round(scores[top20].max(), 3))
print("  이 중 진짜 이상치(뒤쪽 50개)의 개수 :",
      (top20 >= len(X_train)).sum(), "/ 20")
print()
print("전체 데이터의 LOF 점수 분포")
print("  정상 2000개 평균 :", round(scores[:len(X_train)].mean(), 4))
print("  이상치 50개 평균 :", round(scores[len(X_train):].mean(), 4))
print()
print("# 그림을 그리려면 아래 코드를 실행하세요")
print("# plt.scatter(X_mixed[:,0], X_mixed[:,1], s=3, c='k')")
print("# plt.scatter(X_mixed[:,0], X_mixed[:,1], s=800*(scores-1),")
print("#             facecolors='none', edgecolors='r')")
