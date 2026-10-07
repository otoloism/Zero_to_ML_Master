# -*- coding: utf-8 -*-
"""
03-10-04 Isolation Forest — 스무고개로 범인 찾기

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-10장 - 이상탐지·교차검증 실습/03-10-04 Isolation Forest — 스무고개로 범인 찾기.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] ▸ 직접 시뮬레이션해 보기
import numpy as np

def isolate_depth(X, target_idx, seed):
    """
    한 점(target_idx)이 '혼자 남을 때까지' 무작위 분할을 몇 번 해야 하는지 셉니다.
    Isolation Tree를 한 그루 만들면서 그 점을 따라 내려가는 것과 같습니다.
    """
    r = np.random.RandomState(seed)
    idx = np.arange(len(X))   # 현재 묶음에 남아 있는 점들의 번호
    depth = 0                 # 지금까지 자른 횟수

    # 묶음에 2개 이상 남아 있는 동안 계속 자릅니다.
    while len(idx) > 1 and depth < 50:
        lo = X[idx].min(axis=0)   # 현재 묶음의 각 특성 최솟값
        hi = X[idx].max(axis=0)   # 최댓값

        # 최솟값 == 최댓값인 특성은 자를 수 없으므로 후보에서 뺍니다.
        cand = [f for f in range(X.shape[1]) if hi[f] > lo[f]]
        if not cand:
            break

        f = r.choice(cand)                # ① 특성을 무작위로 하나
        v = r.uniform(lo[f], hi[f])       # ② 그 범위 안에서 값을 무작위로 하나

        left = idx[X[idx, f] < v]         # ③ 두 묶음으로 쪼개기
        right = idx[X[idx, f] >= v]

        # 우리가 추적하는 점이 들어간 쪽만 남기고 계속 내려갑니다.
        idx = left if target_idx in left else right
        depth += 1

    return depth

# 정상 200개 + 이상치 5개로 작은 실험판을 만듭니다.
X_sim = np.r_[X_train.values[:200], outliers.values[:5]]

# 트리 30그루를 만든다고 생각하고 평균을 냅니다.
print("【정상 점 5개】 30그루 평균 격리 깊이")
for i in range(5):
    d = np.mean([isolate_depth(X_sim, i, s) for s in range(30)])
    print(f"   {i}번 점 {X_sim[i].round(2)} → {d:.2f} 번 만에 격리")

print()
print("【이상치 5개】 30그루 평균 격리 깊이")
for i in range(5):
    d = np.mean([isolate_depth(X_sim, 200 + i, s) for s in range(30)])
    print(f"   {200+i}번 점 {X_sim[200+i].round(2)} → {d:.2f} 번 만에 격리")


# %% [Block 2] ▸ 유도 — 왜 이 식인가
import numpy as np

def c_factor(n):
    """
    데이터 n개짜리 iTree에서 '평범한 점'의 평균 경로 길이.
    Isolation Forest 논문(Liu et al., 2008)의 식 (1).
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    # 조화수 H(n-1) 근사 : ln(n-1) + 오일러-마스케로니 상수
    #   (정확히 더하려면 1 + 1/2 + ... + 1/(n-1) 이지만 n이 크면 이 근사가 매우 정확합니다)
    euler_gamma = 0.5772156649
    H = np.log(n - 1) + euler_gamma
    return 2 * H - 2 * (n - 1) / n

print(f"{'데이터 개수 n':>14} | {'c(n)':>8} | 해석")
print("-" * 62)
for n in [2, 10, 100, 256, 1000, 100000]:
    if n == 256:
        note = "← 사이킷런 기본 max_samples"
    elif n == 2:
        note = "두 점이면 한 번만 자르면 끝"
    else:
        note = ""
    print(f"{n:>14,} | {c_factor(n):>8.4f} | {note}")

print()
print("데이터가 100개 → 100,000개로 1000배 늘어도")
print(f"c(n)은 {c_factor(100):.2f} → {c_factor(100000):.2f} 로 두 배가 조금 넘게 늘 뿐입니다.")
print("로그 속도로 자라기 때문입니다.")


# %% [Block 3] ▸ 구현 관점 — 사이킷런은 부호를 뒤집는다
import numpy as np
from sklearn.ensemble import IsolationForest

clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(X_train.values)

# 사이킷런 점수에 마이너스를 붙이면 논문의 s(x, n) 이 됩니다.
s_normal = -clf.score_samples(X_test.values[:5])
s_outlier = -clf.score_samples(outliers.values[:5])

print("논문 기준 이상 점수 s(x, n)  — 1에 가까울수록 이상")
print("  정상 데이터 5개 :", np.round(s_normal, 4))
print("  이상치   5개    :", np.round(s_outlier, 4))
print()
print("  정상 전체 평균  :", round(float((-clf.score_samples(X_test.values)).mean()), 4))
print("  이상치 전체 평균:", round(float((-clf.score_samples(outliers.values)).mean()), 4))
print()
print("offset_ (임계값) :", round(clf.offset_, 4))
print("→ score_samples 가 이 값보다 작으면 이상(-1)으로 판정합니다.")
print("→ 논문 점수로 바꾸면 s > ", round(-clf.offset_, 4), " 이면 이상.")


# %% [Block 4] ▸ 구현 관점 — 사이킷런은 부호를 뒤집는다
from sklearn.ensemble import IsolationForest

X_outliers = outliers.copy()

# ── 모델 만들기 ────────────────────────────────────────────
#  contamination=0.1 : 하위 10% 지점을 임계값으로
#  random_state=42   : 트리를 무작위로 만들므로 씨앗 고정이 필수입니다
#                      (이 인자를 빼면 실행할 때마다 결과가 달라집니다)
clf = IsolationForest(contamination=0.1, random_state=42)

# fit 안에서 벌어지는 일:
#   ① 트리 100그루를 만든다 (n_estimators 기본값 100)
#   ② 각 그루마다 전체에서 256개를 무작위로 뽑아 쓴다 (max_samples='auto')
#   ③ 훈련 데이터 전체의 점수를 계산하고 하위 10% 지점을 offset_으로 저장
clf.fit(X_train)

y_pred_train    = clf.predict(X_train)
y_pred_test     = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

print("만든 트리 개수     :", clf.n_estimators)
print("트리 한 그루당 표본:", clf.max_samples_, "개")
print("검증 앞 10개 판정  :", y_pred_test[:10])


# %% [Block 5] ▸ 구현 관점 — 사이킷런은 부호를 뒤집는다
print("테스트 데이터셋에서 정확도:",
      list(y_pred_test).count(1) / y_pred_test.shape[0])
print("이상치 데이터셋에서 정확도:",
      list(y_pred_outliers).count(-1) / y_pred_outliers.shape[0])


# %% [Block 6] ▸ 세 모델 최종 비교
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

models = [
    ("EllipticEnvelope", EllipticEnvelope(contamination=0.1, random_state=42)),
    ("LocalOutlierFactor", LocalOutlierFactor(n_neighbors=20, novelty=True,
                                              contamination=0.1)),
    ("IsolationForest", IsolationForest(contamination=0.1, random_state=42)),
]

print(f"{'모델':>20} | {'훈련':>8} | {'정상 통과율':>10} | {'이상치 검출율':>12}")
print("-" * 62)
for name, m in models:
    m.fit(X_train.values)
    tr = (m.predict(X_train.values) == 1).mean()
    te = (m.predict(X_test.values) == 1).mean()
    ou = (m.predict(outliers.values) == -1).mean()
    print(f"{name:>20} | {tr:>8.4f} | {te:>10.4f} | {ou:>12.4f}")

print()
print("훈련 정확도가 셋 다 0.90 근처인 것에 주목하세요.")
print("contamination=0.1 이 정확히 10%를 잘라 냈기 때문입니다.")


# %% [Block 7] ▸ 결과 시각화 — 강의 자료 19p
import matplotlib.pyplot as plt

X_outliers = X_outliers.assign(y=y_pred_outliers)

plt.scatter(X_train.x1, X_train.x2,
            c='white', s=20*4, edgecolor='k',
            label="training observations")
plt.scatter(X_outliers.loc[X_outliers.y == -1, ['x1']],
            X_outliers.loc[X_outliers.y == -1, ['x2']],
            c='red', s=20*4, edgecolor='k',
            label="detected outliers")
plt.scatter(X_outliers.loc[X_outliers.y == 1, ['x1']],
            X_outliers.loc[X_outliers.y == 1, ['x2']],
            c='green', s=20*4, edgecolor='k',
            label="detected regular obs")
plt.legend(loc='upper right')
plt.show()


# %% [Block 8] ▸ ② 속도 — 이 모델이 존재하는 진짜 이유
import numpy as np
import time
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

rngS = np.random.RandomState(0)

print(f"{'데이터 크기':>12} | {'IsolationForest':>16} | {'LOF':>10} | "
      f"{'EllipticEnv':>12} | {'LOF/IF':>8}")
print("-" * 74)

for m in [2000, 10000, 50000]:
    # 특성 10개짜리 데이터를 만듭니다.
    X_big = rngS.randn(m, 10)

    # time.time() : 현재 시각(초). 앞뒤로 재서 빼면 걸린 시간이 나옵니다.
    t = time.time(); IsolationForest(random_state=42).fit(X_big)
    t_if = time.time() - t

    t = time.time(); LocalOutlierFactor(n_neighbors=20).fit_predict(X_big)
    t_lof = time.time() - t

    t = time.time(); EllipticEnvelope(random_state=42).fit(X_big)
    t_ee = time.time() - t

    print(f"{m:>12,} | {t_if:>15.2f}s | {t_lof:>9.2f}s | "
          f"{t_ee:>11.2f}s | {t_lof/t_if:>7.1f}배")

print()
print("★ IsolationForest는 데이터가 25배로 늘어도 시간이 거의 그대로입니다.")
print("   트리마다 256개만 뽑아 쓰기 때문입니다 (max_samples='auto').")
print("★ LOF는 이웃을 찾느라 시간이 폭발합니다 (거의 O(m²)).")


# %% [Block 9] ▸ ③ 알려진 약점 — 축에 평행한 선만 긋는다
import numpy as np
from sklearn.ensemble import IsolationForest

rngD = np.random.RandomState(3)

# ── 대각선으로 길게 늘어선 정상 데이터를 만듭니다 ──────────────
t = rngD.uniform(-3, 3, 1000)
X_diag = np.c_[t + rngD.randn(1000) * 0.1,
               t + rngD.randn(1000) * 0.1]   # x1 ≈ x2 인 얇은 띠

# 띠에서 '수직으로' 벗어난 점들과, 띠 위의 정상 점을 함께 시험합니다.
#   ※ 각 점의 좌표는 x1, x2 각각으로 보면 모두 정상 범위(-3~3) 안입니다.
#     "각 축만 보면 정상인데 조합으로 보면 이상"인 경우가 핵심입니다.

iso = IsolationForest(contamination=0.05, random_state=42).fit(X_diag)
lof = LocalOutlierFactor(n_neighbors=20, novelty=True,
                         contamination=0.05).fit(X_diag)

labels = ["(-1.5, 1.5) 띠 밖", "(-0.6, 0.6) 띠 밖",
          "(0, 0) 띠 한복판", "(3, 3) 띠의 끝(정상)"]
probe = np.array([[-1.5, 1.5], [-0.6, 0.6], [0.0, 0.0], [3.0, 3.0]])

print(f"{'점':>20} | {'IF 판정':>8} {'IF 점수':>8} | {'LOF 판정':>9} {'LOF 점수':>9}")
print("-" * 68)
for lab, p, si, pl, sl in zip(labels,
                              iso.predict(probe), -iso.score_samples(probe),
                              lof.predict(probe), -lof.score_samples(probe)):
    pi_s = "정상(+1)" if p == 1 else "이상(-1)"
    pl_s = "정상(+1)" if pl == 1 else "이상(-1)"
    print(f"{lab:>20} | {pi_s:>8} {si:>8.4f} | {pl_s:>9} {sl:>9.3f}")

print()
print("★ 문제 1 — 점수의 대비가 거의 없습니다.")
print("   IF  : 정상 0.46 vs 이상 0.56  (차이 겨우 0.10)")
print("   LOF : 정상 1.02 vs 이상 4.98  (차이 4배 가까이)")
print("   IF는 겨우 임계값을 넘겼을 뿐 '확신'이 없습니다.")
print()
print("★ 문제 2 — 띠의 끝(3,3)을 이상이라고 잘못 경보합니다.")
print("   축 방향으로 보면 3은 정상 범위의 가장자리라 빨리 격리되기 때문입니다.")
print("   LOF는 이 점을 정확히 정상으로 봅니다.")
print()
print("→ 원인은 하나입니다. '축에 평행한 선'으로만 자를 수 있기 때문입니다.")
print("   이 한계를 고친 것이 Extended Isolation Forest(2018)입니다.")


# %% [Block 10] ▸ 프레임워크 비교
import numpy as np
from sklearn.ensemble import IsolationForest

# ── 실습 1 + 2 정답을 한 번에 ──────────────────────────────
print("【실습 1】 n_estimators (트리 개수) 바꾸기")
print(f"{'트리 개수':>10} | {'정상 통과율':>10} | {'이상치 검출율':>12}")
print("-" * 40)
for n_tree in [1, 10, 100, 500]:
    m = IsolationForest(n_estimators=n_tree, contamination=0.1,
                        random_state=42).fit(X_train.values)
    a = (m.predict(X_test.values) == 1).mean()
    b = (m.predict(outliers.values) == -1).mean()
    print(f"{n_tree:>10} | {a:>10.4f} | {b:>12.4f}")

print()
print("【실습 2】 max_samples (트리당 표본 수) 바꾸기")
print(f"{'표본 수':>10} | {'정상 통과율':>10} | {'이상치 검출율':>12}")
print("-" * 40)
for ms in [32, 128, 256, 1000, 2000]:
    m = IsolationForest(max_samples=ms, contamination=0.1,
                        random_state=42).fit(X_train.values)
    a = (m.predict(X_test.values) == 1).mean()
    b = (m.predict(outliers.values) == -1).mean()
    print(f"{ms:>10} | {a:>10.4f} | {b:>12.4f}")

print()
print("【실습 3】 random_state 없이 5번 실행 (트리 10그루로 축소)")
#   트리가 100그루면 평균이 잘 잡혀 변동이 거의 안 보입니다.
#   무작위성의 영향을 눈으로 보려고 일부러 10그루로 줄였습니다.
accs = []
for _ in range(5):
    m = IsolationForest(n_estimators=10, contamination=0.1).fit(X_train.values)
    accs.append(round(float((m.predict(outliers.values) == -1).mean()), 4))
print("  5번의 이상치 검출율 :", accs)
print("  변동폭 :", round(max(accs) - min(accs), 4))
print()
print("  → 씨앗을 고정하지 않으면 같은 코드가 매번 다른 답을 냅니다.")
print("    트리를 100그루로 늘리면 평균이 안정되어 변동이 줄어듭니다.")
