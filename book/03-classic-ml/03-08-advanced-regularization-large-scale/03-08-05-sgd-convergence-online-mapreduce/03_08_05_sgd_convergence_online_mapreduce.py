# -*- coding: utf-8 -*-
"""
03-08-05 SGD 수렴 진단 · 온라인 학습 · 맵리듀스 (SGD Convergence · Online Learning · Map Reduce)

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-08장 - 정규화 심화 및 대규모 학습/03-08-05 SGD 수렴 진단 · 온라인 학습 · 맵리듀스.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 4단원 — 실제로 그려서 확인하기
import numpy as np

# ── 데이터 준비 (03-08-04와 같은 설정) ──────────────────
rng = np.random.default_rng(0)
m = 10000
X = np.c_[np.ones(m), rng.uniform(0, 10, m)]
y = X @ np.array([4.0, 3.0]) + rng.normal(0, 2, m)

def sgd_with_monitor(alpha, window=1000, seed=1):
    """SGD를 돌리면서 '최근 window개의 평균 비용'을 기록한다."""
    r = np.random.default_rng(seed)
    theta = np.zeros(2)
    costs = []                      # 샘플 하나짜리 비용을 차곡차곡 쌓는다

    for i in r.permutation(m):      # 1. 데이터를 무작위로 섞는다
        error = X[i] @ theta - y[i]

        # 2. 갱신하기 '전'에 이 샘플의 비용을 기록한다 (커닝 방지)
        costs.append(0.5 * error ** 2)

        # 3. 그다음 갱신한다
        theta = theta - alpha * error * X[i]

        # 값이 폭발했는지 확인 (np.isfinite : 무한대/nan이 아닌지 검사)
        if not np.isfinite(theta).all():
            return None, costs

    # 4. window개씩 끊어서 평균을 낸다
    c = np.array(costs)
    #   range(window, m+1, window) : 1000, 2000, 3000, ... 지점마다
    #   c[k-window:k] : 그 지점 직전 window개를 잘라낸다
    averaged = [c[k - window:k].mean() for k in range(window, m + 1, window)]
    return theta, averaged

# 학습률을 바꿔가며 곡선을 비교한다
print("%-10s %s" % ("alpha", "최근 1000개 평균 비용 (1000스텝마다)"))
print("-" * 76)
for alpha in [0.001, 0.01, 0.05]:
    theta, curve = sgd_with_monitor(alpha)
    if theta is None or max(curve) > 1e6:
        print("%-10s 발산! 비용이 폭발했다 → 패턴 ④" % alpha)
    else:
        print("%-10s %s" % (alpha, np.round(curve, 2)))


# %% [Block 2] 결과 해석 — 창(window) 크기를 바꿔 같은 학습을 다르게 바라보기
# 창(window) 크기를 바꿔 같은 학습을 다르게 바라보기
theta, curve_small = sgd_with_monitor(0.01, window=500)
theta, curve_big   = sgd_with_monitor(0.01, window=2500)

print("창 = 500  (지저분하지만 변화를 빨리 본다)")
print("  ", np.round(curve_small[:10], 2), "...")
print()
print("창 = 2500 (매끄럽지만 변화를 늦게 본다)")
print("  ", np.round(curve_big, 2))
print()
print("→ 같은 학습인데 창 크기만 바꿔도 그래프의 인상이 완전히 달라진다.")
print("   '요동친다'고 느껴지면 먼저 창을 키워보라는 조언이 여기서 나온다.")


# %% [Block 3] 일반 로지스틱 회귀와 무엇이 다른가
import numpy as np

def sigmoid(z):
    """시그모이드 함수 (03-03-03). 어떤 숫자든 0~1 사이 확률로 바꾼다."""
    return 1 / (1 + np.exp(-z))

# ── 온라인 학습 시뮬레이션 ──────────────────────────────
#    도중에 '사용자 성향이 급변'하는 상황을 만들어 대응하는지 본다
r = np.random.default_rng(1)
theta = np.zeros(3)          # 학습할 파라미터 (절편 + 특성 2개)
alpha = 0.05

def get_next_user(t):
    """t번째로 접속한 사용자 한 명을 만들어낸다. (실제로는 웹서버가 주는 데이터)"""
    x = np.r_[1.0, r.uniform(0, 1, 2)]        # np.r_ : 값들을 이어붙여 1차원 배열로

    # 3000번째 이전과 이후의 '진짜 성향'이 완전히 다르다 (성향 급변!)
    if t < 3000:
        true_w = np.array([-1.0, 3.0, 1.0])   # 첫 번째 특성을 중시하던 시기
    else:
        true_w = np.array([-1.0, -1.0, 4.0])  # 두 번째 특성을 중시하는 시기로 급변

    p = sigmoid(x @ true_w)
    return x, float(r.random() < p)           # 확률 p로 y=1

hits = []
for t in range(6000):
    x, y = get_next_user(t)                   # ① 사용자 한 명이 온다

    p = sigmoid(x @ theta)                    # ② 현재 파라미터로 예측
    hits.append(float((p > 0.5) == (y > 0.5)))   # 맞혔는지 기록

    theta = theta - alpha * (p - y) * x       # ③ 그 사람으로 즉시 갱신
    # ④ x, y는 여기서 버려진다. 다음 반복에서 새 값으로 덮어써진다.

hits = np.array(hits)
print("구간별 예측 적중률")
for s in range(0, 6000, 1000):
    mark = "  ← 성향이 급변한 직후!" if s == 3000 else ""
    print("  %4d ~ %4d 스텝 : %.3f%s" % (s, s + 999, hits[s:s + 1000].mean(), mark))
print()
print("학습된 theta :", np.round(theta, 2))
print("바뀐 뒤의 정답 :", np.array([-1.0, -1.0, 4.0]))


# %% [Block 4] 왜 이것이 가능한가 — 덧셈의 성질
import numpy as np

# ── 데이터 400개, 특성 3개 ─────────────────────────────
rng = np.random.default_rng(7)
m, n = 400, 3
X = np.c_[np.ones(m), rng.normal(0, 1, (m, n))]
y = X @ np.array([1.0, 2.0, -1.0, 0.5]) + rng.normal(0, 1, m)
theta = rng.normal(0, 1, n + 1)          # 현재 파라미터(아무 값이나)

# ── 방법 A : 한 대가 전부 계산 ─────────────────────────
full_sum = X.T @ (X @ theta - y)

# ── 방법 B : 4대에 100개씩 나눠 계산 ───────────────────
temps = []
for k in range(4):
    start, end = k * 100, (k + 1) * 100          # 이 기계가 맡을 구간
    X_part, y_part = X[start:end], y[start:end]  # 자기 몫만 가져간다
    temp_j = X_part.T @ (X_part @ theta - y_part)   # temp_j 계산 (Map 단계)
    temps.append(temp_j)
    print("Machine %d 의 temp_j :" % (k + 1), np.round(temp_j, 3))

combined = sum(temps)                            # 네 결과를 합친다 (Reduce 단계)

print()
print("한 대가 전부 계산 :", np.round(full_sum, 6))
print("4대 결과를 합산   :", np.round(combined, 6))
print("최대 차이         :", np.abs(full_sum - combined).max())
print()
print("→ 차이가 사실상 0이다(부동소수점 오차 수준).")
print("   계산을 쪼개도 결과가 똑같다는 것이 맵리듀스의 근거다.")


# %% [Block 5] 실습 — 실습 2 정답 — 학습률이 '변화 대응 속도'를 어떻게 바꾸는가
# 실습 2 정답 — 학습률이 '변화 대응 속도'를 어떻게 바꾸는가
def online_learning(alpha, seed=1):
    r = np.random.default_rng(seed)
    theta = np.zeros(3)
    hits = []
    for t in range(6000):
        x = np.r_[1.0, r.uniform(0, 1, 2)]
        true_w = np.array([-1.0, 3.0, 1.0]) if t < 3000 else np.array([-1.0, -1.0, 4.0])
        y = float(r.random() < sigmoid(x @ true_w))
        p = sigmoid(x @ theta)
        hits.append(float((p > 0.5) == (y > 0.5)))
        theta = theta - alpha * (p - y) * x
    return np.array(hits), theta

print("%-10s %12s %12s %12s" % ("alpha", "변화 직전", "변화 직후", "회복 후"))
print("-" * 50)
for alpha in [0.005, 0.05, 0.2]:
    hits, th = online_learning(alpha)
    print("%-10s %12.3f %12.3f %12.3f" % (
        alpha,
        hits[2000:3000].mean(),      # 성향이 바뀌기 직전
        hits[3000:4000].mean(),      # 바뀐 직후 (충격 구간)
        hits[5000:6000].mean()))     # 충분히 지난 뒤
print()
print("→ alpha가 크면 변화에 빨리 적응하지만 평소 예측이 불안정하다.")
print("→ alpha가 작으면 평소엔 안정적이지만 세상이 바뀌어도 느리게 따라간다.")
print("   '얼마나 빨리 잊을 것인가'를 정하는 손잡이가 alpha다.")


# %% [Block 6] 실습 — 실습 3 정답 — 몇 대로 쪼개든 결과는 같은가?
# 실습 3 정답 — 몇 대로 쪼개든 결과는 같은가?
def map_reduce_gradient(n_machines):
    """데이터를 n_machines대로 나눠 기울기를 계산하고 합친다."""
    chunk = m // n_machines               # // 는 몫만 취하는 나눗셈
    temps = []
    for k in range(n_machines):
        start = k * chunk
        # 마지막 기계는 남은 것을 전부 가져간다(나눠떨어지지 않는 경우 대비)
        end = m if k == n_machines - 1 else (k + 1) * chunk
        Xp, yp = X[start:end], y[start:end]
        temps.append(Xp.T @ (Xp @ theta - yp))
    return sum(temps)

baseline = X.T @ (X @ theta - y)          # 한 대로 계산한 정답
print("%-12s %14s" % ("기계 수", "정답과의 최대 차이"))
print("-" * 30)
for n_mach in [1, 2, 4, 8, 16]:
    diff = np.abs(map_reduce_gradient(n_mach) - baseline).max()
    print("%-12d %14.2e" % (n_mach, diff))
print()
print("→ 몇 대로 쪼개든 차이는 1e-13 수준, 즉 부동소수점 오차뿐이다.")
print("   덧셈의 결합법칙이 분산 학습의 수학적 보증서인 셈이다.")
