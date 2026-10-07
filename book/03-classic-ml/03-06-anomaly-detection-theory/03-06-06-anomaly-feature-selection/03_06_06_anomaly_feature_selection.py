# -*- coding: utf-8 -*-
"""
03-06-06 이상탐지에서의 Feature 선택 — (Choosing Features for Anomaly Detection) - 알고리즘보다 재료가 성능을 결정한다

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-06장 - 이상탐지 이론/03-06-06 이상탐지에서의 Feature 선택.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 코드
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

rng = np.random.default_rng(7)

# lognormal : '로그를 씌우면 정규분포가 되는' 분포.
#   응답 시간, 소득, 거래 금액처럼 현실에서 매우 흔한 모양입니다.
#   mean/sigma 는 '로그를 씌운 뒤'의 평균·표준편차임에 주의!
raw = rng.lognormal(mean=1.0, sigma=0.7, size=5000)

# skew(왜도) : 분포가 얼마나 한쪽으로 치우쳤는지를 나타내는 숫자.
#   0에 가까울수록 좌우 대칭(가우시안에 가까움)
#   양수 = 오른쪽 꼬리가 김 / 음수 = 왼쪽 꼬리가 김
#   경험칙 : |왜도| < 0.5 이면 "대체로 대칭"으로 봐도 무방
print("원본        왜도 = %+.4f" % skew(raw))
print("log(x)      왜도 = %+.4f" % skew(np.log(raw)))
print("sqrt(x)     왜도 = %+.4f" % skew(np.sqrt(raw)))
print("x**(1/3)    왜도 = %+.4f" % skew(raw ** (1/3)))

# ── 네 가지를 나란히 그려 비교 ─────────────────────────────
# subplots(1, 4) : 1행 4열. axes 는 그림 4개가 담긴 배열이 됩니다.
fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))

# 이름과 변환된 데이터를 짝지어 리스트로 만듭니다.
#   튜플 (이름, 데이터) 을 담고 for 문에서 한꺼번에 꺼내 씁니다.
variants = [
    ("raw x",       raw),
    ("log(x)",      np.log(raw)),
    ("sqrt(x)",     np.sqrt(raw)),
    ("x**(1/3)",    raw ** (1/3)),
]

for ax, (name, data) in zip(axes, variants):
    # zip : 두 리스트를 지퍼처럼 하나씩 짝지어 꺼내주는 함수
    ax.hist(data, bins=50, density=True, color='tab:blue', alpha=0.65)
    ax.set_title(f"{name}\nskew = {skew(data):+.3f}")

plt.tight_layout()
plt.show()


# %% [Block 2] 실험 — 비율 특성 하나가 만드는 차이
import numpy as np
rng = np.random.default_rng(7)
n = 6000

# ── 정상 서버 시뮬레이션 ──────────────────────────────────
# 핵심 설정 : 정상 서버는 "일이 많으면 CPU도 트래픽도 함께" 올라갑니다.
#   → 공통 요인(base)을 만들고 두 특성이 그것을 함께 따르게 합니다.
#   → 결과적으로 두 특성은 강한 양의 상관을 갖습니다.
base = rng.normal(50, 12, n)                       # 그 서버의 '일감의 양'

# np.clip(값, 최솟값, 최댓값) : 범위를 벗어나면 잘라냅니다.
#   None 은 '제한 없음'. 여기서는 음수만 막습니다(부하는 음수일 수 없으니까).
cpu = np.clip(base + rng.normal(0, 3, n), 1, None)
net = np.clip(base * 1.8 + rng.normal(0, 6, n), 1, None)

# np.c_ : 1차원 배열들을 '열'로 세워 붙입니다. (n,) + (n,) → (n, 2)
X2 = np.c_[cpu, net]

def fit_and_threshold(X, q=0.001):
    """
    파라미터를 추정하고, 정상 데이터의 하위 0.1% 지점을 ε 으로 잡는다.
    (라벨 없이 ε 을 정하는 간이 방법 — "정상 중 가장 드문 0.1%까지는 허용")
    """
    mu, sigma2 = X.mean(axis=0), X.var(axis=0)

    def p(Z):
        coef = 1.0 / np.sqrt(2 * np.pi * sigma2)
        return np.prod(coef * np.exp(-((Z - mu) ** 2) / (2 * sigma2)), axis=1)

    # np.quantile(데이터, 0.001) : 아래에서 0.1% 위치의 값
    return mu, sigma2, p, np.quantile(p(X), q)

# ══ 실험 A : 특성 2개 (CPU, 트래픽) ═══════════════════════
mu2, s2_2, p2, eps2 = fit_and_threshold(X2)

# 문제의 서버 : CPU는 높은데(70) 트래픽은 낮다(55)
#   → 무한 루프에 빠져 '일은 안 하면서 CPU만 태우는' 전형적 장애
bad = np.array([[70.0, 55.0]])
z2 = (bad - mu2) / np.sqrt(s2_2)

print("[A] 특성 2개")
print(f"  z = ({z2[0,0]:+.2f}, {z2[0,1]:+.2f})")
print(f"  p(x) = {p2(bad)[0]:.4g},  ε = {eps2:.4g}")
print(f"  판정: {'이상 🚨' if p2(bad)[0] < eps2 else '놓침 😱'}")

# ══ 실험 B : 비율 특성 x₃ = CPU / 트래픽 추가 ═════════════
ratio = cpu / net                       # ★ 이 한 줄이 핵심입니다
X3 = np.c_[cpu, net, ratio]
mu3, s2_3, p3, eps3 = fit_and_threshold(X3)

bad3 = np.array([[70.0, 55.0, 70.0 / 55.0]])
z3 = (bad3[0, 2] - mu3[2]) / np.sqrt(s2_3[2])

print("\n[B] 비율 특성 추가")
print(f"  정상 비율: 평균 {ratio.mean():.4f}, 표준편차 {ratio.std():.4f}")
print(f"  문제 서버 비율: {70/55:.4f}  →  z = {z3:+.2f}")
print(f"  p(x) = {p3(bad3)[0]:.4g},  ε = {eps3:.4g}")
print(f"  판정: {'이상 🚨' if p3(bad3)[0] < eps3 else '놓침 😱'}")
