# -*- coding: utf-8 -*-
"""
01-12🎲 확률과통계기초 — 불확실성을 숫자로

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 01권  수포자들을 위한 머신러닝 수학 완전기초와 파이썬 맛보기/01-12🎲 확률과통계기초 — 불확실성을 숫자로.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 01-12-D 4단계 — 퍼짐 ↔️ 분산 · 표준편차 · 자유도(ddof)
import numpy as np

scores = np.array([70, 80, 80, 90, 100])   # 다섯 명의 시험 점수

mean = np.mean(scores)          # μ = (1/n)Σxᵢ
var_pop = np.var(scores)        # σ² : ddof=0 (모분산, n으로 나눔)
var_smp = np.var(scores, ddof=1) # s² : ddof=1 (표본분산, n-1로 나눔)
std_pop = np.std(scores)        # σ = √σ²
median = np.median(scores)      # 중앙값

print(f"평균      : {mean}")
print(f"중앙값    : {median}")
print(f"모분산    : {var_pop}    (ddof=0)")
print(f"표본분산  : {var_smp}    (ddof=1)")
print(f"표준편차  : {std_pop:.4f}")

# 직접 구현해서 np.var와 같은지 확인
deviation = scores - mean                 # ① 편차 (벡터 연산)
manual = np.sum(deviation ** 2) / len(scores)  # ②제곱 ③평균
print(f"직접 계산 : {manual}")


# %% [Block 2] 01-12-E 5단계 — 관계 🔀 공분산 · 상관계수 · 공분산 행렬 → PCA
import numpy as np

hours  = np.array([1, 2, 3, 4, 5])          # 공부 시간
scores = np.array([60, 65, 70, 80, 85])  # 시험 점수

# ① 손으로 계산한 그대로 구현 (n-1로 나눔)
dx = hours - hours.mean()       # x - x̄
dy = scores - scores.mean()     # y - ȳ
cov_manual = np.sum(dx * dy) / (len(hours) - 1)
print(f"공분산(직접)  : {cov_manual}")

# ② 넘파이 공분산 행렬 (기본 ddof=1)
cov_matrix = np.cov(hours, scores)
print("공분산 행렬:")
print(cov_matrix)

# ③ 상관계수 행렬 (단위 제거 → -1 ~ 1)
corr = np.corrcoef(hours, scores)
print(f"상관계수 r    : {corr[0, 1]:.4f}")

# ④ 단위를 '분'으로 바꿔보기 → 공분산은 변하지만 r은 불변!
cov_min = np.cov(hours * 60, scores)[0, 1]
corr_min = np.corrcoef(hours * 60, scores)[0, 1]
print(f"분 단위 공분산: {cov_min}  (60배!)")
print(f"분 단위 r     : {corr_min:.4f}  (그대로!)")


# %% [Block 3] 01-12-F 6단계 — 모양 🎨 왜도와 첨도
import numpy as np
from scipy import stats

# ① 앞의 시험 점수 데이터
scores = np.array([70, 80, 80, 90, 100])
print(f"왜도: {stats.skew(scores):.3f}   첨도: {stats.kurtosis(scores):.3f}")

# ② 소득처럼 오른쪽으로 심하게 치우친 데이터 만들기 (로그정규분포)
rng = np.random.default_rng(42)          # 재현성을 위한 시드 고정
income = rng.lognormal(mean=8.0, sigma=1.0, size=1000)

print(f"[원본]   평균 {income.mean():.1f} / 중앙값 {np.median(income):.1f}")
print(f"[원본]   왜도 {stats.skew(income):.3f}")

# ③ 로그 변환 — 큰 값을 강하게 압축해 꼬리를 끌어당긴다
log_income = np.log(income)
print(f"[로그후] 왜도 {stats.skew(log_income):.3f}")


# %% [Block 4] 01-12-G 7단계 — 베르누이 · 범주형 분포 🎯 분류 문제의 출력층
import numpy as np

def softmax(z):
    """로짓을 범주형 확률분포로 변환합니다.

    비유:
    반 학생들의 '득표수'를 '득표율(%)'로 바꾸는 것과 같습니다.
    단, 음수 득표는 있을 수 없으므로 먼저 exp로 전부 양수로 만듭니다."""
    z_shift = z - np.max(z)   # ⚠️ 오버플로 방지: 최댓값을 빼도 결과는 동일
    exp_z = np.exp(z_shift)   # ① 전부 양수로
    return exp_z / np.sum(exp_z)  # ② 합이 1이 되게 나눔

logits = np.array([2.0, 1.0, 0.1])   # 고양이 / 개 / 새
probs = softmax(logits)

print(f"확률분포 : {probs.round(4)}")
print(f"합       : {probs.sum():.4f}")

# 정답이 '고양이'(index 0)일 때의 교차 엔트로피 손실
loss = -np.log(probs[0])
print(f"손실     : {loss:.4f}   (-log(정답 확률))")

# 베르누이: 평균과 분산
p = 0.3
print(f"베르누이(p=0.3) 평균 {p}, 분산 {p*(1-p):.4f}")


# %% [Block 5] 01-12-H 8단계 — 정규분포 🔔 오차와 노이즈의 언어
import numpy as np
from scipy import stats

# cdf(x) = -∞부터 x까지의 넓이 = "미리 적분해 둔 함수" (🔗 01-11 7단계)
for k in [1, 2, 3]:
    area = stats.norm.cdf(k) - stats.norm.cdf(-k)
    print(f"P(-{k}σ ≤ Z ≤ +{k}σ) = {area*100:.2f}%")

# 실전: 평균 170cm, 표준편차 6cm인 집단에서 180cm 이상일 확률
mu, sigma = 170, 6
z = (180 - mu) / sigma                # 표준화 (z-score)
print(f"z-score      : {z:.4f}")
print(f"P(키 ≥ 180) : {(1 - stats.norm.cdf(z))*100:.2f}%")


# %% [Block 6] 01-12-J 10단계 — 최대가능도추정(MLE) 🎯 모든 손실 함수의 뿌리
import numpy as np

# 동전 10번 던져 앞면 7번 관측
n_trials, n_heads = 10, 7

# 가능한 모든 p 후보를 촘촘히 만든다 (= 연못 후보들)
p_grid = np.linspace(0.01, 0.99, 99)

# 각 후보에서 "이 데이터가 나올 확률" = 가능도
likelihood = p_grid**n_heads * (1 - p_grid)**(n_trials - n_heads)

best_p = p_grid[np.argmax(likelihood)]   # 가능도가 최대인 p
print(f"MLE 추정값 : {best_p:.2f}   (이론값 7/10 = 0.70)")

# 몇 개 후보의 가능도를 직접 비교해보자
for p in [0.3, 0.5, 0.7, 0.9]:
    L = p**n_heads * (1 - p)**(n_trials - n_heads)
    print(f"  p={p}: 가능도 {L:.6f}, 로그가능도 {np.log(L):.4f}")


# %% [Block 7] 01-12-K 11단계 — 가설검정 · p값 · 신뢰구간 ⚖️ "진짜 좋아진 건가, 운이 좋았나?"
import numpy as np
from scipy import stats

# 10-fold 교차검증에서 얻은 각 fold의 정확도
model_a = np.array([0.912, 0.905, 0.921, 0.898, 0.915,
                    0.908, 0.919, 0.902, 0.911, 0.907])
model_b = np.array([0.921, 0.918, 0.930, 0.915, 0.926,
                    0.919, 0.928, 0.913, 0.924, 0.920])

print(f"모델 A 평균: {model_a.mean():.4f}")
print(f"모델 B 평균: {model_b.mean():.4f}")
print(f"차이       : {model_b.mean()-model_a.mean():.4f}")

# ① 대응표본 t-검정 — 같은 fold끼리 짝지어 비교 (교차검증에 적합)
t_stat, p_val = stats.ttest_rel(model_a, model_b)
print(f"\n[대응표본] t = {t_stat:.4f},  p = {p_val:.8f}")

# ② 95% 신뢰구간 — "얼마나 좋아졌나"를 범위로 표현
for name, data in [("A", model_a), ("B", model_b)]:
    ci = stats.t.interval(0.95, len(data)-1,
                          loc=data.mean(), scale=stats.sem(data))
    print(f"모델 {name} 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

# ③ 판정
if p_val < 0.05:
    print("\n→ 귀무가설 기각: 차이는 우연으로 보기 어렵다")
else:
    print("\n→ 기각 실패: 증거가 부족하다 (차이 없음의 증명은 아님)")


# %% [Block 8] 01-12-L 12단계 — 통합 실습: 데이터 진단 파이프라인 만들기
import numpy as np
from scipy import stats

def diagnose(data, name="데이터"):
    """데이터의 중심·퍼짐·모양을 한 번에 진단합니다.

    비유:
    건강검진표와 같습니다. 키·몸무게(중심), 체지방 분포(퍼짐),
    체형(모양)을 한 장에 정리해서 '어디를 손봐야 하는지' 알려줍니다."""
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data, ddof=1)      # 표본이므로 ddof=1
    skew = stats.skew(data)
    kurt = stats.kurtosis(data)

    print(f"===== {name} 진단 =====")
    print(f"  📍 평균   {mean:10.3f}   중앙값 {median:10.3f}")
    print(f"  ↔️ 표준편차 {std:9.3f}")
    print(f"  🎨 왜도   {skew:10.3f}   첨도  {kurt:10.3f}")

    # 자동 진단 — 배운 규칙을 조건문으로
    if abs(skew) > 1:
        print("  ⚠️ 왜도가 큽니다 → 로그 변환(np.log1p)을 검토하세요")
    if kurt > 3:
        print("  ⚠️ 꼬리가 두껍습니다 → 이상치 처리를 검토하세요")

    # 3-시그마 규칙으로 이상치 탐지 (8단계)
    z = (data - mean) / std
    outliers = data[np.abs(z) > 3]
    print(f"  🔍 3σ 밖 이상치 {len(outliers)}개")
    return {"mean": mean, "std": std, "skew": skew}

rng = np.random.default_rng(42)
diagnose(rng.normal(170, 6, 1000), "키(정규분포)")
print()
diagnose(rng.lognormal(8.0, 1.0, 1000), "소득(치우친 분포)")


# %% [Block 9] 📝 연습문제
def my_variance(x, ddof=0):
    n = len(x)
    mean = sum(x) / n                                # ① 평균
    squared = sum((xi - mean) ** 2 for xi in x)     # ② 편차 제곱의 합
    return squared / (n - ddof)                     # ③ 자유도로 나눔
