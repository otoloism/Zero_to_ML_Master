# -*- coding: utf-8 -*-
"""
Zero to 머신러닝 딥러닝 Master — Medium 시리즈 #25 · 2부
베이즈 정리와 최대가능도추정(MLE) — 정규분포부터 손실 함수의 뿌리, p값·신뢰구간까지 (2부)

원문(책): 01-12 「확률과 통계 기초 — 불확실성을 숫자로 (2부)」 https://wikidocs.net/439756
Medium : TODO (게시 후 추가)
책     : https://wikidocs.net/book/21464
홈페이지: https://mldict.net
블로그 : https://wikidocs.net/blog/@mldict/

실행: pip install -r requirements.txt && python 25_2_normal_bayes_mle_testing.py
※ 아래 코드는 책 본문의 코드 블록을 순서대로 옮긴 것입니다. [추가]/[수정] 표시가 있는 부분만 실행을 위해 보탰습니다.
"""

#======================================================================
# [1] 01-12-H 8단계 — 정규분포 확률 계산
#     stats.norm.cdf 의 차로 ±1σ·±2σ·±3σ 구간 확률(68.27% · 95.45% · 99.73%)을 구합니다.
#======================================================================
print("\n" + "=" * 60)
print("[1] 01-12-H 8단계 — 정규분포 확률 계산")
print("=" * 60)
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


#======================================================================
# [2] 01-12-J 10단계 — MLE 를 수치적으로 확인
#     p 후보를 촘촘히 깔고 가능도 p⁷(1−p)³ 이 최대가 되는 p 를 찾아 이론값 7/10 과 비교합니다.
#     ⚠️ 책 본문의 실행 결과 중 p=0.3·p=0.9 의 로그가능도(-9.4949 · -7.6461)는 실제 계산값(7·ln0.3+3·ln0.7 = -9.4978, 7·ln0.9+3·ln0.1 = -7.6453)과 소수 셋째 자리가 다릅니다. 코드는 그대로이며 이 노트북의 출력이 정확한 값입니다.
#======================================================================
print("\n" + "=" * 60)
print("[2] 01-12-J 10단계 — MLE 를 수치적으로 확인")
print("=" * 60)
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


#======================================================================
# [3] 01-12-K 11단계 — t-검정과 신뢰구간
#     두 모델의 fold 별 정확도로 대응표본 t-검정과 95% 신뢰구간을 계산해 '진짜 좋아진 건지'를 판단합니다.
#======================================================================
print("\n" + "=" * 60)
print("[3] 01-12-K 11단계 — t-검정과 신뢰구간")
print("=" * 60)
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


#======================================================================
# [4] 01-12-L 12단계 — 데이터 진단 파이프라인
#     평균·중앙값·표준편차·왜도·첨도·3σ 이상치를 한 번에 출력하는 diagnose() 로 정규분포와 치우친 분포를 진단합니다 (seed 42).
#======================================================================
print("\n" + "=" * 60)
print("[4] 01-12-L 12단계 — 데이터 진단 파이프라인")
print("=" * 60)
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


#======================================================================
# [5] 연습문제 A9 — 분산 직접 구현
#     평균 → 편차 제곱의 합 → 자유도(n − ddof)로 나누기, 세 줄로 분산을 구현합니다.
#     ⚠️ 연습문제 해설 코드는 함수 정의만 있어, 4단계의 시험 점수로 np.var 와 비교하는 코드를 추가했습니다(표시된 부분).
#======================================================================
print("\n" + "=" * 60)
print("[5] 연습문제 A9 — 분산 직접 구현")
print("=" * 60)
def my_variance(x, ddof=0):
    n = len(x)
    mean = sum(x) / n                                # ① 평균
    squared = sum((xi - mean) ** 2 for xi in x)     # ② 편차 제곱의 합
    return squared / (n - ddof)                     # ③ 자유도로 나눔

# [추가 코드 — 책 본문에는 없음] np.var 와 같은 값인지 확인 (4단계의 시험 점수)
import numpy as np
scores = [70, 80, 80, 90, 100]
print("직접 구현 ddof=0:", my_variance(scores), "| np.var          :", np.var(scores))
print("직접 구현 ddof=1:", my_variance(scores, ddof=1), "| np.var(ddof=1)  :", np.var(scores, ddof=1))
