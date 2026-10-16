# -*- coding: utf-8 -*-
"""
Zero to 머신러닝 딥러닝 Master — Medium 시리즈 #25 · 1부
확률과 통계 기초 — 평균·분산·공분산으로 데이터의 중심·퍼짐·관계 읽기 (1부)

원문(책): 01-12 「확률과 통계 기초 — 불확실성을 숫자로 (1부)」 https://wikidocs.net/439756
Medium : TODO (게시 후 추가)
책     : https://wikidocs.net/book/21464
홈페이지: https://mldict.net
블로그 : https://wikidocs.net/blog/@mldict/

실행: pip install -r requirements.txt && python 25_1_descriptive_statistics.py
※ 아래 코드는 책 본문의 코드 블록을 순서대로 옮긴 것입니다(내용 변경 없음).
"""

#======================================================================
# [1] 01-12-D 4단계 — 평균·분산·표준편차 (ddof)
#     np.var 의 기본 ddof=0(모분산)과 ddof=1(표본분산)을 비교하고, 편차 → 제곱 → 평균을 직접 구현해 같은 값인지 확인합니다.
#======================================================================
print("\n" + "=" * 60)
print("[1] 01-12-D 4단계 — 평균·분산·표준편차 (ddof)")
print("=" * 60)
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


#======================================================================
# [2] 01-12-E 5단계 — 공분산 · 상관계수 · 공분산 행렬
#     공부 시간과 점수의 공분산을 직접/np.cov 로 구하고, 상관계수 r 과 공분산 행렬(→ PCA 의 재료)을 확인합니다.
#======================================================================
print("\n" + "=" * 60)
print("[2] 01-12-E 5단계 — 공분산 · 상관계수 · 공분산 행렬")
print("=" * 60)
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


#======================================================================
# [3] 01-12-F 6단계 — 왜도·첨도 측정과 전처리
#     scipy.stats 로 왜도·첨도를 재고, 오른쪽으로 치우친 소득 데이터를 np.log1p 로 변환해 왜도가 줄어드는 것을 확인합니다.
#======================================================================
print("\n" + "=" * 60)
print("[3] 01-12-F 6단계 — 왜도·첨도 측정과 전처리")
print("=" * 60)
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


#======================================================================
# [4] 01-12-G 7단계 — 소프트맥스 직접 구현
#     로짓을 범주형 확률분포로 바꾸는 softmax 와 정답 클래스의 −log p (크로스엔트로피) 손실을 계산합니다.
#======================================================================
print("\n" + "=" * 60)
print("[4] 01-12-G 7단계 — 소프트맥스 직접 구현")
print("=" * 60)
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
