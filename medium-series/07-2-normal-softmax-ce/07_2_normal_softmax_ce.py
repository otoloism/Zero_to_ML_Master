# -*- coding: utf-8 -*-
"""
Zero to 머신러닝 딥러닝 Master — Medium 시리즈 #07 · 2부
정규분포·소프트맥스·교차 엔트로피 — AI는 확률로 대답한다 (2부)

원문(책): 00-05 「확률·베이즈 정리 — 병원 검사의 함정 (2부)」 https://wikidocs.net/439838
Medium : TODO (게시 후 추가)
책     : https://wikidocs.net/book/21464
홈페이지: https://mldict.net
블로그 : https://wikidocs.net/blog/@mldict/

실행: pip install -r requirements.txt && python 07_2_normal_softmax_ce.py
※ 아래 코드는 책 본문의 코드 블록을 순서대로 옮긴 것입니다(내용 변경 없음).
"""

#======================================================================
# [1] 00-05-D · 4단계 — 정규분포와 68-95-99.7 규칙
#     PDF 직접 구현 vs SciPy, 10만 개 표본으로 규칙 검증, 주사위 평균으로 중심극한정리를 확인합니다 (seed 42).
#======================================================================
print("\n" + "=" * 60)
print("[1] 00-05-D · 4단계 — 정규분포와 68-95-99.7 규칙")
print("=" * 60)
import numpy as np
import math
from scipy import stats

def normal_pdf(x, mu, sigma):
    """
    정규분포 확률밀도함수를 공식 그대로 구현합니다.
        f(x) = 1/(σ√(2π)) · exp(-(x-μ)²/(2σ²))

    비유:
    과녁 중심(μ)에서 멀어질수록 화살 자국이
    급격히 드물어지는 정도를 수식으로 나타낸 것입니다.
    """
    coefficient = 1 / (sigma * math.sqrt(2 * math.pi))   # 정규화 상수
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)     # 지수 부분
    return coefficient * math.exp(exponent)

def z_score(x, mu, sigma):
    """z = (x-μ)/σ — 평균에서 표준편차 몇 개만큼 떨어졌는가"""
    return (x - mu) / sigma

mu, sigma = 170, 8

# ══════ ① 공식 직접 구현 vs SciPy 비교 ══════
print("--- PDF 검증 (직접 구현 vs SciPy) ---")
for x in [170, 178, 186]:
    mine = normal_pdf(x, mu, sigma)
    lib = stats.norm.pdf(x, mu, sigma)
    print(f"  x={x}: 직접 {mine:.6f} / SciPy {lib:.6f}"
          f" / z={z_score(x,mu,sigma):+.1f}")

# ══════ ② 68-95-99.7 규칙을 데이터로 검증 ══════
rng = np.random.default_rng(42)
heights = rng.normal(mu, sigma, 100_000)   # ⚠️ 2번째 인자는 '표준편차'

print("\n--- 68-95-99.7 규칙 검증 ---")
for k in [1, 2, 3]:
    # |x-μ| ≤ kσ 인 데이터의 비율 (불리언 평균 = 비율!)
    empirical = np.mean(np.abs(heights - mu) <= k * sigma)
    theory = stats.norm.cdf(k) - stats.norm.cdf(-k)
    print(f"  ±{k}σ: 실측 {empirical:.4f} / 이론 {theory:.4f}")

# ══════ ③ "186cm 이상"인 사람의 비율 ══════
print(f"\n186cm 이상 이론값: {1 - stats.norm.cdf(2):.4f}", "(z=2)")
print(f"186cm 이상 실측값: {np.mean(heights >= 186):.4f}")

# ══════ ④ 중심극한정리 — 왜 정규분포가 흔한가? ══════
# 주사위(전혀 종 모양이 아님!)를 n개 굴려 평균을 내면
# n이 커질수록 그 평균은 정규분포에 가까워집니다
print("\n--- 중심극한정리: 주사위 n개의 평균 ---")
for n in [1, 2, 5, 30]:
    means = rng.integers(1, 7, (20_000, n)).mean(axis=1)   # 행마다 평균
    print(f"  주사위 {n:2d}개 → 평균 {means.mean():.3f}, "
          f"표준편차 {means.std():.3f} (이론 {1.7078/math.sqrt(n):.3f})")


#======================================================================
# [2] 00-05-E · 5단계 — 소프트맥스와 교차 엔트로피
#     max 빼기로 안전한 softmax, 확신 정도에 따른 손실, 그리고 dL/dz = ŷ − y 를 수치미분으로 검증합니다.
#======================================================================
print("\n" + "=" * 60)
print("[2] 00-05-E · 5단계 — 소프트맥스와 교차 엔트로피")
print("=" * 60)
import numpy as np

def softmax(z):
    """
    원시 점수를 합이 1인 확률로 변환합니다.
        softmax(z_i) = exp(z_i) / Σ exp(z_j)

    비유:
    후보들의 제각각인 점수를 항상 합계 100%의 득표율로
    환산하는 선거관리위원회와 같습니다.
    """
    # ★ 최댓값 빼기: 수식 ⑥에서 증명했듯 결과는 그대로,
    #   대신 exp가 절대 넘치지 않아 안전해집니다
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum()          # 정규화 → 합이 1

def cross_entropy(y, y_hat, eps=1e-15):
    """
    L = -Σ y_i · log(ŷ_i)  — 정답과 예측의 차이를 재는 손실

    비유:
    정답에 몇 %를 줬는지 보고 채점하는 채점관입니다.
    확신하고 틀리면 벌점이 폭발합니다.
    """
    return -np.sum(y * np.log(y_hat + eps))   # eps: log(0) 방지

def softmax_ce_grad(y, y_hat):
    """수식 ⑧에서 유도한 기울기: dL/dz = ŷ - y  (단 한 줄!)"""
    return y_hat - y

# ══════ ① 기본 예측 ══════
z = np.array([3.5, 1.2, 0.8])   # 강아지, 고양이, 토끼의 원시 점수
y = np.array([1.0, 0.0, 0.0])   # 정답: 강아지 (원-핫)

y_hat = softmax(z)
print("예측 확률:", np.round(y_hat, 4), "| 합계:", round(y_hat.sum(), 6))
print(f"손실 L = {cross_entropy(y, y_hat):.4f}")

# ══════ ② 'max 빼기'가 정말 안전한지 확인 ══════
print("\n--- 오버플로 테스트 (z가 아주 클 때) ---")
big = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(over='ignore', invalid='ignore'):
    naive = np.exp(big) / np.exp(big).sum()   # max를 안 뺀 위험한 버전
print("  max 안 뺀 식:", naive, "← 폭발! 💥")
print("  안전한 softmax:", np.round(softmax(big), 6))

# ══════ ③ 확신 정도에 따른 손실 변화 ══════
print("\n--- 정답에 준 확률 vs 손실 ---")
for q in [0.99, 0.9, 0.5, 0.1, 0.01]:
    print(f"  정답 확률 {q:>5.0%} → 손실 {-np.log(q):.4f}")

# ══════ ④ ★ 유도한 기울기를 수치미분으로 검증 (00-04 방식) ★ ══════
def loss_of(zz):
    """z를 받아 손실을 돌려주는 함수 (수치미분용)"""
    return cross_entropy(y, softmax(zz))

h = 1e-5
numeric_grad = np.zeros_like(z)
for i in range(z.size):        # 00-04의 numerical_gradient와 동일한 구조
    tmp = z[i]
    z[i] = tmp + h; f_plus = loss_of(z)
    z[i] = tmp - h; f_minus = loss_of(z)
    numeric_grad[i] = (f_plus - f_minus) / (2 * h)
    z[i] = tmp                   # ★ 원상복구 필수!

print("\n--- 기울기 검증: dL/dz = ŷ - y ? ---")
print("  수치미분 :", np.round(numeric_grad, 6))
print("  공식 ŷ-y :", np.round(softmax_ce_grad(y, y_hat), 6))
print("  기울기 합:", round(float(numeric_grad.sum()), 10), "← 0이어야 정상")


#======================================================================
# [3] 🔴 심화 — 나이브 베이즈 스팸 필터
#     단어별 우도비를 로그 오즈에 더해 스팸 확률을 계산합니다.
#     ⚠️ 책 본문에 실린 실행 결과(무료 당첨 → 99.4%, 회의 보고서 → 0.9%)는 이 코드를 실제로 실행한 값(98.8%, 0.8%)과 다릅니다. 사전 오즈 0.3/0.7 × 10 × 20 ≈ 85.7 → 확률 ≈ 98.8% 이므로 코드의 계산이 맞습니다.
#======================================================================
print("\n" + "=" * 60)
print("[3] 🔴 심화 — 나이브 베이즈 스팸 필터")
print("=" * 60)
import math

# 단어별 우도비 LR = P(단어|스팸) / P(단어|정상)
#   LR > 1 이면 스팸 쪽 증거, LR < 1 이면 정상 쪽 증거
LIKELIHOOD_RATIO = {
    "무료": 10.0,   # 스팸에 10배 자주 등장
    "당첨": 20.0,   # 강력한 스팸 신호
    "회의": 0.1,    # 정상 메일에 10배 자주 등장
    "보고서": 0.2,
}

def spam_probability(words, prior=0.3):
    """
    나이브 베이즈로 스팸 확률을 구합니다.
        사후 오즈 = 사전 오즈 × ∏(우도비)

    비유:
    우체국 분류원이 단어를 하나씩 보며
    저울추를 스팸/정상 쪽으로 옮기는 것과 같습니다.
    """
    # 로그로 계산 (곱셈 → 덧셈, 언더플로 방지)
    log_odds = math.log(prior / (1 - prior))      # 사전 로그 오즈

    for w in words:
        if w in LIKELIHOOD_RATIO:
            log_odds += math.log(LIKELIHOOD_RATIO[w])   # 증거 누적!

    odds = math.exp(log_odds)                # 로그 오즈 → 오즈
    return odds / (1 + odds)                  # 오즈 → 확률

for mail in [["무료", "당첨"], ["회의", "보고서"], ["무료", "회의"]]:
    print(f"{' '.join(mail):<12} → 스팸 확률 {spam_probability(mail):.1%}")
