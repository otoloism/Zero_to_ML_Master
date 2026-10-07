# -*- coding: utf-8 -*-
"""
00-05확률·베이즈 정리— 병원 검사의 함정

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 00권  비전공자 처음부터 — 수포자 훑어보기/00-05확률·베이즈 정리— 병원 검사의 함정.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 📍 00-05-A · 1단계 — 확률이란 무엇인가
import numpy as np

def theoretical_prob(sample_space, condition):
    """
    공식 P(A) = n(A) / n(Ω) 를 그대로 계산합니다.

    비유:
    추첨함 안의 공 전체(n(Ω)) 중에서
    원하는 색 공(n(A))이 몇 개인지 세는 것과 같습니다.

    매개변수:
        sample_space: 표본공간 (모든 가능한 결과의 리스트)
        condition   : 원하는 결과인지 판정하는 함수
    """
    favorable = [x for x in sample_space if condition(x)]   # n(A)
    return len(favorable) / len(sample_space)                # n(A)/n(Ω)

def simulate_prob(n_trials, seed=0):
    """주사위를 n_trials 번 던져 '짝수가 나온 비율'을 셉니다."""
    rng = np.random.default_rng(seed)          # 재현 가능한 난수 생성기
    rolls = rng.integers(1, 7, n_trials)      # 1~6 (7은 미포함!)
    return np.mean(rolls % 2 == 0)                # True 비율 = 확률 추정

dice = [1, 2, 3, 4, 5, 6]

# ── 예제 ①: 짝수 ──────────────────────────────────────
print("[짝수]   이론값:", theoretical_prob(dice, lambda x: x % 2 == 0))

# ── 예제 ②: 두 주사위 합이 7 (표본공간 36개) ───────────
two_dice = [(a, b) for a in dice for b in dice]   # 6×6 = 36가지
print("[합=7]   이론값:", round(theoretical_prob(two_dice, lambda p: sum(p) == 7), 4))

# ── 예제 ③: 여사건 — 3번 던져 6이 적어도 한 번 ─────────
p_no_six = (5 / 6) ** 3          # 6이 한 번도 안 나올 확률
print("[6 등장] 이론값:", round(1 - p_no_six, 4), "← 1 - P(여사건)")

# ── 대수의 법칙: 시행 횟수를 늘리면 이론값에 수렴 ───────
print("\n--- 대수의 법칙 (짝수 확률, 이론값 0.5) ---")
for n in [10, 100, 1000, 10000, 100000]:
    est = simulate_prob(n)
    print(f"{n:>7}회 → 추정 {est:.4f}  (오차 {abs(est-0.5):.4f})")


# %% [Block 2] 📍 00-05-B · 2단계 — 조건부 확률: 정보가 확률을 바꾼다
import numpy as np

def conditional_prob(p_joint, p_condition):
    """
    P(A|B) = P(A∩B) / P(B) 를 계산합니다.

    비유:
    조건(B)에 해당하는 사람들만 방에 남기고,
    그 방 안에서만 다시 비율을 재는 것과 같습니다.

    매개변수:
        p_joint    : P(A∩B) — 둘 다 만족할 확률 (분자)
        p_condition: P(B)   — 조건 사건의 확률 (분모)
    """
    if p_condition <= 0:
        raise ValueError("조건 사건의 확률이 0이면 정의되지 않습니다")
    return p_joint / p_condition

# ══════ 방법 1) 공식에 확률을 직접 대입 ══════
p_high  = 400 / 1000          # P(고득점) = 0.4
p_pass  = 200 / 1000          # P(합격)   = 0.2
p_both  = 160 / 1000          # P(고득점 ∩ 합격) = 0.16 (분자는 공통!)

print(f"P(합격|고득점) = {conditional_prob(p_both, p_high):.2f}")   # 분모=0.4
print(f"P(고득점|합격) = {conditional_prob(p_both, p_pass):.2f}")   # 분모=0.2

# ══════ 방법 2) 실제 1000명 데이터에서 마스크로 계산 ══════
# 분할표대로 1000명을 만듭니다: [고득점여부, 합격여부]
high  = np.array([True]*400 + [False]*600)   # 고득점 400명
# 고득점 400명 중 160명 합격 / 저득점 600명 중 40명 합격
passed = np.array([True]*160 + [False]*240
                  + [True]*40  + [False]*560)

# ★ 핵심: high 마스크로 '고득점자만' 골라낸 뒤 그 안에서 평균 = 조건부 확률
print(f"\n[데이터] P(합격|고득점) = {passed[high].mean():.2f}",
      f"(분모 {high.sum()}명)")
print(f"[데이터] P(고득점|합격) = {high[passed].mean():.2f}",
      f"(분모 {passed.sum()}명)")

# ══════ 독립인지 확인 ══════
# 독립이라면 P(A∩B) == P(A)×P(B) 여야 합니다
print(f"\n실제 P(A∩B)   = {p_both:.3f}")
print(f"독립이라면     = {p_high * p_pass:.3f}")
print("→ 다르므로 두 사건은 독립이 아니다 (고득점은 유의미한 정보!)")


# %% [Block 3] 📍 00-05-C · 3단계 — 베이즈 정리: 병원 검사의 함정
def bayes_posterior(prior, likelihood, false_positive_rate):
    """
    P(H|E) = P(E|H)·P(H) / P(E) 를 계산합니다.

    비유:
    탐정이 '원래 의심 정도(prior)'에 '증거의 힘(likelihood)'을 곱한 뒤,
    '증거가 나올 모든 경우(evidence)'로 나눠 갱신된 의심 정도를 구합니다.

    매개변수:
        prior              : P(H)     — 사전 확률(발병률)
        likelihood         : P(E|H)   — 우도(민감도, 검사 정확도)
        false_positive_rate: P(E|H^c) — 오탐률(위양성률)
    반환:
        P(H|E) — 사후 확률
    """
    # 분자: '원인이 참이면서 증거가 나온' 경로 하나
    numerator = likelihood * prior

    # 분모: 전확률 정리 — 증거가 나올 모든 경로의 합
    evidence = numerator + false_positive_rate * (1 - prior)

    return numerator / evidence

def bayes_by_odds(prior, likelihood, false_positive_rate):
    """오즈 형태로 같은 값을 계산합니다: 사후오즈 = 사전오즈 × 우도비"""
    prior_odds = prior / (1 - prior)                    # P(H)/P(H^c)
    likelihood_ratio = likelihood / false_positive_rate    # LR
    posterior_odds = prior_odds * likelihood_ratio      # 곱하기만!
    return posterior_odds / (1 + posterior_odds)       # 오즈 → 확률

# ══════ ① 기본 시나리오 ══════
p = bayes_posterior(prior=0.001, likelihood=0.99, false_positive_rate=0.05)
print(f"양성 판정 후 실제 발병 확률: {p:.4f} ({p:.2%})")
print(f"오즈 형태로 계산해도 동일 : {bayes_by_odds(0.001, 0.99, 0.05):.4f}")

# ══════ ② 인원수로 검증 (공식이 맞는지 확인) ══════
total = 100_000
sick = total * 0.001                    # 환자 100명
healthy = total - sick                    # 비환자 99,900명
true_pos = sick * 0.99                 # 진짜 양성 99명
false_pos = healthy * 0.05            # 가짜 양성 4,995명
print(f"\n[인원수] 진짜양성 {true_pos:.0f}명 / 전체양성 {true_pos+false_pos:.0f}명"
      f" = {true_pos/(true_pos+false_pos):.4f}")

# ══════ ③ 발병률을 바꿔가며 탐색 ══════
print("\n--- 발병률의 영향 (정확도 99%, 오탐 5% 고정) ---")
for rate in [0.001, 0.01, 0.1, 0.5]:
    print(f"  발병률 {rate:>6.1%} → 사후 확률 {bayes_posterior(rate,0.99,0.05):>6.1%}")

# ══════ ④ 오탐률을 바꿔가며 탐색 (더 중요!) ══════
print("\n--- 오탐률의 영향 (발병률 0.1%, 정확도 99% 고정) ---")
for fpr in [0.05, 0.01, 0.001, 0.0001]:
    print(f"  오탐률 {fpr:>7.2%} → 사후 확률 {bayes_posterior(0.001,0.99,fpr):>6.2%}")

# ══════ ⑤ 재검사 — 사후 확률을 다음 사전 확률로! ══════
print("\n--- 같은 검사를 반복하면? (사후 → 다음 사전) ---")
current = 0.001
for n in range(1, 4):
    current = bayes_posterior(current, 0.99, 0.05)   # ★ 갱신된 값을 다시 입력
    print(f"  {n}차 양성 후 → {current:.4f} ({current:.2%})")


# %% [Block 4] 📍 00-05-D · 4단계 — 정규분포: 자연계의 종 모양 곡선
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


# %% [Block 5] 📍 00-05-E · 5단계 — 소프트맥스와 교차 엔트로피: AI는 확률로 대답한다 🔵 중급
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


# %% [Block 6] 🔴 심화 — 나이브 베이즈 스팸 필터: 3단계를 실무로
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
