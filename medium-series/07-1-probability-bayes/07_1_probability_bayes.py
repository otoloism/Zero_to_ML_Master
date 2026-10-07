# -*- coding: utf-8 -*-
"""
Zero to 머신러닝 딥러닝 Master — Medium 시리즈 #07 · 1부
베이즈 정리 쉽게 이해하기 — 병원 검사의 함정 (1부: 확률·조건부 확률)

원문(책): 00-05 「확률·베이즈 정리 — 병원 검사의 함정 (1부)」 https://wikidocs.net/439838
Medium : TODO (게시 후 추가)
책     : https://wikidocs.net/book/21464
홈페이지: https://mldict.net
블로그 : https://wikidocs.net/blog/@mldict/

실행: pip install -r requirements.txt && python 07_1_probability_bayes.py
※ 아래 코드는 책 본문의 코드 블록을 순서대로 옮긴 것입니다(내용 변경 없음).
"""

#======================================================================
# [1] 00-05-A · 1단계 — 확률이란? (이론값과 대수의 법칙)
#     주사위 표본공간으로 이론 확률을 계산하고, 시행 횟수를 늘리며 시뮬레이션이 0.5에 수렴하는지 봅니다 (seed 고정).
#======================================================================
print("\n" + "=" * 60)
print("[1] 00-05-A · 1단계 — 확률이란? (이론값과 대수의 법칙)")
print("=" * 60)
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


#======================================================================
# [2] 00-05-B · 2단계 — 조건부 확률과 독립
#     공식 P(A|B)=P(A∩B)/P(B) 와 1000명 데이터의 마스크 계산을 비교하고, 독립 여부를 확인합니다.
#======================================================================
print("\n" + "=" * 60)
print("[2] 00-05-B · 2단계 — 조건부 확률과 독립")
print("=" * 60)
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


#======================================================================
# [3] 00-05-C · 3단계 — 베이즈 정리: 병원 검사의 함정
#     사후 확률을 공식·오즈·인원수 세 방법으로 계산하고, 발병률·오탐률·재검사의 영향을 탐색합니다.
#======================================================================
print("\n" + "=" * 60)
print("[3] 00-05-C · 3단계 — 베이즈 정리: 병원 검사의 함정")
print("=" * 60)
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
