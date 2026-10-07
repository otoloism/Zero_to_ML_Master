# -*- coding: utf-8 -*-
"""
04-02-11 오프-정책 MC · n단계 TD · DoubleDQN·정책 경사법증명

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-02장 강화 학습 알고리즘/04-02-11 오프-정책 MC · n단계 TD · DoubleDQN·정책 경사법증명.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-02-11-A 부록 A — 오프-정책 MC와 중요도 샘플링의 분산 문제
import numpy as np

# 보정 비율 ρ = π/b 가 에피소드 길이에 따라 어떻게 폭발하는지
pi_확률 = 0.5   # 목표 정책이 이 행동을 선택할 확률
b_확률  = 0.1   # 행동 정책이 이 행동을 선택할 확률
rho = pi_확률 / b_확률   # 한 스텝의 보정 비율 = 5

print(f"한 스텝 보정 비율 ρ = {rho}")
print()

for 길이 in [1, 5, 10, 20, 50]:
    누적_rho = rho ** 길이   # ρ를 길이만큼 곱함
    print(f"에피소드 길이 {길이:>3} → ρ의 곱 = {누적_rho:.2e}")

print(f"\n→ 50스텝이면 보정 비율이 10^34! 이러면 추정이 완전히 불안정합니다.")
print(f"→ 해결: 가중 중요도 샘플링 (ρ들의 합으로 정규화)")


# %% [Block 2] 04-02-11-B 부록 B — n단계 TD: MC와 TD 사이의 모든 중간 단계
def n_step_td_target(rewards, next_V, gamma, n):
    """n걸음의 실제 보상 + 나머지는 추정치로 대체.
    비유: 'n이닝 결과를 보고, 나머지 이닝은 현재 예측으로 대체'"""
    G = 0
    for i in range(n):
        G += (gamma ** i) * rewards[i]    # n걸음의 실제 보상 합산
    G += (gamma ** n) * next_V            # 나머지는 추정치!
    return G

# 예시: 보상이 [1, 1, 1, 1, 1]이고, V(s')=10, γ=0.9
보상들 = [1, 1, 1, 1, 1]
V_next = 10
gamma = 0.9

for n in [1, 2, 3, 5]:
    목표 = n_step_td_target(보상들[:n], V_next, gamma, n)
    print(f"n={n} → TD 목표 = {목표:.3f}  (실제 보상 {n}걸음 + 추정 {5-n}걸음)")


# %% [Block 3] 04-02-11-C 부록 C — Double DQN: max의 과대평가 편향 해결 — 기존 DQN (04-02-08)
# ── 기존 DQN (04-02-08) ──
# 타깃 네트워크가 행동 선택과 가치 평가를 "둘 다" 수행
next_qs = qnet_target(next_state)
next_q = next_qs.max(axis=1)           # 타깃이 선택 + 평가 (과대평가 위험!)

# ── Double DQN (코드 3줄 수정!) ──
# 메인이 행동 선택, 타깃이 가치 평가 → 과대평가 방지!
next_qs_main = qnet(next_state)         # ★ 메인 네트워크로 행동 선택
best_action = next_qs_main.argmax(axis=1)  # ★ "어떤 행동이 최선?"
next_qs_target = qnet_target(next_state)
next_q = next_qs_target[np.arange(len(best_action)), best_action]  # ★ 타깃이 평가

print("DQN: 같은 네트워크가 선택+평가 → 과대평가 편향")
print("Double DQN: 선택(메인) ≠ 평가(타깃) → 편향 감소!")
print("→ 코드 3줄만 바꿔도 성능이 크게 개선됩니다.")


# %% [Block 4] 04-02-11-D 부록 D — 정책 경사 정리: 로그 미분 트릭의 수학적 유도
import numpy as np

# 로그 미분 트릭의 핵심: ∇f(x) = f(x) · ∇log(f(x))
# 예시: f(x) = x²일 때

def f(x):
    return x ** 2

def f_grad(x):
    """f(x) = x²의 진짜 미분 = 2x"""
    return 2 * x

def log_trick_grad(x):
    """로그 트릭: f(x) · ∇log(f(x)) = x² · (2/x) = 2x"""
    return f(x) * (2 / x)   # f · d/dx[log(x²)] = x² · 2/x

# 두 방법이 같은 결과를 내는지 확인
for x in [1.0, 2.0, 3.0, 5.0]:
    직접 = f_grad(x)
    트릭 = log_trick_grad(x)
    print(f"x={x}  직접 미분: {직접:.1f}  로그 트릭: {트릭:.1f}  같음? {직접 == 트릭}")

print(f"\n→ 로그 트릭의 핵심 가치: 분포 안의 미분을 '샘플링+log미분'으로 바꿀 수 있음!")
print(f"→ 이것이 REINFORCE의 -log(π)×G 식의 수학적 기반입니다.")
