# -*- coding: utf-8 -*-
"""
04-02-04 동적 프로그래밍 (DP) — 환경 모델을 알 때

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-02장 강화 학습 알고리즘/04-02-04 동적 프로그래밍 (DP) — 환경 모델을 알 때.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-02-04-B 정책 평가 — "이 정책이면 각 칸의 가치는?"
from collections import defaultdict

def policy_eval(env, pi, gamma=0.9, threshold=0.001):
    """정책 π에 대한 V(s)를 벨만 기대 방정식으로 계산.
    비유: '현재 내비 경로대로 가면 각 교차로에서 몇 분 남았는지 계산'"""
    V = defaultdict(lambda: 0)
    while True:
        V_new = defaultdict(lambda: 0)
        for state in env.states():
            if env.is_done(state) or state == env.wall:
                continue
            value = 0
            for action in env.actions():
                ns = env.next_state(state, action)
                r = env.reward(ns)
                value += pi[state][action] * (r + gamma * V[ns])
            V_new[state] = value
        # 수렴 확인
        delta = max(abs(V_new[s] - V[s]) for s in env.states())
        V = V_new
        if delta < threshold:
            break
    return V


# %% [Block 2] 04-02-04-C 정책 개선 — "가치가 높은 방향으로 바꾸기"
def greedy_policy(env, V, gamma=0.9):
    """V(s)를 기반으로 각 상태에서 최선의 행동을 선택하는 정책 생성.
    비유: '각 교차로에서 도착 예상 시간이 가장 짧은 방향 선택'"""
    pi = defaultdict(lambda: {})
    for state in env.states():
        if env.is_done(state) or state == env.wall:
            continue

        # 각 행동의 가치(R + γV') 계산
        action_values = {}
        for action in env.actions():
            ns = env.next_state(state, action)
            r = env.reward(ns)
            action_values[action] = r + gamma * V[ns]

        # 최대 가치 행동에 확률 1, 나머지는 0
        best_action = max(action_values, key=action_values.get)
        for action in env.actions():
            pi[state][action] = 1.0 if action == best_action else 0.0

    return pi


# %% [Block 3] 04-02-04-D 정책 반복 — "평가와 개선을 번갈아 반복"
def policy_iteration(env, gamma=0.9):
    """정책 반복: 평가 ↔ 개선을 반복하여 최적 정책 찾기.
    비유: '내비 재탐색을 반복 → 최적 경로 확정'"""
    # ① 초기 정책: 균일 랜덤 (모든 방향 25%)
    pi = defaultdict(lambda: {})
    for s in env.states():
        for a in env.actions():
            pi[s][a] = 0.25

    iteration = 0
    while True:
        iteration += 1
        # ② 정책 평가: 현재 정책의 V(s) 계산
        V = policy_eval(env, pi, gamma)

        # ③ 정책 개선: V를 기반으로 greedy 정책 생성
        new_pi = greedy_policy(env, V, gamma)

        # ④ 수렴 확인: 정책이 바뀌지 않으면 종료
        changed = False
        for s in env.states():
            if env.is_done(s) or s == env.wall:
                continue
            for a in env.actions():
                if abs(pi[s][a] - new_pi[s][a]) > 1e-6:
                    changed = True

        pi = new_pi
        if not changed:
            break

    print(f"정책 반복 수렴! 반복 횟수: {iteration}")
    return pi, V

# ── 실행 ──
env = GridWorld()
pi_star, V_star = policy_iteration(env, gamma=0.9)

# V*(s) 출력
arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}
print("\n[V*(s)] 최적 가치:")
for h in range(3):
    row = ""
    for w in range(4):
        if (h,w) == (1,1): row += "  [벽]  "
        else: row += f" {V_star[(h,w)]:+.3f} "
    print(row)

# π*(s) 출력
print("\n[π*] 최적 정책 (각 칸의 최선 방향):")
for h in range(3):
    row = ""
    for w in range(4):
        if env.is_done((h,w)): row += "  ★   "
        elif (h,w) == env.wall: row += " [벽]  "
        else:
            best = max(pi_star[(h,w)], key=pi_star[(h,w)].get)
            row += f"  {arrows[best]}   "
    print(row)


# %% [Block 4] 04-02-04-E 가치 반복 — "평가와 개선을 한 줄로 합치기"
def value_iteration(env, gamma=0.9, threshold=0.001):
    """가치 반복: 벨만 최적 방정식을 반복 적용.
    비유: '매 교차로에서 무조건 가장 빠른 길을 선택' — 평가+개선 동시!"""
    V = defaultdict(lambda: 0)
    iteration = 0
    while True:
        V_new = defaultdict(lambda: 0)
        for s in env.states():
            if env.is_done(s) or s == env.wall:
                continue
            # 핵심: Σπ(평균) 대신 max(최대)!
            V_new[s] = max(
                env.reward(env.next_state(s, a)) + gamma * V[env.next_state(s, a)]
                for a in env.actions()
            )
        iteration += 1
        delta = max(abs(V_new[s] - V[s]) for s in env.states())
        V = V_new
        if delta < threshold:
            break
    print(f"가치 반복 수렴! 반복 횟수: {iteration}")
    return V

V_star = value_iteration(env, gamma=0.9)
pi_star = greedy_policy(env, V_star, gamma=0.9)  # V*에서 정책 추출
