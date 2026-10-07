# -*- coding: utf-8 -*-
"""
04-02-03 벨만 방정식— 가치 함수의 재귀 구조

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-02장 강화 학습 알고리즘/04-02-03 벨만 방정식— 가치 함수의 재귀 구조.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-02-03-B 상태 가치 Vπ(s) — "이 칸에서 시작하면 앞으로 얼마를 벌까?"
import numpy as np
from collections import defaultdict

# 04-02-02의 GridWorld, create_random_policy, play_episode를 사용합니다

def estimate_V_montecarlo(env, pi, gamma=0.9, num_episodes=5000):
    """몬테카를로 방식으로 V(s) 추정: 시뮬레이션 평균.
    비유: '각 칸에서 수천 번 게임을 해보고 평균 점수를 매기기'"""
    V = defaultdict(lambda: 0)           # 각 상태의 가치 추정치
    count = defaultdict(lambda: 0)       # 각 상태의 방문 횟수

    for ep in range(num_episodes):
        # 에피소드 실행: 상태-보상 기록
        state = env.reset()
        episode = []                      # [(상태, 보상), ...]

        for _ in range(100):
            action_probs = pi[state]
            action = np.random.choice(
                list(action_probs.keys()),
                p=list(action_probs.values()))
            next_s, reward, done = env.step(action)
            episode.append((state, reward))
            state = next_s
            if done:
                break

        # 에피소드 끝에서부터 수익 G 역산
        G = 0
        for s, r in reversed(episode):
            G = r + gamma * G             # G = r + γ·G (뒤에서부터!)
            count[s] += 1
            # 증분 평균 갱신 (04-02-01 밴디트와 같은 공식!)
            V[s] += (G - V[s]) / count[s]

    return V

# ── 실행 ──
np.random.seed(42)
env = GridWorld()
pi = create_random_policy(env)
V = estimate_V_montecarlo(env, pi, gamma=0.9)

# 결과 출력
print("[랜덤 정책의 V(s)] 각 칸의 가치:")
for h in range(3):
    row = ""
    for w in range(4):
        if (h, w) == (1, 1):
            row += "  [벽]  "
        else:
            row += f" {V[(h,w)]:+.3f} "
    print(row)


# %% [Block 2] 04-02-03-E 벨만 방정식으로 V(s) 계산하기 — 코드 구현
import numpy as np
from collections import defaultdict

def bellman_one_step(env, pi, V, gamma=0.9):
    """벨만 방정식을 한 번 적용하여 V를 갱신합니다.
    비유: '모든 칸의 부동산 가치를 이웃 칸 기준으로 한 번 재평가'"""
    V_new = defaultdict(lambda: 0)

    for state in env.states():
        # 종료 상태나 벽은 가치 0
        if env.is_done(state) or state == env.wall:
            V_new[state] = 0
            continue

        value = 0
        for action in env.actions():
            # π(a|s): 이 행동을 선택할 확률
            action_prob = pi[state][action]

            # 다음 상태 (결정론적: P=1)
            next_s = env.next_state(state, action)
            reward = env.reward(next_s)

            # 벨만 방정식 핵심!
            # V(s) += π(a|s) × [R + γ × V(s')]
            value += action_prob * (reward + gamma * V[next_s])

        V_new[state] = value

    return V_new


# %% [Block 3] 04-02-03-E 벨만 방정식으로 V(s) 계산하기 — 코드 구현
def iterative_policy_eval(env, pi, gamma=0.9, threshold=0.001):
    """벨만 방정식을 반복 적용하여 V(s)를 수렴시킵니다.
    비유: '부동산 시세를 매일 재평가 → 결국 안정적인 시세에 수렴'"""
    V = defaultdict(lambda: 0)              # 모든 V(s) = 0으로 시작
    iteration = 0

    while True:
        V_new = bellman_one_step(env, pi, V, gamma)
        iteration += 1

        # 수렴 확인: 변화량이 threshold 이하면 종료
        max_delta = 0
        for state in env.states():
            delta = abs(V_new[state] - V[state])
            max_delta = max(max_delta, delta)

        V = V_new

        if max_delta < threshold:
            break

    print(f"수렴 완료! 반복 횟수: {iteration}")
    return V

# ── 실행 ──
env = GridWorld()
pi = create_random_policy(env)
V = iterative_policy_eval(env, pi, gamma=0.9)

print("\n[벨만 방정식] 각 칸의 가치 (랜덤 정책):")
for h in range(3):
    row = ""
    for w in range(4):
        if (h, w) == (1, 1):
            row += "  [벽]  "
        else:
            row += f" {V[(h,w)]:+.3f} "
    print(row)


# %% [Block 4] 04-02-03-F 벨만 최적 방정식 — "최고의 선택을 했을 때의 가치"
def value_iteration(env, gamma=0.9, threshold=0.001):
    """벨만 최적 방정식으로 V*(s)를 구합니다 (가치 반복).
    비유: '각 칸에서 최고의 행동만 골랐을 때의 가치를 반복 계산'"""
    V = defaultdict(lambda: 0)
    iteration = 0

    while True:
        V_new = defaultdict(lambda: 0)
        for state in env.states():
            if env.is_done(state) or state == env.wall:
                V_new[state] = 0
                continue

            # max를 사용! (평균이 아님)
            max_value = float('-inf')
            for action in env.actions():
                next_s = env.next_state(state, action)
                r = env.reward(next_s)
                value = r + gamma * V[next_s]
                max_value = max(max_value, value)  # 핵심: max!

            V_new[state] = max_value
        iteration += 1

        # 수렴 확인
        max_delta = max(abs(V_new[s] - V[s]) for s in env.states())
        V = V_new
        if max_delta < threshold:
            break

    print(f"수렴 완료! 반복 횟수: {iteration}")
    return V

# ── 실행 ──
V_star = value_iteration(env, gamma=0.9)

print("\n[벨만 최적] V*(s) — 최적 정책의 가치:")
for h in range(3):
    row = ""
    for w in range(4):
        if (h, w) == (1, 1):
            row += "  [벽]  "
        else:
            row += f" {V_star[(h,w)]:+.3f} "
    print(row)


# %% [Block 5] 04-02-03-G V*에서 최적 정책 추출 — "가치가 높은 쪽으로 가!"
def extract_policy(env, V, gamma=0.9):
    """V*로부터 최적 정책을 추출합니다.
    비유: '부동산 가치 지도를 보고 가장 비싼 동네 쪽으로 이동!'"""
    arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    print("[최적 정책] 각 칸에서의 최선의 행동:")
    for h in range(3):
        row = ""
        for w in range(4):
            state = (h, w)
            if env.is_done(state):
                row += "  ★   "            # 종료 상태
                continue
            if state == env.wall:
                row += " [벽]  "
                continue

            # 각 행동의 가치를 비교
            best_action = 0
            best_value = float('-inf')
            for a in env.actions():
                ns = env.next_state(state, a)
                v = env.reward(ns) + gamma * V[ns]
                if v > best_value:
                    best_value = v
                    best_action = a

            row += f"  {arrows[best_action]}   "
        print(row)

extract_policy(env, V_star, gamma=0.9)
