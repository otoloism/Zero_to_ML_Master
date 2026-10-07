# -*- coding: utf-8 -*-
"""
04-02-02 마르코프 결정 과정(MDP) — 강화학습의 수학적 틀

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-02장 강화 학습 알고리즘/04-02-02 마르코프 결정 과정(MDP) — 강화학습의 수학적 틀.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-02-02-D 그리드월드 만들기 — "4×4 격자 보드게임"
import numpy as np

class GridWorld:
    """4×4 격자 보드게임 환경.
    비유: 보물찾기 보드게임 — 보물(+1)을 찾되, 함정(-1)은 피하세요!"""

    def __init__(self):
        # 행동 정의: 0=상, 1=하, 2=좌, 3=우
        self.action_space = [0, 1, 2, 3]
        self.action_meaning = {
            0: "↑ 상",
            1: "↓ 하",
            2: "← 좌",
            3: "→ 우",
        }

        # 격자 크기
        self.height = 3                      # 행 수 (0, 1, 2)
        self.width = 4                       # 열 수 (0, 1, 2, 3)

        # 특수 칸 정의
        self.wall = (1, 1)                  # 벽: 지나갈 수 없음
        self.goal = (2, 3)                  # 골인: 보상 +1, 종료
        self.trap = (1, 3)                  # 함정: 보상 -1, 종료

        # 보상 테이블 (대부분 0, 특수 칸만 설정)
        self.reward_map = {
            self.goal: 1.0,                # 골인 보상
            self.trap: -1.0,               # 함정 페널티
        }

        # 시작 위치
        self.start = (0, 0)
        self.agent_state = self.start      # 현재 에이전트 위치

    def states(self):
        """모든 상태(칸)의 목록을 반환합니다.
        비유: 보드게임의 모든 칸 목록."""
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)               # (행, 열) 튜플 반환

    def actions(self):
        """가능한 행동 목록 반환."""
        return self.action_space

    def next_state(self, state, action):
        """현재 상태에서 행동을 하면 다음 상태는?
        비유: '주사위를 던져서 말을 옮겼을 때 도착하는 칸'"""
        # 방향에 따른 이동량: (행 변화, 열 변화)
        move = {
            0: (-1, 0),   # 상: 행 -1
            1: (1, 0),    # 하: 행 +1
            2: (0, -1),   # 좌: 열 -1
            3: (0, 1),    # 우: 열 +1
        }
        dh, dw = move[action]
        next_h = state[0] + dh
        next_w = state[1] + dw
        next_s = (next_h, next_w)

        # 범위 밖이면 제자리 / 벽이면 제자리
        if next_h < 0 or next_h >= self.height:
            return state               # 위/아래 밖 → 제자리
        if next_w < 0 or next_w >= self.width:
            return state               # 좌/우 밖 → 제자리
        if next_s == self.wall:
            return state               # 벽 → 제자리

        return next_s

    def reward(self, state):
        """해당 칸에 도착했을 때의 보상."""
        return self.reward_map.get(state, 0)  # 특수 칸 아니면 0

    def is_done(self, state):
        """종료 상태인지 확인합니다."""
        return state == self.goal or state == self.trap

    def reset(self):
        """에이전트를 시작 위치로 되돌립니다."""
        self.agent_state = self.start
        return self.agent_state

    def step(self, action):
        """한 걸음 이동: (다음 상태, 보상, 종료 여부) 반환.
        비유: 보드게임에서 말을 옮기고, 칸에 적힌 결과를 확인."""
        next_s = self.next_state(self.agent_state, action)
        r = self.reward(next_s)
        done = self.is_done(next_s)
        self.agent_state = next_s          # 상태 갱신
        return next_s, r, done


# %% [Block 2] 04-02-02-D 그리드월드 만들기 — "4×4 격자 보드게임"
env = GridWorld()
state = env.reset()
print(f"시작 위치: {state}")

# 수동으로 경로 이동: 우(3) → 우(3) → 하(1) → 하(1) → 우(3)
path = [3, 3, 1, 1, 3]
for action in path:
    next_s, reward, done = env.step(action)
    print(f"  행동: {env.action_meaning[action]:4s}"
          f" → 이동: {next_s}"
          f" | 보상: {reward:+.1f}"
          f" | 종료: {done}")
    if done:
        print("  🏆 에피소드 종료!")
        break


# %% [Block 3] 04-02-02-E 정책(Policy) π(a|s) — "각 칸에서 어떻게 움직일까?"
from collections import defaultdict

def create_random_policy(env):
    """모든 상태에서 모든 행동을 동일 확률로 선택하는 정책.
    비유: 눈 감고 주사위 던져서 방향을 정하는 것."""
    pi = defaultdict(lambda: {})           # 빈 정책 테이블
    actions = env.actions()
    action_prob = 1.0 / len(actions)       # 1/4 = 0.25

    for state in env.states():
        for action in actions:
            pi[state][action] = action_prob  # 모든 방향 25%

    return pi

# ── 정책 만들고 확인 ──
env = GridWorld()
pi = create_random_policy(env)

# (0,0)에서의 행동 확률 출력
print("(0,0)에서의 정책:")
for action, prob in pi[(0, 0)].items():
    print(f"  {env.action_meaning[action]}: {prob:.2f}")


# %% [Block 4] 04-02-02-F 감가율(γ)과 수익(G) — "내일의 만 원 vs 오늘의 만 원"
def calc_return(rewards, gamma=0.9):
    """보상 리스트를 받아 감가된 수익(Return) G를 계산합니다.
    비유: '미래 용돈을 현재 가치로 환산하기'"""
    G = 0
    for t, r in enumerate(rewards):
        G += (gamma ** t) * r              # γ^t × r_t
    return G

# ── 예제: 5턴 동안 매 턴 보상 1을 받았다면? ──
rewards = [0, 0, 0, 0, 1]  # 4턴 동안 0, 마지막에 +1
G = calc_return(rewards, gamma=0.9)
print(f"보상 리스트: {rewards}")
print(f"γ=0.9일 때 수익 G = {G:.4f}")
print(f"  → 0.9^4 × 1 = {0.9**4:.4f}")

# γ값에 따른 수익 비교
print("\n[감가율별 수익 비교]")
for g in [0.1, 0.5, 0.9, 0.99, 1.0]:
    print(f"  γ={g:.2f} → G = {calc_return(rewards, g):.4f}")


# %% [Block 5] 04-02-02-G MDP 시뮬레이션 — "랜덤 정책으로 보드게임 돌려보기"
import numpy as np

def play_episode(env, pi, gamma=0.9, max_steps=100, verbose=False):
    """정책 pi에 따라 한 에피소드(게임 1판)를 플레이합니다.
    비유: 주사위 던져서 보드게임 한 판 끝까지 진행하기."""
    state = env.reset()                    # ① 시작 위치로
    rewards = []

    for step in range(max_steps):
        # ② 정책에 따라 행동 선택 (확률적)
        action_probs = pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        action = np.random.choice(actions, p=probs)

        # ③ 환경에서 한 발짝 이동
        next_state, reward, done = env.step(action)

        if verbose:
            print(f"  [{step}] {state} → {env.action_meaning[action]}"
                  f" → {next_state} (r={reward:+.1f})")

        rewards.append(reward)
        state = next_state                 # ④ 상태 갱신

        if done:
            break                            # ⑤ 종료 칸 도달 → 끝

    G = calc_return(rewards, gamma)
    return G, rewards

# ── 한 판 플레이 (경로 출력) ──
np.random.seed(42)
env = GridWorld()
pi = create_random_policy(env)

print("=== 에피소드 1판 (랜덤 정책) ===")
G, rewards = play_episode(env, pi, gamma=0.9, verbose=True)
print(f"\n보상 기록: {rewards}")
print(f"수익 G (γ=0.9): {G:.4f}")


# %% [Block 6] 04-02-02-G MDP 시뮬레이션 — "랜덤 정책으로 보드게임 돌려보기"
np.random.seed(0)
num_episodes = 1000
returns = []

for ep in range(num_episodes):
    G, _ = play_episode(env, pi, gamma=0.9)
    returns.append(G)

avg_return = np.mean(returns)
goal_count = sum(1 for g in returns if g > 0)
trap_count = sum(1 for g in returns if g < 0)

print(f"[랜덤 정책] {num_episodes}판 결과:")
print(f"  평균 수익 G: {avg_return:.4f}")
print(f"  골인(+) 횟수: {goal_count}")
print(f"  함정(-) 횟수: {trap_count}")
print(f"  → 랜덤 정책의 한계: 골인보다 함정이 더 많음!")
