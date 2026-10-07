# -*- coding: utf-8 -*-
"""
04-02-01 밴디트 문제 — 강화학습의 작은 출발점

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-02장 강화 학습 알고리즘/04-02-01 밴디트 문제 — 강화학습의 작은 출발점.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-02-01-A 밴디트 문제란? — "어느 슬롯머신이 최고인지 찾기"
import numpy as np

class Bandit:
    """슬롯머신 한 대를 표현하는 클래스.
    비유: 뷔페의 음식 코너 하나 — 먹어볼 때마다 점수가 조금씩 다릅니다."""

    def __init__(self, rate):
        # rate: 이 슬롯의 '진짜 실력' (평균 보상)
        # 비유: 이 음식 코너의 '기본 맛 점수'
        self.rate = rate

    def play(self):
        """레버를 한 번 당기면 보상을 돌려줍니다.
        비유: 음식을 한 입 먹으면 점수를 받습니다."""
        # np.random.randn(): 평균 0, 표준편차 1인 정규분포에서 난수 1개
        # self.rate + 랜덤노이즈 → 매번 약간 다른 보상
        reward = self.rate + np.random.randn()
        return reward


# %% [Block 2] 04-02-01-C 탐욕 전략 (Greedy) — "최고만 고집하는 전략"
import numpy as np

class Agent:
    """슬롯머신을 고르는 '플레이어' 클래스.
    비유: 뷔페에서 접시를 들고 다니는 손님 — 어떤 코너를 선택할지 결정합니다."""

    def __init__(self, epsilon, action_size=10):
        # epsilon: 탐험할 확률 (0이면 순수 탐욕, 0.1이면 10% 탐험)
        # action_size: 슬롯머신 개수
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)    # 각 슬롯의 추정 가치 (처음엔 0)
        self.ns = np.zeros(action_size)    # 각 슬롯의 시도 횟수 (처음엔 0)

    def update(self, action, reward):
        """시도 후 추정 가치를 업데이트합니다.
        비유: 음식을 먹고 '이 코너 평균 점수' 수첩을 갱신하는 것."""
        self.ns[action] += 1               # 시도 횟수 +1
        # 표본 평균 갱신 공식 (점진적 계산)
        # Q_new = Q_old + (1/n) * (reward - Q_old)
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        """어떤 슬롯을 고를지 결정합니다.
        비유: 뷔페에서 다음에 어떤 코너로 갈지 결정하는 순간."""
        if np.random.random() < self.epsilon:
            # 탐험: 랜덤으로 아무 슬롯이나 선택
            return np.random.randint(0, len(self.Qs))
        else:
            # 활용: 추정 가치가 가장 높은 슬롯 선택
            return np.argmax(self.Qs)


# %% [Block 3] 04-02-01-C 탐욕 전략 (Greedy) — "최고만 고집하는 전략"
import numpy as np

# ── 환경 설정 ──
np.random.seed(42)                        # 재현 가능하도록 시드 고정
arms = 10                                  # 슬롯머신 10대
steps = 1000                                # 총 1,000번 당기기

# 각 슬롯의 '진짜 실력'을 랜덤으로 결정 (표준정규분포)
true_rates = np.random.randn(arms)
print("각 슬롯의 진짜 실력:", np.round(true_rates, 2))
print("최고의 슬롯:", np.argmax(true_rates),
      "(실력:", round(true_rates[np.argmax(true_rates)], 2), ")")

# ── 슬롯머신 생성 ──
bandits = [Bandit(rate) for rate in true_rates]

# ── 순수 탐욕 에이전트 (epsilon=0: 탐험 없음) ──
greedy_agent = Agent(epsilon=0, action_size=arms)

total_reward = 0
for step in range(steps):
    action = greedy_agent.get_action()     # ① 슬롯 선택
    reward = bandits[action].play()       # ② 보상 받기
    greedy_agent.update(action, reward)    # ③ 수첩 갱신
    total_reward += reward

print(f"\n[Greedy] 총 보상: {total_reward:.1f}")
print(f"[Greedy] 가장 많이 선택한 슬롯: {np.argmax(greedy_agent.ns)}")


# %% [Block 4] 04-02-01-D ε-탐욕 전략 (ε-Greedy) — "가끔은 모험하자"
import numpy as np

def run_experiment(epsilon, bandits, steps=1000):
    """에이전트를 만들어 실험을 진행하고, 매 스텝의 보상을 기록합니다.
    비유: 뷔페 손님 한 명이 정해진 전략으로 음식을 고르며 만족도를 기록."""
    agent = Agent(epsilon=epsilon, action_size=len(bandits))
    rewards = []                             # 매 스텝 보상 기록장

    for step in range(steps):
        action = agent.get_action()           # ① 슬롯 선택
        reward = bandits[action].play()      # ② 보상 받기
        agent.update(action, reward)           # ③ 추정 가치 갱신
        rewards.append(reward)

    return rewards, agent

# ── 실험 실행: 3가지 epsilon 비교 ──
np.random.seed(0)
arms = 10
true_rates = np.random.randn(arms)
bandits = [Bandit(r) for r in true_rates]

for eps in [0, 0.1, 0.3]:
    rewards, agent = run_experiment(eps, bandits, steps=1000)
    print(f"ε={eps:.1f} | 총 보상: {sum(rewards):7.1f}"
          f" | 최다 선택 슬롯: {np.argmax(agent.ns)}"
          f" | 진짜 최고 슬롯: {np.argmax(true_rates)}")


# %% [Block 5] 04-02-01-E UCB 전략 — "덜 시도한 슬롯에 보너스를 주자"
import numpy as np

class UCBAgent:
    """UCB(상한 신뢰 구간) 전략으로 슬롯을 고르는 에이전트.
    비유: '덜 가본 식당에 가산점을 주는 맛집 탐험가'"""

    def __init__(self, action_size=10, c=1.0):
        # c: 탐험 강도 (클수록 탐험 많이 함)
        self.c = c
        self.Qs = np.zeros(action_size)        # 추정 가치
        self.ns = np.zeros(action_size)        # 시도 횟수
        self.t = 0                              # 전체 시도 횟수

    def get_action(self):
        """UCB 점수가 가장 높은 슬롯을 선택합니다."""
        self.t += 1

        # 아직 한 번도 시도하지 않은 슬롯이 있으면 우선 시도
        for i in range(len(self.Qs)):
            if self.ns[i] == 0:
                return i                        # 안 가본 식당은 무조건 방문!

        # UCB 점수 계산: Q(a) + c * sqrt(ln(t) / n(a))
        ucb_scores = self.Qs + self.c * np.sqrt(
            np.log(self.t) / self.ns           # 적게 시도 → 보너스 큼
        )
        return np.argmax(ucb_scores)          # 점수 최고인 슬롯 선택

    def update(self, action, reward):
        """시도 후 추정 가치를 업데이트합니다."""
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

# ── UCB 실험 ──
np.random.seed(0)
true_rates = np.random.randn(10)
bandits = [Bandit(r) for r in true_rates]
ucb_agent = UCBAgent(action_size=10, c=1.0)

total_reward = 0
for step in range(1000):
    action = ucb_agent.get_action()
    reward = bandits[action].play()
    ucb_agent.update(action, reward)
    total_reward += reward

print(f"[UCB] 총 보상: {total_reward:.1f}")
print(f"[UCB] 최다 선택 슬롯: {np.argmax(ucb_agent.ns)}")
print(f"[UCB] 진짜 최고 슬롯: {np.argmax(true_rates)}")


# %% [Block 6] 04-02-01-F 비정상 문제 — "맛이 바뀌는 뷔페에서 살아남기"
import numpy as np

class NonStationaryBandit:
    """보상 확률이 매 스텝 조금씩 변하는 슬롯머신.
    비유: 매일 요리사가 살짝 바뀌는 뷔페 코너."""

    def __init__(self, rate):
        self.rate = rate

    def play(self):
        # 매번 rate에 작은 변동을 추가 (비정상성!)
        self.rate += 0.01 * np.random.randn()  # 진짜 실력이 조금씩 변함
        reward = self.rate + np.random.randn()
        return reward

class EMAAgent:
    """지수 이동 평균(EMA)으로 추정 가치를 갱신하는 에이전트.
    비유: '최근 시험 점수를 더 중시하는 학생'"""

    def __init__(self, epsilon, alpha, action_size=10):
        self.epsilon = epsilon
        self.alpha = alpha                       # 학습률 (고정값)
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        """EMA로 추정 가치를 갱신합니다."""
        self.ns[action] += 1
        # 핵심: 1/n 대신 고정 alpha를 사용!
        # → 오래된 보상은 자동으로 잊혀짐
        self.Qs[action] += self.alpha * (reward - self.Qs[action])

    def get_action(self):
        """ε-greedy 정책으로 행동을 선택합니다."""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)

# ── 비정상 환경 비교 실험 ──
np.random.seed(42)
arms = 10
steps = 2000
true_rates = np.random.randn(arms)
bandits = [NonStationaryBandit(r) for r in true_rates]

# 단순 평균 에이전트 vs EMA 에이전트
agent_avg = Agent(epsilon=0.1, action_size=arms)       # 1/n 평균
agent_ema = EMAAgent(epsilon=0.1, alpha=0.1, action_size=arms) # EMA

total_avg, total_ema = 0, 0
for step in range(steps):
    # 두 에이전트가 같은 환경에서 행동
    a1 = agent_avg.get_action()
    r1 = bandits[a1].play()
    agent_avg.update(a1, r1)
    total_avg += r1

    a2 = agent_ema.get_action()
    r2 = bandits[a2].play()
    agent_ema.update(a2, r2)
    total_ema += r2

print(f"[단순 평균] 총 보상: {total_avg:.1f}")
print(f"[EMA α=0.1] 총 보상: {total_ema:.1f}")
