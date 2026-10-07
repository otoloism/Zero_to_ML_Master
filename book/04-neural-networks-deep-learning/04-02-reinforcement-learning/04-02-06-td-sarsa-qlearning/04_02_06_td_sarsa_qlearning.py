# -*- coding: utf-8 -*-
"""
04-02-06 TD법 —SARSA와Q-Learning

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-02장 강화 학습 알고리즘/04-02-06 TD법 —SARSA와Q-Learning.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-02-06-B TD의 통찰 — "한 걸음의 보상 + 다음 상태의 추정치"
class TdAgent:
    """TD법으로 V(s)를 학습하는 에이전트.
    비유: '매 이닝마다 스코어보드를 보고 승리 예측을 업데이트하는 해설자'"""

    def __init__(self):
        self.gamma = 0.9    # 할인율: 미래 보상의 중요도
        self.alpha = 0.01   # 학습률: 한 번에 얼마나 수정할지
        self.V = {}           # 각 상태의 가치 추정치

    def eval(self, state, reward, next_state, done):
        """한 스텝의 경험으로 즉시 V(s)를 갱신합니다.
        비유: '한 이닝 결과를 보고 바로 승리 예측 수정'"""
        # 다음 상태의 가치 (에피소드 끝이면 0)
        next_V = 0 if done else self.V.get(next_state, 0)

        # TD 목표 = 한 걸음 보상 + 할인된 다음 상태 가치
        target = reward + self.gamma * next_V

        # V(s) 초기화 (처음 방문하는 상태면)
        self.V.setdefault(state, 0)

        # TD 갱신: V ← V + α·(목표 - V)  ← 핵심 한 줄!
        self.V[state] += self.alpha * (target - self.V[state])


# %% [Block 2] 04-02-06-C SARSA — "(S, A, R, S', A')의 다섯 글자"
import numpy as np

class SarsaAgent:
    """SARSA: 온-정책 TD 학습.
    비유: '내가 실제로 걸어간 길의 만족도로 지도를 업데이트하는 여행자'"""

    def __init__(self, epsilon=0.1, alpha=0.8, gamma=0.9):
        self.epsilon = epsilon  # 탐험 확률
        self.alpha = alpha      # 학습률
        self.gamma = gamma      # 할인율
        self.Q = {}             # Q(s,a) 테이블

    def get_action(self, state, action_space):
        """ε-greedy로 행동 선택. 04-02-05 복습: 탐험 vs 활용"""
        if np.random.rand() < self.epsilon:
            return np.random.choice(action_space)  # 탐험
        qs = [self.Q.get((state, a), 0) for a in action_space]
        return action_space[np.argmax(qs)]      # 활용

    def update(self, state, action, reward, next_state, next_action, done):
        """SARSA 갱신: Q(s,a) ← Q(s,a) + α·(r + γ·Q(s',a') - Q(s,a))
        핵심: next_action = 실제로 선택한 다음 행동 (온-정책!)"""
        # 다음 상태-행동의 Q값 (에피소드 끝이면 0)
        next_q = 0 if done else self.Q.get((next_state, next_action), 0)

        # TD 목표 = 보상 + 할인된 다음 Q값
        target = reward + self.gamma * next_q

        # Q(s,a) 갱신
        key = (state, action)
        self.Q.setdefault(key, 0)
        self.Q[key] += self.alpha * (target - self.Q[key])  # TD 갱신!


# %% [Block 3] 04-02-06-D Q-Learning — "최선을 다했다고 가정하고 학습"
class QLearningAgent:
    """Q-Learning: 오프-정책 TD 학습.
    비유: '실제로는 안전한 길로 가면서도, 지도에는 최단 경로를 기록하는 탐험가'"""

    def __init__(self, epsilon=0.1, alpha=0.8, gamma=0.9):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = {}
        self.action_space = [0, 1, 2, 3]  # 상, 하, 좌, 우

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        qs = [self.Q.get((state, a), 0) for a in self.action_space]
        return self.action_space[np.argmax(qs)]

    def update(self, state, action, reward, next_state, done):
        """Q-Learning 갱신: Q(s,a) ← Q(s,a) + α·(r + γ·max Q(s',·) - Q(s,a))
        핵심: SARSA와 달리 next_action이 필요 없음! max를 사용!"""

        if done:
            next_q = 0
        else:
            # ★ SARSA와의 유일한 차이: max! (실제 행동이 아닌 최선의 행동)
            next_qs = [self.Q.get((next_state, a), 0) for a in self.action_space]
            next_q = max(next_qs)   # ← 벨만 최적 방정식의 max!

        target = reward + self.gamma * next_q
        key = (state, action)
        self.Q.setdefault(key, 0)
        self.Q[key] += self.alpha * (target - self.Q[key])


# %% [Block 4] 04-02-06-E 클리프 워킹 — SARSA와 Q-Learning의 차이를 눈으로 보기 — SARSA와 Q-Learning의 차이를 한눈에
# SARSA와 Q-Learning의 차이를 한눈에

# 같은 상황: 상태 s에서 행동 a를 해서 보상 r을 받고 다음 상태 s'로 이동
# s'에서 가능한 행동들의 Q값: {상:5, 하:2, 좌:8, 우:3}
# 실제로 ε-greedy가 선택한 다음 행동: "하" (탐험으로 무작위 선택됨)

Q_next = {"상": 5, "하": 2, "좌": 8, "우": 3}
실제_다음_행동 = "하"  # ε-greedy가 무작위로 선택

# SARSA: 실제 다음 행동의 Q값 사용
sarsa_next_q = Q_next[실제_다음_행동]       # Q(s', "하") = 2
print(f"SARSA가 쓰는 next_q: Q(s', '{실제_다음_행동}') = {sarsa_next_q}")

# Q-Learning: 최대 Q값 사용
qlearn_next_q = max(Q_next.values())     # max(5,2,8,3) = 8
최선_행동 = max(Q_next, key=Q_next.get)
print(f"Q-Learning이 쓰는 next_q: max Q(s', ·) = {qlearn_next_q} (행동 '{최선_행동}')")

print(f"\n→ Q-Learning은 실제 행동과 상관없이 '최선'을 기준으로 학습!")
print(f"→ SARSA는 실제 행동을 그대로 반영 (탐험의 위험도 포함)")
