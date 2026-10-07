# -*- coding: utf-8 -*-
"""
04-02-08 DQN— 아타리를 정복한 딥 강화학습의 시작

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-02장 강화 학습 알고리즘/04-02-08 DQN— 아타리를 정복한 딥 강화학습의 시작.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-02-08-B 트릭 1: 경험 재생(Experience Replay) — "카드를 섞어서 읽기"
from collections import deque
import random
import numpy as np

class ReplayBuffer:
    """경험을 저장하고 무작위로 꺼내는 버퍼.
    비유: '일기장에 매일 기록하고, 복습할 때는 무작위 페이지를 펼치기'"""

    def __init__(self, buffer_size, batch_size):
        # deque: 크기가 넘으면 가장 오래된 것을 자동으로 버리는 리스트
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """경험 하나를 일기장에 기록"""
        self.buffer.append((state, action, reward, next_state, done))

    def get_batch(self):
        """무작위로 batch_size개의 경험을 꺼냄 — 카드 섞기!"""
        data = random.sample(self.buffer, self.batch_size)
        states = np.stack([x[0] for x in data])
        actions = np.array([x[1] for x in data])
        rewards = np.array([x[2] for x in data])
        next_states = np.stack([x[3] for x in data])
        dones = np.array([x[4] for x in data])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# %% [Block 2] 04-02-08-C 트릭 2: 타깃 신경망(Target Network) — "과녁을 잠시 고정"
import copy

class DQNAgent:
    """DQN = Q 신경망 + 경험 재생 + 타깃 신경망.
    비유: '카드 섞기(경험 재생) + 채점 기준 고정(타깃 신경망)으로 안정 학습'"""

    def __init__(self, action_size):
        self.qnet = QNet(action_size)             # 메인 네트워크 (매 스텝 갱신)
        self.qnet_target = QNet(action_size)      # 타깃 네트워크 (주기적 복사)
        self.optimizer = Adam().setup(self.qnet)
        self.gamma = 0.98

    def sync_qnet(self):
        """메인 → 타깃으로 가중치 복사 (주기적으로 호출)"""
        self.qnet_target = copy.deepcopy(self.qnet)

    def update(self, state, action, reward, next_state, done):
        # ★ 타깃 계산에 qnet_target 사용! (고정된 과녁)
        next_qs = self.qnet_target(next_state)  # ← target 네트워크!
        next_q = next_qs.max(axis=1)
        next_q.unchain()                         # 04-03장: 이 경로로 grad 안 흐르게
        target = reward + (1 - done) * self.gamma * next_q

        # 메인 네트워크의 Q값과 비교
        q = self.qnet(state)[np.arange(len(action)), action]
        loss = mean_squared_error(target, q)     # 04-01-03: MSE

        self.qnet.cleargrads()                    # ③ 기울기 초기화
        loss.backward()                            # ④ 역전파
        self.optimizer.update()                    # ⑤ 파라미터 갱신


# %% [Block 3] 04-02-08-D OpenAI Gym — 표준화된 실험 환경
import gym

# 환경 생성: CartPole (막대 위에 세운 봉 균형 잡기)
env = gym.make('CartPole-v0')

# 환경 초기화 → 초기 상태 받기
state = env.reset()

# 한 스텝 실행: 행동을 주면, (다음상태, 보상, 종료여부, 정보)가 돌아옴
action = env.action_space.sample()  # 무작위 행동
next_state, reward, done, info = env.step(action)

# 이것이 바로 04-02-02에서 정의한 MDP의 (s,a) → (s',r) 그 자체입니다!
print(f"상태: {state.shape}, 행동: {action}, 보상: {reward}, 종료: {done}")
