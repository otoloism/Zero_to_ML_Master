# -*- coding: utf-8 -*-
"""
04-02-09 정책 경사법— Policy Gradient와Actor-Critic

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-02장 강화 학습 알고리즘/04-02-09 정책 경사법— Policy Gradient와Actor-Critic.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-02-09-B REINFORCE — "-log(π) × G"가 교차 엔트로피의 변형
class PolicyNet(Model):
    """행동 확률을 출력하는 정책 신경망.
    비유: '각 행동의 선택 확률을 출력하는 직감 시스템'"""
    def __init__(self, action_size):
        super().__init__()
        self.l1 = Linear(128)
        self.l2 = Linear(action_size)

    def forward(self, x):
        x = relu(self.l1(x))
        x = softmax(self.l2(x))   # Softmax → 행동 확률분포!
        return x

def reinforce_update(memory, pi, optimizer, gamma=0.98):
    """REINFORCE: 에피소드 끝나면, 좋은 행동 확률↑ 나쁜 행동 확률↓
    비유: '공연 끝난 후, 관객 반응(G)을 보고 연기 방향을 조정'"""
    G = 0
    pi.cleargrads()                          # ③ 기울기 초기화

    for reward, prob in reversed(memory):    # 04-02-05 MC: 거꾸로 G 누적
        G = reward + gamma * G
        loss = -log(prob) * G                 # 핵심: -log(π(a|s)) × G
        loss.backward()                         # ④ 역전파
    optimizer.update()                          # ⑤ 파라미터 갱신


# %% [Block 2] 04-02-09-C Actor-Critic — "배우(Actor)와 평론가(Critic)" — Actor 업데이트: G 대신 어드밴티지(G - V(s)) 사용
# Actor 업데이트: G 대신 어드밴티지(G - V(s)) 사용
advantage = target - V_w(state)              # 어드밴티지 = G - V(s)
actor_loss = -log(pi(state)[action]) * advantage.unchain()

# Critic 업데이트: V(s)를 target에 가깝게 — 04-01-03 MSE
critic_loss = mean_squared_error(target, V_w(state))

# 두 손실을 합쳐서 역전파
loss = actor_loss + critic_loss
loss.backward()                                # ④ 역전파
