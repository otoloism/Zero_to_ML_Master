# -*- coding: utf-8 -*-
"""
04-02-05 몬테카를로(MC) 방법 — 환경 모델 없이

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-02장 강화 학습 알고리즘/04-02-05 몬테카를로(MC) 방법 — 환경 모델 없이.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 📖 처음부터 끝까지 친절하게 풀어보기 — 1) 딕셔너리(dict): "이름표가 붙은 상자"
# ── 1) 딕셔너리(dict): "이름표가 붙은 상자" ──
# 키(이름)로 값을 저장하고 꺼냅니다. 리스트는 번호(0,1,2...)로 접근하지만,
# 딕셔너리는 원하는 이름으로 접근할 수 있습니다.
점수 = {"수학": 90, "영어": 85, "과학": 95}
print(점수["수학"])           # → 90

# ── 2) .get(키, 기본값): "없으면 기본값 쓰기" ──
# 딕셔너리에 키가 없으면 에러 대신 기본값을 돌려줍니다.
print(점수.get("국어", 0))     # → 0  ("국어"가 없으니 기본값 0)

# ── 3) .setdefault(키, 기본값): "없으면 넣고, 있으면 그대로" ──
점수.setdefault("국어", 70)   # "국어"가 없었으므로 70을 넣음
print(점수["국어"])           # → 70

# ── 4) reversed(): 리스트를 거꾸로 읽기 ──
기록 = [10, 20, 30]
for 값 in reversed(기록):
    print(값, end=" ")        # → 30 20 10 (뒤에서부터!)

# ── 5) 튜플(tuple): 바꿀 수 없는 묶음 ──
# 소괄호로 여러 값을 묶습니다. 리스트와 비슷하지만 수정 불가합니다.
경험 = ("상태A", 10)          # (상태, 보상) 쌍
print(경험[0], 경험[1])      # → 상태A 10


# %% [Block 2] 04-02-05-A 동적 프로그래밍(DP) 복습 — "환경의 설계도를 안다면" — 가치 반복법: 벨만 최적 방정식을 모든 상태에 반복 적용합니다.
# 가치 반복법: 벨만 최적 방정식을 모든 상태에 반복 적용합니다.
# 비유: "내비게이션이 모든 교차로의 '남은 거리'를 동시에 업데이트"

def value_iteration(env, gamma=0.9, threshold=1e-4):
    """벨만 최적 방정식을 반복 적용해서 최적 가치 V*를 구합니다.
    비유: '목적지에서부터 거꾸로, 모든 교차로의 거리를 동시에 업데이트'"""
    V = {s: 0 for s in env.states()}  # 모든 상태의 가치를 0으로 초기화

    while True:
        delta = 0                          # 변화량 추적
        for s in env.states():
            if s == env.goal_state:        # 목표 상태는 건너뜀
                continue
            values = []
            for a in env.actions():
                next_s = env.next_state(s, a)     # P를 알아야 가능! (DP의 전제)
                r = env.reward(s, a, next_s)      # R도 알아야 함
                values.append(r + gamma * V[next_s])  # 벨만 방정식 우변
            new_V = max(values)               # 벨만 "최적" 방정식의 max
            delta = max(delta, abs(new_V - V[s]))
            V[s] = new_V
        if delta < threshold:              # 변화가 충분히 작으면 수렴!
            break
    return V


# %% [Block 3] 04-02-05-B MC의 핵심 — "많이 해볼수록, 평균은 진짜 기댓값에 가까워진다"
import numpy as np

class RandomAgent:
    """무작위로 행동하면서 경험을 모으고, 에피소드가 끝나면 가치를 갱신합니다.
    비유: '여러 번 여행을 다녀와서, 각 도시의 평균 만족도를 기록하는 여행자'"""

    def __init__(self):
        self.gamma = 0.9    # 할인율: 미래 보상을 얼마나 중요시할지 (0~1)
        self.V = {}           # 각 상태의 가치 추정치 저장소
        self.cnts = {}        # 각 상태를 몇 번 방문했는지 카운터
        self.memory = []      # 한 에피소드 동안의 (상태, 보상) 기록

    def add(self, state, reward):
        """한 스텝의 경험을 기억에 추가합니다."""
        self.memory.append((state, reward))

    def reset(self):
        """에피소드가 끝나면 기억을 초기화합니다."""
        self.memory = []

    def eval(self):
        """에피소드 종료 후, 거꾸로 훑으며 G(누적 보상)를 계산하고 V를 갱신합니다.
        비유: '여행이 끝난 뒤, 마지막 도시부터 거꾸로 돌아보며 만족도 집계'"""
        G = 0                 # 누적 보상 (04-02-03 복습: 뒤에서부터 계산)

        # 에피소드를 거꾸로 훑습니다 (목표→시작 방향)
        for state, reward in reversed(self.memory):
            G = reward + self.gamma * G      # 04-02-03: G_t = r_t + γ·G_{t+1}

            # 방문 횟수 카운터 증가
            self.cnts[state] = self.cnts.get(state, 0) + 1
            self.V.setdefault(state, 0)

            # 점진적 평균 갱신: V ← V + (G - V) / n
            # 04-02-01 복습: Q_{n+1} = Q_n + (새값 - Q_n) / n
            self.V[state] += (G - self.V[state]) / self.cnts[state]


# %% [Block 4] 04-02-05-B MC의 핵심 — "많이 해볼수록, 평균은 진짜 기댓값에 가까워진다"
import numpy as np

# 주사위의 진짜 기댓값: (1+2+3+4+5+6)/6 = 3.5
진짜_기댓값 = 3.5

# 주사위를 N번 던져서, 평균이 진짜 기댓값에 수렴하는 과정을 봅시다
np.random.seed(42)   # 결과 재현을 위한 시드 고정

for N in [10, 100, 1000, 10000, 100000]:
    주사위_결과 = np.random.randint(1, 7, size=N)  # 1~6 랜덤
    평균 = 주사위_결과.mean()
    오차 = abs(평균 - 진짜_기댓값)
    print(f"{N:>7,}번 던짐 → 평균: {평균:.4f}, 오차: {오차:.4f}")


# %% [Block 5] 04-02-05-C 온-정책(On-policy) — "내가 직접 하면서 배운다"
import numpy as np

def epsilon_greedy(Q, state, actions, epsilon=0.1):
    """ε 확률로 랜덤, (1-ε) 확률로 최선의 행동을 선택합니다.
    비유: '맛집 탐색 — 90%는 단골집, 10%는 새 식당 도전'"""

    if np.random.rand() < epsilon:
        # 탐험: 무작위로 행동 선택 (새 식당 도전!)
        return np.random.choice(actions)
    else:
        # 활용: Q값이 가장 큰 행동 선택 (단골집!)
        qs = [Q.get((state, a), 0) for a in actions]
        return actions[np.argmax(qs)]

# 테스트: 10번 선택해보기
Q_예시 = {("A", "상"): 5, ("A", "하"): 2, ("A", "좌"): 1, ("A", "우"): 3}
행동들 = ["상", "하", "좌", "우"]

np.random.seed(0)
for i in range(10):
    선택 = epsilon_greedy(Q_예시, "A", 행동들, epsilon=0.3)
    print(f"{i+1}번째 선택: {선택}", end="  ")


# %% [Block 6] 04-02-05-D 오프-정책과 중요도 샘플링 — "다른 분포에서 뽑은 샘플을 보정하기"
import numpy as np

# ── 상황: 주사위 기댓값을 구하고 싶은데, 조작된 주사위밖에 없다면? ──
# 공정한 주사위(목표 π): 각 면 1/6 확률
pi = np.array([1/6] * 6)

# 조작된 주사위(행동 b): 6이 더 자주 나옴
b = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])

# 조작된 주사위로 10000번 던지기
np.random.seed(42)
faces = np.arange(1, 7)            # [1, 2, 3, 4, 5, 6]
samples = np.random.choice(faces, size=10000, p=b)

# 보정 없이 단순 평균 → 6이 많이 나와서 편향됨!
단순_평균 = samples.mean()

# 중요도 샘플링으로 보정 → 보정 비율 = π(x)/b(x)
보정_비율 = pi[samples - 1] / b[samples - 1]  # 각 샘플의 보정 가중치
보정_평균 = (samples * 보정_비율).mean()

print(f"진짜 기댓값:        3.5000")
print(f"보정 없이 (편향):   {단순_평균:.4f}")
print(f"중요도 샘플링 보정: {보정_평균:.4f}")


# %% [Block 7] 04-02-05-E MC의 한계 — "에피소드가 끝나야 학습할 수 있다" — MC와 TD의 학습 타이밍 차이를 코드로 비교합니다
# MC와 TD의 학습 타이밍 차이를 코드로 비교합니다

에피소드_길이 = 1000  # 한 에피소드가 1000스텝이라고 가정

# ── MC: 에피소드가 끝나야 학습 ──
mc_학습_횟수 = 1       # 1000스텝 후에 딱 1번 학습
print(f"MC: {에피소드_길이}스텝 동안 학습 횟수 = {mc_학습_횟수}번")
print(f"     → 1000스텝을 기다려야 비로소 V를 갱신!")

# ── TD: 매 스텝마다 학습 ──
td_학습_횟수 = 에피소드_길이  # 매 스텝마다 학습!
print(f"\nTD: {에피소드_길이}스텝 동안 학습 횟수 = {td_학습_횟수}번")
print(f"     → 매 스텝마다 V를 바로 갱신! (다음 장에서 배웁니다)")

print(f"\nTD는 MC보다 {td_학습_횟수 // mc_학습_횟수}배 더 자주 학습합니다!")
