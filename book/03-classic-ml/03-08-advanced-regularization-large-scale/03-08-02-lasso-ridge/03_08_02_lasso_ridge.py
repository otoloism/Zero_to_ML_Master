# -*- coding: utf-8 -*-
"""
03-08-02 L1 정규화(Lasso)와 L2 정규화(Ridge) (L1 & L2 Regularization)

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-08장 - 정규화 심화 및 대규모 학습/03-08-02 L1 정규화(라쏘)와 L2 정규화(릿지).md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 식 변형 한 줄이 모든 것을 설명한다
import numpy as np

# 가중치 감쇠 계수를 계산해 본다
eta = 0.1      # 학습률 eta
lam = 1.0      # 규제 강도 lambda
n   = 100      # 데이터 개수

decay = 1 - eta * lam / n      # (1 - eta*lambda/n)
print("한 스텝마다 곱해지는 계수 :", decay)
print()

# 기울기가 0이라고 가정하고(=이미 잘 맞히고 있다), 규제만으로 w가 어떻게 줄어드는지
w = 1.0
for step in [0, 100, 500, 1000, 3000]:
    print("  %5d 스텝 후 w = %.5f" % (step, 1.0 * decay ** step))
print()
print("→ 기울기가 0인데도 계속 줄어든다. 하지만 아무리 곱해도 0에는 '닿지 않는다'.")


# %% [Block 2] 5단원 — 숫자로 확인 : 정말 0에 닿는가
import numpy as np

w_start = 1.0        # 출발 가중치
step_size = 0.1      # eta * lambda / n 을 합친 한 스텝의 규제 세기

# ── L2 방식 : w에서 (step_size * w) 만큼 뺀다 → 비례해서 줄어듦
w_l2 = w_start
traj_l2 = [w_l2]
for _ in range(12):
    w_l2 = w_l2 - step_size * w_l2      # 크기에 '비례'하는 감소
    traj_l2.append(w_l2)

# ── L1 방식 : w에서 (step_size * sign(w)) 만큼 뺀다 → 항상 같은 양만큼 줄어듦
w_l1 = w_start
traj_l1 = [w_l1]
for _ in range(12):
    # np.sign : 부호만 돌려준다. 양수면 1, 음수면 -1, 0이면 0
    w_l1 = w_l1 - step_size * np.sign(w_l1)   # 크기와 '무관'한 정액 감소
    traj_l1.append(w_l1)

print("스텝 :", list(range(13)))
print("L2   :", np.round(traj_l2, 4))
print("L1   :", np.round(traj_l1, 4))


# %% [Block 3] 5단원 — 숫자로 확인 : 정말 0에 닿는가 — 소프트 임계(soft thresholding)를 적용한 올바른 L1 갱신
# 소프트 임계(soft thresholding)를 적용한 올바른 L1 갱신
def soft_threshold(w, amount):
    """0을 넘어가려 하면 0에 딱 붙인다."""
    if w > amount:
        return w - amount          # 양수 쪽에서 amount만큼 당김
    elif w < -amount:
        return w + amount          # 음수 쪽에서 amount만큼 당김
    else:
        return 0.0                 # amount보다 작으면 그냥 0으로 확정

w = 1.0
traj = [w]
for _ in range(12):
    w = soft_threshold(w, 0.1)
    traj.append(w)

print("소프트 임계 적용 L1 :", np.round(traj, 4))
print()
print("→ 0에 도착한 뒤 진동하지 않고 0에 머문다. 이것이 특성이 '제거'된 상태다.")


# %% [Block 4] 큰 가중치와 작은 가중치, 누가 먼저 죽는가 — 크기가 다른 가중치 세 개를 같은 규제로 동시에 줄여본다
# 크기가 다른 가중치 세 개를 같은 규제로 동시에 줄여본다
weights = np.array([3.0, 0.5, 0.05])       # 큰 것 / 중간 / 아주 작은 것
names = ["큰 가중치 3.0", "중간 0.5", "작은 0.05"]

print("%-14s %10s %10s %10s %10s" % ("가중치", "시작", "L2 10스텝", "L1 10스텝", "L1 결과"))
print("-" * 60)
for w0, name in zip(weights, names):
    # L2
    w = w0
    for _ in range(10):
        w = w - 0.1 * w
    l2_result = w
    # L1 (소프트 임계)
    w = w0
    for _ in range(10):
        w = soft_threshold(w, 0.1)
    l1_result = w
    verdict = "제거됨 ✂" if l1_result == 0 else "살아남음"
    print("%-14s %10.3f %10.4f %10.4f %10s" % (name, w0, l2_result, l1_result, verdict))


# %% [Block 5] 실습 — 실습 1 정답
# 실습 1 정답
eta, lam, n = 0.1, 5.0, 50
decay = 1 - eta * lam / n
print("감쇠 계수 :", decay)

# 절반이 되는 스텝 수: decay^k = 0.5  →  k = log(0.5) / log(decay)
k = np.log(0.5) / np.log(decay)
print("절반이 되는 스텝 수 : %.1f 스텝" % k)
print("검산 : %d스텝 후 = %.4f" % (round(k), decay ** round(k)))


# %% [Block 6] 실습 — 실습 3 정답 — 10개 가중치를 L1/L2로 각각 50스텝 규제
# 실습 3 정답 — 10개 가중치를 L1/L2로 각각 50스텝 규제
rng = np.random.default_rng(0)
w_init = rng.uniform(-2, 2, 10)      # -2 ~ 2 사이 랜덤 가중치 10개

w1 = w_init.copy()                   # .copy() : 원본을 건드리지 않으려면 반드시 복사
w2 = w_init.copy()
for _ in range(50):
    w1 = np.array([soft_threshold(v, 0.02) for v in w1])   # L1
    w2 = w2 - 0.02 * w2                                     # L2

print("초기 가중치 :", np.round(w_init, 3))
print("L1 50스텝  :", np.round(w1, 3))
print("L2 50스텝  :", np.round(w2, 3))
print()
print("L1에서 정확히 0이 된 개수 :", int((w1 == 0).sum()), "/ 10")
print("L2에서 정확히 0이 된 개수 :", int((w2 == 0).sum()), "/ 10")
print()
print("→ L1은 작은 가중치를 실제로 제거했고, L2는 하나도 제거하지 못했다.")
