# -*- coding: utf-8 -*-
"""
03-02-02 경사하강법의 직관 — 학습률의 영향 — 보폭이 너무 크면 넘어지고, 너무 작으면 하염없는 산행

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-02장 - Parameter Learning/03-02-02 경사하강법의 직관 - 학습률의 영향.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] learning_rate_experiment.py
import numpy as np

# ── 아주 단순한 실험용 함수 ──────────────────────────────
# J(θ) = θ² 이라는 밥그릇 모양 함수를 씁니다.
# 이 함수의 최소점은 누가 봐도 θ = 0 입니다.
# 정답을 아는 상태에서 실험해야 무엇이 잘못됐는지 알 수 있습니다.
def J(theta):
    return theta ** 2

# θ² 을 미분하면 2θ 입니다. (지수 2가 앞으로 내려오고, 지수는 1 줄어듭니다)
def grad_J(theta):
    return 2 * theta

# ── 경사하강법을 n_steps 번 돌려 경로를 기록하는 함수 ──────
def run(lr, theta=4.0, n_steps=8):
    """
    학습률 lr 로 경사하강법을 실행하고 θ 의 이동 경로를 돌려줍니다.

    비유:
    보폭(lr)을 정해 주고 산에서 내려오게 한 뒤,
    발자국을 하나하나 기록해 두는 것과 같습니다.
    """
    path = [theta]                      # 출발 지점부터 기록

    for _ in range(n_steps):
        # 갱신식 그대로: θ := θ − α·(기울기)
        # 빼기(−)라는 점에 주의하세요. 더하면 산 위로 올라갑니다.
        theta = theta - lr * grad_J(theta)
        path.append(theta)   # 새 위치를 목록 끝에 추가

    return path

# ── 세 가지 보폭을 나란히 비교 ──────────────────────────
# 언더스코어(_)는 "값을 쓰지 않을 변수"라는 파이썬 관례입니다.
for lr in [0.01, 0.4, 1.1]:
    path = run(lr)
    # f-string: 문자열 앞에 f를 붙이면 {} 안의 값이 그대로 들어갑니다.
    # :.4f 는 "소수점 넷째 자리까지"라는 서식 지정입니다.
    print(f"lr={lr}")
    print("  ", [f"{p:.4f}" for p in path])


# %% [Block 2] 다. 구현 문제
prev_loss = float('inf')   # 무한대로 시작 (첫 비교는 무조건 통과)

for epoch in range(n_epochs):
    # ... 학습 코드 ...
    loss = compute_cost(theta0, theta1, x, y)

    # J(θ)는 매 iteration마다 감소해야 합니다.
    # 커졌다면 학습률이 너무 크다는 강력한 신호입니다.
    if loss > prev_loss:
        print(f"⚠️ epoch {epoch}: loss 증가! α를 줄이세요 ({prev_loss:.4f} → {loss:.4f})")

    prev_loss = loss   # 다음 비교를 위해 저장
