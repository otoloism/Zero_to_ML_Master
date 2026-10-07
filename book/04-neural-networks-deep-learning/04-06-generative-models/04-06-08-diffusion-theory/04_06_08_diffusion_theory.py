# -*- coding: utf-8 -*-
"""
04-06-08 확산 모델이론 —VAE에서 Diffusion으로

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-06장 이미지 생성 모델의 원리/04-06-08 확산 모델이론 —VAE에서 Diffusion으로.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-06-08-02 2단계 — 순방향 과정: 이미지를 노이즈로
import numpy as np

def make_noise_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    """
    선형 노이즈 스케줄을 생성합니다.

    비유:
    T개의 물통에 순서대로 조금씩 늘어나는 양의 잉크를 담아두는 것과 같습니다.
    """
    betas = np.linspace(beta_start, beta_end, T)      # β_1 ~ β_T
    alphas = 1.0 - betas                                # α_t = 1 - β_t
    alpha_bars = np.cumprod(alphas)                     # ᾱ_t = α_1 * α_2 * ... * α_t (누적곱)
    return betas, alphas, alpha_bars

def forward_diffusion_sample(x0, t, alpha_bars):
    """
    x0에서 x_t로 '한 번에' 점프합니다. (폐형식 공식 사용)

    x_t = sqrt(ᾱ_t) * x0 + sqrt(1 - ᾱ_t) * eps
    """
    eps = np.random.randn(*x0.shape)                    # ε ~ N(0, I)  ─ 재매개변수화 트릭 재사용
    sqrt_ab   = np.sqrt(alpha_bars[t])                   # sqrt(ᾱ_t)
    sqrt_1mab = np.sqrt(1 - alpha_bars[t])               # sqrt(1-ᾱ_t)
    x_t = sqrt_ab * x0 + sqrt_1mab * eps
    return x_t, eps   # eps도 함께 반환 (학습 시 정답 라벨로 사용됨! → 3단계에서 등장)


# %% [Block 2] 04-06-08-05 5단계 — DDPM 학습 루프 구현
import torch
import torch.nn as nn

def train_step(model, x0, alpha_bars, T, optimizer):
    """
    DDPM 논문 Algorithm 1: 학습 한 스텝

    비유:
    범인(노이즈)이 남긴 흔적을 보고, 신경망 탐정이
    "범인의 모습(노이즈 벡터)"을 그려내는 훈련을 시킵니다.
    """
    batch_size = x0.shape[0]

    # ① 매 스텝마다 무작위 시점 t를 뽑음 (0 ~ T-1)
    t = torch.randint(0, T, (batch_size,))

    # ② 정답 노이즈 ε를 샘플링
    eps = torch.randn_like(x0)

    # ③ 폐형식 공식으로 x_t를 '한 번에' 생성 (2단계 참고)
    sqrt_ab   = alpha_bars[t].sqrt().view(-1, 1, 1, 1)
    sqrt_1mab = (1 - alpha_bars[t]).sqrt().view(-1, 1, 1, 1)
    x_t = sqrt_ab * x0 + sqrt_1mab * eps

    # ④ 신경망이 노이즈를 예측
    eps_pred = model(x_t, t)

    # ⑤ 단순화된 손실 (4단계 공식) — 단순 MSE!
    loss = nn.functional.mse_loss(eps_pred, eps)

    optimizer.zero_grad()
    loss.backward()          # 04-01-04의 역전파가 그대로 사용됨
    optimizer.step()
    return loss.item()


# %% [Block 3] 04-06-08-05 5단계 — DDPM 학습 루프 구현
@torch.no_grad()
def sample(model, shape, betas, alphas, alpha_bars, T):
    """
    DDPM 논문 Algorithm 2: 샘플링 (역방향 과정을 실제로 T번 반복)

    학습 때는 x_0->x_t를 '한 번에' 점프했지만(폐형식),
    생성할 때는 반드시 x_T부터 x_0까지 T번을 '하나씩' 걸어야 합니다.
    지름길이 없는 이유: x_0을 모르기 때문입니다 (우리가 만들려는 대상이므로).
    """
    x_t = torch.randn(shape)          # x_T ~ N(0, I) 에서 출발

    for t in reversed(range(T)):
        eps_pred = model(x_t, torch.tensor([t]))   # 신경망이 현재 노이즈를 추정

        alpha_t      = alphas[t]
        alpha_bar_t  = alpha_bars[t]
        beta_t       = betas[t]

        # 예측한 노이즈를 이용해 평균 μ_θ 계산 (3단계 ~ 4단계 관계식)
        mean = (1 / alpha_t.sqrt()) * (x_t - (beta_t / (1 - alpha_bar_t).sqrt()) * eps_pred)

        if t > 0:
            noise = torch.randn_like(x_t)
            x_t = mean + beta_t.sqrt() * noise      # 재매개변수화 트릭 재사용
        else:
            x_t = mean   # 마지막 단계는 노이즈를 더하지 않음

    return x_t   # 최종적으로 x_0 (생성된 이미지)
