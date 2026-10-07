# -*- coding: utf-8 -*-
"""
04-06-07 변이형 오토인코더 (VAE) — 신경망 기반 생성 모델

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-06장 이미지 생성 모델의 원리/04-06-07 변이형 오토인코더 (VAE) — 신경망 기반 생성 모델.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 구현 관점
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    변이형 오토인코더.

    비유:
    화가(인코더)가 그림을 짧은 메모(z)로 요약하고,
    복원가(디코더)가 그 메모만 보고 그림을 다시 그린다.
    """
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        # --- 인코더: x -> (mu, logvar) ---
        self.enc_fc1 = nn.Linear(input_dim, hidden_dim)   # 입력을 은닉층으로 압축
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)   # 평균 mu 출력
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)  # 로그분산 출력 (분산 대신 log를 써서 항상 양수 보장)

        # --- 디코더: z -> x_hat ---
        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.enc_fc1(x))       # 2단계: 은닉 표현 추출
        mu = self.enc_mu(h)               # 2단계: 평균 mu_phi(x)
        logvar = self.enc_logvar(h)       # 2단계: 로그분산 log(sigma_phi^2(x))
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # 4단계: z = mu + sigma * eps  (재매개변수화 트릭)
        std = torch.exp(0.5 * logvar)     # logvar -> sigma = exp(logvar/2)
        eps = torch.randn_like(std)       # eps ~ N(0, I), 그래프 바깥의 상수 취급
        return mu + std * eps             # 미분 가능한 z

    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        x_hat = torch.sigmoid(self.dec_out(h))  # 픽셀값 [0,1] 범위로 복원
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

def vae_loss(x_hat, x, mu, logvar):
    """
    ELBO에 마이너스를 붙인 손실 = 복원 손실 + KL 손실.
    """
    # 복원 손실: -E[log p(x|z)], 픽셀을 베르누이 분포로 보고 BCE 사용
    recon_loss = F.binary_cross_entropy(x_hat, x, reduction="sum")

    # KL 손실: D_KL(q(z|x) || N(0,I))의 닫힌 형태 공식 (3단계에서 유도)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss, recon_loss, kl_loss

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 04-03-06의 Adam

for epoch in range(20):
    total_loss = 0.0
    for x, _ in train_loader:                   # 04-06-06절에서 만든 DataLoader
        x = x.view(x.size(0), -1).to(device)     # 이미지를 1차원 벡터로 펼침

        x_hat, mu, logvar = model(x)
        loss, recon, kl = vae_loss(x_hat, x, mu, logvar)

        optimizer.zero_grad()   # 04-03장: model.cleargrads()
        loss.backward()         # 04-03-01: y.backward()
        optimizer.step()        # 04-03장: optimizer.update()

        total_loss += loss.item()
    print(f"epoch {epoch+1}: loss={total_loss/len(train_loader.dataset):.4f}")


# %% [Block 2] 핵심 정의
model.eval()
with torch.no_grad():
    # 1) 무작위 생성: 사전분포에서 직접 샘플링
    z = torch.randn(16, 20).to(device)          # N(0, I)에서 16개 샘플
    samples = model.decode(z)                    # 디코더만 사용해 새 이미지 생성

    # 2) 잠재 공간 선형 보간: z1과 z2 사이를 부드럽게 이동
    z1, z2 = torch.randn(1, 20).to(device), torch.randn(1, 20).to(device)
    alphas = torch.linspace(0, 1, steps=10).view(-1, 1).to(device)
    z_interp = (1 - alphas) * z1 + alphas * z2    # 잠재 공간에서 직선 경로
    interp_images = model.decode(z_interp)
