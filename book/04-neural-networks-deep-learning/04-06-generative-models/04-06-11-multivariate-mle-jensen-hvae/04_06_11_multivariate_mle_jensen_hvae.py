# -*- coding: utf-8 -*-
"""
04-06-11 다변량MLE도출 ·옌센 부등식· 계층형VAE· 수식 기호 목록

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-06장 이미지 생성 모델의 원리/04-06-11 다변량MLE도출 ·옌센 부등식· 계층형VAE· 수식 기호 목록.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 0] [추가] 책에서는 앞 절에서 import 한 모듈 — 단독 실행을 위해 보충 (원문에는 없음)
import torch.nn as nn


# %% [Block 1] 04-06-11-C1 3단계 — 부록 C: 계층형 VAE (HVAE) — VAE와 확산 모델의 다리
class HierarchicalVAE(nn.Module):
    """
    2층 계층형 VAE.

    비유:
    z2(상위 결재) -> z1(중간 결재) -> x(최종 결재)
    각 단계가 이전 단계를 '조건'으로 받아 조금씩 구체화합니다.
    """
    def __init__(self, x_dim, z1_dim, z2_dim):
        super().__init__()
        # 인코더: x -> z1 -> z2  (04-06-07 VAE 인코더를 두 번 쌓음)
        self.enc_z1 = Encoder(x_dim, z1_dim)
        self.enc_z2 = Encoder(z1_dim, z2_dim)
        # 디코더: z2 -> z1 -> x  (조건부로 한 층씩 생성)
        self.dec_z1 = Decoder(z2_dim, z1_dim)
        self.dec_x  = Decoder(z1_dim, x_dim)

    def forward(self, x):
        # 순방향(인코딩): 위로 올라가며 압축
        mu1, logvar1 = self.enc_z1(x)
        z1 = reparameterize(mu1, logvar1)          # 04-06-07 재사용

        mu2, logvar2 = self.enc_z2(z1)
        z2 = reparameterize(mu2, logvar2)

        # 역방향(디코딩): 아래로 내려가며 복원
        z1_recon_mu, z1_recon_logvar = self.dec_z1(z2)
        x_recon_mu = self.dec_x(z1)

        return x_recon_mu, (mu1, logvar1, mu2, logvar2, z1_recon_mu, z1_recon_logvar)
