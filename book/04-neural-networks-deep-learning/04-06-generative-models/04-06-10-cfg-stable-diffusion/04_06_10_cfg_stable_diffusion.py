# -*- coding: utf-8 -*-
"""
04-06-10 확산 모델응용 — Classifier-Free Guidance와Stable Diffusion

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-06장 이미지 생성 모델의 원리/04-06-10 확산 모델응용 — Classifier-Free Guidance와Stable Diffusion.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 0] [추가] 책에서는 앞 절에서 import 한 모듈 — 단독 실행을 위해 보충 (원문에는 없음)
import torch
import torch.nn as nn
import torch.nn.functional as F


# %% [Block 1] 구현 관점 — MNIST 라벨 조건 (방법 ①)
class ConditionalUNet(nn.Module):
    """04-06-09의 UNet에 클래스 조건 c를 추가한 버전.

    비유:
    같은 요리사(U-Net)에게 '오늘의 주문서'(라벨)를 함께 건네는 것.
    주문서가 비어 있으면(=null) 요리사는 아무거나 만든다.
    """

    def __init__(self, ch=64, time_dim=256, num_classes=10):
        super().__init__()
        # ★변경 1: 클래스 임베딩 테이블. 인덱스 10번은 '조건 없음(∅)' 전용!
        self.label_emb = nn.Embedding(num_classes + 1, time_dim)
        self.null_idx  = num_classes                # ∅ 를 가리키는 인덱스 = 10
        # ... 나머지 층은 04-06-09와 동일 ...

    def forward(self, x, t, y):
        """y: (B,) 클래스 라벨. null_idx이면 무조건부로 동작한다."""
        te = self.time_mlp(t)                    # (B, 256) 시간 정보
        # ★변경 2: 라벨 임베딩을 시간 임베딩에 '더한다'
        te = te + self.label_emb(y)             # (B, 256) 시간 + 조건
        # ★변경 3: 이후 모든 ResidualBlock이 te를 그대로 받는다 (코드 수정 불필요)
        # ... 04-06-09와 완전히 동일한 인코더/병목/디코더 ...
        return self.out(h)


# %% [Block 2] 구현 연결 — 점수를 노이즈 언어로 번역
def classifier_guided_eps(model, classifier, x, t, y, scale=3.0):
    """분류기의 기울기를 빌려와 노이즈 예측을 조건 쪽으로 밀어준다."""
    with torch.enable_grad():                  # 샘플링 중이지만 여기선 기울기가 필요!
        x_in = x.detach().requires_grad_(True)   # x로 미분하겠다고 선언
        logits = classifier(x_in, t)              # 노이즈 낀 이미지를 분류
        logp   = F.log_softmax(logits, dim=-1)
        sel    = logp[range(len(y)), y].sum()      # 정답 클래스의 로그확률
        grad   = torch.autograd.grad(sel, x_in)[0]  # ∇_x log p(c|x)

    eps = model(x, t)                             # 무조건부 노이즈 예측
    return eps - scale * torch.sqrt(1 - alpha_bars[t]) * grad


# %% [Block 3] 구현 관점 ① — 학습 : 라벨 드롭아웃
def cfg_loss(model, x0, y, p_drop=0.1):
    """조건부·무조건부를 하나의 모델로 동시에 학습한다.

    비유:
    요리사에게 주문서를 주되, 10번에 1번은 백지 주문서를 준다.
    그러면 요리사는 '주문대로 만드는 법'과 '알아서 만드는 법'을 둘 다 익힌다.
    """
    B = x0.size(0)
    t = torch.randint(0, T, (B,), device=x0.device)
    xt, eps = forward_diffusion(x0, t)          # 04-06-09와 동일

    # ★ 핵심 한 줄: 10% 확률로 라벨을 ∅(=null_idx)로 교체
    drop = torch.rand(B, device=x0.device) < p_drop
    y_in = torch.where(drop, model.null_idx, y)

    eps_pred = model(xt, t, y_in)
    return F.mse_loss(eps_pred, eps)          # 손실 함수는 04-06-09와 완전히 동일!


# %% [Block 4] 구현 관점 ② — 샘플링 : 배치를 두 배로
@torch.no_grad()
def sample_cfg(model, y, w=3.0, n=16):
    """CFG를 적용해 조건부 샘플링을 수행한다."""
    model.eval()
    x = torch.randn(n, 1, 28, 28, device=device)
    y_null = torch.full_like(y, model.null_idx)      # ∅ 라벨 묶음

    for i in reversed(range(T)):
        t = torch.full((n,), i, device=device, dtype=torch.long)

        # ★ 트릭: 배치를 2배로 쌓아 한 번의 forward로 두 예측을 동시에 얻는다
        x_in = torch.cat([x, x], dim=0)                # (2n, 1, 28, 28)
        t_in = torch.cat([t, t], dim=0)
        y_in = torch.cat([y, y_null], dim=0)           # 앞쪽=조건, 뒤쪽=무조건

        eps_all = model(x_in, t_in, y_in)
        eps_c, eps_u = eps_all.chunk(2, dim=0)         # 다시 둘로 쪼갠다

        # ★ CFG 공식: 차이를 w배 증폭
        eps = eps_c + w * (eps_c - eps_u)

        a, ab, b = alphas[i], alpha_bars[i], betas[i]
        mean = (x - (1 - a) / torch.sqrt(1 - ab) * eps) / torch.sqrt(a)

        if i > 0:
            ab_prev = alpha_bars[i - 1]
            sigma   = torch.sqrt(b * (1 - ab_prev) / (1 - ab))
            x = mean + sigma * torch.randn_like(x)
        else:
            x = mean

    return x.clamp(-1, 1)


# %% [Block 5] 시각화 ③ — 픽셀 확산 vs 잠재 확산 — 학습 전처리: 이미지를 잠재로
# --- 학습 전처리: 이미지를 잠재로 ---
with torch.no_grad():                       # VAE는 얼려둔다 → 기울기 불필요
    latent = vae.encode(image).latent_dist.sample()
    latent = latent * 0.18215              # ★ 스케일 팩터 (7단계 주의사항 참고)

# --- 이제 04-06-09와 완전히 동일한 확산 학습을 latent에 적용 ---
loss = cfg_loss(unet, latent, text_emb)

# --- 생성 후처리: 잠재를 이미지로 ---
with torch.no_grad():
    image = vae.decode(latent / 0.18215).sample   # 나눠서 되돌린 뒤 디코딩


# %% [Block 6] 공식과 변수 정의
class CrossAttention(nn.Module):
    """이미지 특징이 텍스트 토큰을 참조하도록 만드는 층.

    비유:
    도서관 사서. 질문(Q)을 받아 색인(K)과 대조하고,
    가장 관련 있는 책의 내용(V)을 골라 건네준다.
    """

    def __init__(self, img_ch, ctx_dim=768, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = (img_ch // heads) ** -0.5          # 1/sqrt(d)
        self.norm  = nn.GroupNorm(8, img_ch)
        self.to_q  = nn.Linear(img_ch, img_ch, bias=False)   # 이미지 → Q
        self.to_k  = nn.Linear(ctx_dim, img_ch, bias=False)   # 텍스트 → K
        self.to_v  = nn.Linear(ctx_dim, img_ch, bias=False)   # 텍스트 → V
        self.proj  = nn.Linear(img_ch, img_ch)

    def forward(self, x, context):
        """x: (B,C,H,W) 이미지 특징 / context: (B,77,768) 텍스트 임베딩"""
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)   # (B, HW, C) 공간을 시퀀스로

        q = self.to_q(h)             # (B, HW, C)  이미지가 던지는 질문
        k = self.to_k(context)       # (B, 77, C)  단어들의 색인
        v = self.to_v(context)       # (B, 77, C)  단어들의 내용

        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        # attn: (B, HW, 77) — 각 픽셀이 각 단어를 얼마나 볼지의 확률
        out = self.proj(attn @ v)                     # (B, HW, C)
        out = out.transpose(1, 2).view(B, C, H, W)  # 다시 이미지 모양으로
        return x + out                                # 잔차 연결


# %% [Block 7] 전체 코드 — 20줄로 보는 Stable Diffusion
@torch.no_grad()
def stable_diffusion(prompt, negative="", steps=50, guidance=7.5):
    """텍스트에서 이미지까지, 전 과정의 뼈대."""

    # ① 텍스트를 벡터로 — 조건부와 무조건부 둘 다 준비 (CFG용)
    cond   = text_encoder(tokenize(prompt))      # (1, 77, 768)
    uncond = text_encoder(tokenize(negative))    # (1, 77, 768) 보통 빈 문자열
    ctx    = torch.cat([uncond, cond])            # (2, 77, 768) 배치로 묶기

    # ② 잠재 공간에서 순수 노이즈로 시작 (512×512가 아니라 64×64!)
    latent = torch.randn(1, 4, 64, 64, device=device)
    scheduler.set_timesteps(steps)               # 1000단계를 50단계로 건너뛰기(DDIM)
    latent = latent * scheduler.init_noise_sigma

    # ③ 역방향 루프 — 04-06-09의 샘플링과 구조가 완전히 같다
    for t in scheduler.timesteps:
        latent_in = torch.cat([latent] * 2)          # CFG: 배치 2배
        latent_in = scheduler.scale_model_input(latent_in, t)

        eps_all = unet(latent_in, t, encoder_hidden_states=ctx).sample
        eps_u, eps_c = eps_all.chunk(2)

        # ④ CFG — diffusers 표기: guidance = w + 1
        eps = eps_u + guidance * (eps_c - eps_u)

        latent = scheduler.step(eps, t, latent).prev_sample

    # ⑤ 잠재를 진짜 이미지로 — 딱 한 번만 디코딩
    image = vae.decode(latent / 0.18215).sample
    return ((image + 1) / 2).clamp(0, 1)          # [-1,1] → [0,1]


# %% [Block 8] 📖 해설
eps_c = model(x, t, y)                    # 조건부
eps_u = model(x, t, y_null)               # 무조건부
eps   = eps_c + w * (eps_c - eps_u)
