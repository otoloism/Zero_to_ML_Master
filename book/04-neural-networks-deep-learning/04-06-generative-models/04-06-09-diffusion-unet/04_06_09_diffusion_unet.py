# -*- coding: utf-8 -*-
"""
04-06-09 확산 모델구현 —U-Net과 데이터 생성

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-06장 이미지 생성 모델의 원리/04-06-09 확산 모델구현 —U-Net과 데이터 생성.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 코드 구현 — 순진한 방식(느림)
import torch                     # PyTorch: 동적 계산 그래프 딥러닝 프레임워크

T = 1000                            # 전체 확산 단계 수
betas = torch.linspace(1e-4, 0.02, T)  # β_t 선형 스케줄: 처음엔 아주 조금, 뒤로 갈수록 많이

x = x0.clone()                       # 원본 복사 (원본 훼손 방지)
for t in range(T):                    # 1000번 반복 — 매우 느리다!
    noise = torch.randn_like(x)      # x와 같은 shape의 표준정규 노이즈
    x = torch.sqrt(1 - betas[t]) * x + torch.sqrt(betas[t]) * noise
# 결과: x ≈ N(0, I) — 원본 정보가 거의 다 사라짐


# %% [Block 2] Python — diffusion.py
import torch

T = 1000
betas       = torch.linspace(1e-4, 0.02, T)     # β_t
alphas      = 1.0 - betas                          # α_t = 1 - β_t
alpha_bars  = torch.cumprod(alphas, dim=0)      # ᾱ_t : 누적곱 (cumulative product)

def extract(arr, t, x_shape):
    """배치별로 서로 다른 t에 해당하는 값을 뽑아 브로드캐스팅 형태로 바꾼다.

    비유: 1000칸짜리 사물함(arr)에서 학생마다 자기 번호(t)의 물건을 꺼내
          이미지 모양(B,1,1,1)의 상자에 담아주는 것.
    """
    out = arr.gather(0, t)                       # (B,) 형태로 값 추출
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))  # (B,1,1,1)

def forward_diffusion(x0, t):
    """x0에서 임의 시점 t의 x_t를 '한 번에' 만든다 (폐형식 점프)."""
    eps        = torch.randn_like(x0)               # 정답이 될 노이즈 ε
    sqrt_ab    = extract(alpha_bars.sqrt(), t, x0.shape)
    sqrt_1m_ab = extract((1 - alpha_bars).sqrt(), t, x0.shape)
    xt = sqrt_ab * x0 + sqrt_1m_ab * eps            # 단 한 줄로 x_t 완성!
    return xt, eps                                  # eps는 학습의 정답 라벨


# %% [Block 3] 손실 함수 — 다시 한 번 04-01-03의 MSE
import torch.nn.functional as F

def diffusion_loss(model, x0):
    """확산 모델의 학습 손실 — 단 4줄."""
    B = x0.size(0)
    t = torch.randint(0, T, (B,), device=x0.device)   # ① 무작위 시점 (균등분포 샘플링)
    xt, eps = forward_diffusion(x0, t)              # ② 폐형식 점프 + 정답 노이즈
    eps_pred = model(xt, t)                            # ③ U-Net의 노이즈 예측
    return F.mse_loss(eps_pred, eps)                 # ④ 04-01-03 MSE


# %% [Block 4] 코드 구현
import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    """정수 타임스텝 t를 dim차원 벡터로 변환한다.

    비유:
    숫자 하나(43200초)를 시침·분침·초침 여러 개의 각도로 펼쳐서
    시계처럼 '한눈에 읽히는 형태'로 바꾸는 것.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim                     # 임베딩 차원 (짝수여야 함)

    def forward(self, t):
        """t: (B,) 정수 텐서  →  반환: (B, dim) 실수 텐서"""
        half  = self.dim // 2                                  # sin용 절반 + cos용 절반
        # 주파수를 지수적으로 배치: 1 → 1/10000 까지 부드럽게 감소
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )                                                       # (half,)
        args  = t[:, None].float() * freqs[None, :]           # (B, half) 브로드캐스팅
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


# %% [Block 5] 부품 ① ResidualBlock — 시간 정보를 주입하는 기본 블록
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """합성곱 2번 + 시간 임베딩 주입 + 잔차 연결.

    비유:
    공장 생산 라인의 한 작업대. 재료(x)가 들어오면 두 번 가공하고,
    작업 지시서(t_emb)를 보고 가공 방식을 조절한 뒤,
    원재료를 그대로 옆에 붙여서(잔차) 다음 작업대로 넘긴다.
    """

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)                       # 배치 크기에 둔감한 정규화
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)     # 3x3, 해상도 유지
        self.time  = nn.Linear(time_dim, out_ch)                  # t 임베딩을 채널 수에 맞춤
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # 입출력 채널이 다르면 1x1 합성곱으로 맞춰야 더할 수 있다
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))                    # ① 정규화 → 활성화 → 합성곱
        h = h + self.time(F.silu(t_emb))[:, :, None, None]     # ② 시간 정보 주입 (B,C)→(B,C,1,1)
        h = self.conv2(F.silu(self.norm2(h)))                    # ③ 한 번 더 가공
        return h + self.skip(x)                                 # ④ 잔차 연결


# %% [Block 6] 부품 ② 전체 U-Net 조립
class UNet(nn.Module):
    """MNIST(1x28x28)용 소형 U-Net. 입력과 같은 크기의 노이즈를 예측한다."""

    def __init__(self, ch=64, time_dim=256):
        super().__init__()

        # --- 시간 임베딩 경로 : 정수 t → 256차원 벡터 ---
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(ch),        # (B,) → (B, 64)
            nn.Linear(ch, time_dim),        # (B, 64) → (B, 256)
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # --- 인코더 (내려가는 길) ---
        self.init_conv = nn.Conv2d(1, ch, 3, padding=1)             # 28x28, 1→64
        self.down1 = ResidualBlock(ch, ch, time_dim)                # 28x28, 64
        self.pool1 = nn.Conv2d(ch, ch, 4, stride=2, padding=1)     # 28→14
        self.down2 = ResidualBlock(ch, ch * 2, time_dim)            # 14x14, 128
        self.pool2 = nn.Conv2d(ch * 2, ch * 2, 4, stride=2, padding=1)  # 14→7

        # --- 병목 (골짜기) ---
        self.mid = ResidualBlock(ch * 2, ch * 2, time_dim)         # 7x7, 128

        # --- 디코더 (올라오는 길) ---
        self.up2  = nn.ConvTranspose2d(ch * 2, ch * 2, 4, stride=2, padding=1)  # 7→14
        self.dec2 = ResidualBlock(ch * 4, ch, time_dim)             # concat 후 256→64
        self.up1  = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)          # 14→28
        self.dec1 = ResidualBlock(ch * 2, ch, time_dim)             # concat 후 128→64

        self.out = nn.Conv2d(ch, 1, 1)                              # 64→1, 노이즈 예측

    def forward(self, x, t):
        """x: (B,1,28,28) 노이즈 낀 이미지 / t: (B,) 정수 시점"""
        te = self.time_mlp(t)                    # (B, 256) — 모든 블록이 공유

        h  = self.init_conv(x)                   # (B, 64, 28, 28)
        s1 = self.down1(h, te)                   # (B, 64, 28, 28)  🍞 skip1
        h  = self.pool1(s1)                      # (B, 64, 14, 14)
        s2 = self.down2(h, te)                   # (B,128, 14, 14)  🍞 skip2
        h  = self.pool2(s2)                      # (B,128,  7,  7)

        h  = self.mid(h, te)                     # (B,128,  7,  7) 병목

        h  = self.up2(h)                         # (B,128, 14, 14)
        h  = torch.cat([h, s2], dim=1)          # (B,256, 14, 14) 🍞 회수
        h  = self.dec2(h, te)                    # (B, 64, 14, 14)
        h  = self.up1(h)                         # (B, 64, 28, 28)
        h  = torch.cat([h, s1], dim=1)          # (B,128, 28, 28) 🍞 회수
        h  = self.dec1(h, te)                    # (B, 64, 28, 28)

        return self.out(h)                      # (B,  1, 28, 28) = 예측 노이즈


# %% [Block 7] Python — train.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 데이터: 반드시 [-1, 1] 범위로 정규화! (분산 보존 가정) ---
tf = transforms.Compose([
    transforms.ToTensor(),                      # [0,1] 범위 텐서로
    transforms.Normalize((0.5,), (0.5,)),        # (x-0.5)/0.5 → [-1,1]
])
train_set = datasets.MNIST("./data", train=True, download=True, transform=tf)
loader    = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)

# --- 모델과 옵티마이저 ---
model = UNet().to(device)
opt   = torch.optim.Adam(model.parameters(), lr=2e-4)

# --- 스케줄 상수도 device로 옮겨두기 (자주 빠뜨리는 부분!) ---
betas      = betas.to(device)
alphas     = alphas.to(device)
alpha_bars = alpha_bars.to(device)

# --- 학습 루프 ---
for epoch in range(30):
    model.train()
    total = 0.0
    for x0, _ in loader:          # 라벨(_)은 쓰지 않는다 — 비지도 생성!
        x0 = x0.to(device)
        loss = diffusion_loss(model, x0)

        opt.zero_grad()          # ① 이전 기울기 초기화 (안 하면 누적된다)
        loss.backward()           # ② 역전파
        opt.step()                # ③ 파라미터 갱신
        total += loss.item() * x0.size(0)

    print(f"epoch {epoch:02d}  loss={total / len(train_set):.4f}")


# %% [Block 8] Python — sample.py
@torch.no_grad()                     # 생성에는 기울기가 필요 없다 → 메모리 절약
def sample(model, n=16):
    """순수 노이즈에서 시작해 T번 되돌려 이미지를 만든다."""
    model.eval()                                       # 평가 모드 (dropout/BN 비활성)
    x = torch.randn(n, 1, 28, 28, device=device)      # x_T ~ N(0, I) 눈보라

    for i in reversed(range(T)):                       # t = T-1, T-2, ..., 0
        t = torch.full((n,), i, device=device, dtype=torch.long)
        eps = model(x, t)                              # 이 단계의 노이즈 예측

        a, ab, b = alphas[i], alpha_bars[i], betas[i]
        # 평균 = (x - 노이즈 기여분) / sqrt(α_t)
        mean = (x - (1 - a) / torch.sqrt(1 - ab) * eps) / torch.sqrt(a)

        if i > 0:                                      # 마지막 스텝이 아니면 노이즈 추가
            ab_prev = alpha_bars[i - 1]
            sigma   = torch.sqrt(b * (1 - ab_prev) / (1 - ab))
            x = mean + sigma * torch.randn_like(x)
        else:                                          # 마지막 스텝 — 노이즈 없이 확정
            x = mean

    return x.clamp(-1, 1)                            # [-1,1] 범위로 자르기

imgs = sample(model, n=16)
imgs = (imgs + 1) / 2                                # [-1,1] → [0,1] 시각화용 역정규화


# %% [Block 9] 📖 해설
def extract(arr, t, x_shape):
    out = arr[t]                              # 고급 인덱싱 → (B,)
    return out.view(-1, *([1] * (len(x_shape) - 1)))


# %% [Block 10] 📖 해설
class SelfAttention(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)   # (B, HW, C)
        h, _ = self.attn(h, h, h)                            # 자기 자신에 어텐션
        h = h.transpose(1, 2).view(B, C, H, W)               # 되돌리기
        return x + h                                          # 잔차 연결
