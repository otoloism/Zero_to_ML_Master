# -*- coding: utf-8 -*-
"""
01-01🤔 왜 수학인가 — 요리를 배울 것인가 라면만 끓이다 갈 것인가!

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 01권  수포자들을 위한 머신러닝 수학 완전기초와 파이썬 맛보기/01-01🤔 왜 수학인가  — 요리를 배울 것인가  라면만 끓이다 갈 것인가!.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 📍 01-01-A · 1단계 — 🍜 왜 수학이 필요한가: 라면과 요리
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()      # 모델 준비
model.fit(X_train, y_train)       # 학습 — 이 한 줄이 전부!
print(model.score(X_test, y_test))  # 정확도 확인


# %% [Block 2] 📍 01-01-D · 4단계 — 🛠️ 부엌 차리기: 환경 준비
import sys
import numpy as np      # 관례적으로 np라는 짧은 별칭을 씁니다

print("파이썬 버전:", sys.version.split()[0])
print("NumPy 버전 :", np.__version__)

# 첫 벡터를 만들어봅니다 (00-01에서 자세히 배웁니다)
v = np.array([3.0, 4.0])

print("\n첫 벡터 :", v)
print("모양     :", v.shape)      # (2,) → 숫자가 2개인 1차원
print("자료형   :", v.dtype)      # float64 (실수)
print("길이(노름):", np.linalg.norm(v))   # √(3²+4²) = 5.0

print("\n✅ 여기까지 출력되면 준비 완료입니다!")


# %% [Block 3] 📍 01-01-E · 5단계 — 🎬 맛보기: 여섯 장을 20줄로
import numpy as np
np.set_printoptions(suppress=True)   # 지수 표기 끄기 (보기 편하게)

# ═══ 🛒 00-01 벡터: 장바구니 총액 = 내적 ═══
# 수식: a·b = Σ aᵢbᵢ
basket = np.array([3, 2, 1])           # 사과3, 바나나2, 우유1
price  = np.array([2000, 1500, 3000])  # 각 단가
print("00-01 총액:", basket @ price, "원")   # @ 가 내적!

# ═══ 📋 00-02 행렬: 행과 열을 짝지어 곱하기 ═══
# 수식: Cᵢⱼ = Σₖ AᵢₖBₖⱼ
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print("00-02 행렬곱:\n", A @ B)          # 같은 @ 기호를 행렬에!

# ═══ 🧾 00-03 최소제곱: 점 3개에 가장 가까운 직선 ═══
# 수식: x̂ = (AᵀA)⁻¹Aᵀb
X = np.array([1., 2., 3.])
Y = np.array([2., 3., 5.])
M = np.column_stack([X, np.ones(3)])       # 1로 채운 열 = 절편용
w = np.linalg.lstsq(M, Y, rcond=None)[0]
print(f"00-03 직선: y = {w[0]:.4f}x + {w[1]:.4f}")

# ═══ 🌫️ 00-04 경사하강법: 골짜기 바닥 찾기 ═══
# 수식: x ← x - η·f'(x)
df = lambda x: 2*x - 4                 # f(x)=x²-4x+5 의 미분
x = 10.0
for _ in range(30):
    x = x - 0.1 * df(x)  # ← 이 한 줄이 AI 학습의 전부!
print(f"00-04 30스텝 후 x = {x:.6f}  (정답 2)")

# ═══ 🏥 00-05 소프트맥스: 점수를 확률로 ═══
# 수식: ŷᵢ = exp(zᵢ) / Σ exp(zⱼ)
z = np.array([3.5, 1.2, 0.8])         # 강아지/고양이/토끼 점수
e = np.exp(z - z.max())                  # max 빼기 = 안전장치
p = e / e.sum()
print("00-05 확률:", p.round(4), "합계:", round(float(p.sum()), 4))

# ═══ 📚 00-06 SVD: 행렬을 세 조각으로 ═══
# 수식: A = UΣVᵀ
D = np.array([[1., 2, 3, 4], [2, 3, 4, 5],
              [3, 4, 5, 6], [4, 5, 6, 7]])
S = np.linalg.svd(D, compute_uv=False)      # 특잇값만 뽑기
print("00-06 특잇값:", S.round(4))
print(f"     첫 조각 하나로 정보의 {S[0]**2/np.sum(S**2):.1%} 를 담는다")
