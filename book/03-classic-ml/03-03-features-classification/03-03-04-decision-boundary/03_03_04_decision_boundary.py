# -*- coding: utf-8 -*-
"""
03-03-04 결정 경계 — (Decision Boundary) — 모델이 세상에 그은 국경선

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-03장 - Feature 설계와 분류 모델/03-03-04 결정 경계 - 모델이 세상에 그은 국경선.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 구현 원리
import numpy as np
import plotly.graph_objects as go     # 그래프 도구 (matplotlib 대신 Plotly 사용)

def sigmoid(z):
    """앞 절에서 만든 시그모이드 (간단 버전)."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    # np.clip(값, 최소, 최대) : 범위를 벗어나면 잘라낸다
    # → exp 폭발(overflow)을 막는 가장 간단한 안전장치

# ==================================================================
# ① 선형 결정 경계 : θ = [-3, 1, 1]  →  x₁ + x₂ = 3
# ==================================================================
theta_lin = np.array([-3.0, 1.0, 1.0])       # [θ0, θ1, θ2]

# 평면을 격자로 덮는다
# linspace(시작, 끝, 개수) : 구간을 균등하게 나눈 배열
x1_axis = np.linspace(-1, 5, 300)
x2_axis = np.linspace(-1, 5, 300)

# meshgrid : 1차원 축 2개로 2차원 좌표판을 만든다
# 비유) 바둑판의 가로줄·세로줄을 주면 모든 교차점 좌표를 만들어 주는 함수
XX1, XX2 = np.meshgrid(x1_axis, x2_axis)      # 둘 다 (300, 300) 모양

# 격자의 모든 점에서 z = θᵀx 계산 (한꺼번에, 반복문 없이!)
Z_lin = theta_lin[0] + theta_lin[1]*XX1 + theta_lin[2]*XX2
P_lin = sigmoid(Z_lin)                        # 확률로 변환 (색칠용)

# 그림 그리기 : 확률을 배경색으로, 0.5 등고선을 경계선으로
fig = go.Figure()
fig.add_trace(go.Heatmap(                    # Heatmap : 값에 따라 색을 칠하는 그래프
    x=x1_axis, y=x2_axis, z=P_lin,
    colorscale="RdBu", reversescale=True,
    zmin=0, zmax=1, colorbar=dict(title="P(y=1|x)")
))
fig.add_trace(go.Contour(                    # Contour : 등고선
    x=x1_axis, y=x2_axis, z=P_lin,
    contours=dict(start=0.5, end=0.5, size=0.1,   # 0.5 등고선 딱 하나만
                  coloring="none", showlabels=False),
    line=dict(color="#16a34a", width=4),
    showscale=False
))
fig.update_layout(title="선형 결정 경계 : x1 + x2 = 3",
                  xaxis_title="x1", yaxis_title="x2", width=600, height=560)
fig.show()

# ==================================================================
# ② 원형(비선형) 결정 경계 : θ = [-1, 0, 0, 1, 1] → x₁² + x₂² = 1
# ==================================================================
theta_cir = np.array([-1.0, 0.0, 0.0, 1.0, 1.0])  # [θ0, θ1, θ2, θ3, θ4]

g1 = np.linspace(-2, 2, 400)
G1, G2 = np.meshgrid(g1, g1)

# θ0 + θ1·x1 + θ2·x2 + θ3·x1² + θ4·x2²
Z_cir = (theta_cir[0]
         + theta_cir[1]*G1 + theta_cir[2]*G2
         + theta_cir[3]*G1**2 + theta_cir[4]*G2**2)

# 검증 : 원 위의 점 (1,0)과 (0,1)에서 z가 정확히 0이어야 한다
for (a, b) in [(1, 0), (0, 1), (0, 0), (2, 0)]:
    z = -1 + a**2 + b**2
    print(f"점 ({a},{b}) → z = {z:+.1f}, h = {sigmoid(z):.4f}, 예측 y = {int(z >= 0)}")
    # f"..." 는 f-string : 중괄호 안의 값을 문자열에 끼워 넣는 문법
    # {z:+.1f} 는 '부호를 항상 표시하고 소수 1자리'라는 서식 지정


# %% [Block 2] 📝 해설
import numpy as np

def classify_point(theta, x1, x2):
    """
    한 점의 예측 라벨·확률·경계까지의 거리를 함께 알려준다.

    비유:
    국경 검문소에서 "입국 허가 / 확신도 / 국경선에서 몇 km 떨어졌는지"를
    한 번에 찍어 주는 도장.
    """
    theta = np.asarray(theta, dtype=float)

    # z = θ0 + θ1·x1 + θ2·x2
    z = theta[0] + theta[1]*x1 + theta[2]*x2

    # 확률 (안전한 clip 포함)
    proba = 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    # 점과 직선 사이의 부호 있는 거리 공식 : z / ||[θ1, θ2]||
    # norm = 벡터의 길이 (01-03 벡터 장 참조)
    norm = np.linalg.norm(theta[1:])
    distance = z / norm if norm > 0 else 0.0
    # 'A if 조건 else B' = 조건부 표현식 (한 줄 if문)

    return {
        "label": int(z >= 0),        # True→1, False→0
        "proba": float(proba),
        "signed_distance": float(distance),   # 양수면 y=1쪽, 음수면 y=0쪽
    }
    # 딕셔너리로 반환하면 호출한 쪽에서 result["proba"] 처럼 이름으로 꺼낼 수 있다
