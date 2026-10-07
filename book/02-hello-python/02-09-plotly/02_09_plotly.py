# -*- coding: utf-8 -*-
"""
02-09 📊9부 — Plotly 결과를 눈으로 확인하기

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 02권 🐍 헬로 파이썬 — 파이썬의 탄생부터 NumPy · Pandas  · Plotly까지/02-09 📊9부 — Plotly  결과를 눈으로 확인하기.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 02-09-AT Plotly 입문 — 왜 matplotlib이 아니라 Plotly인가 🟢
import numpy as np
import plotly.graph_objects as go

# ── 데이터 준비 ──
x = np.arange(0, 6, 0.1)          # 0~6을 0.1 간격 (60개 점)
y_sin = np.sin(x)
y_cos = np.cos(x)

# ── 도화지(Figure) 만들기 ──
fig = go.Figure()

# ── 선 하나씩 올리기 (trace = 그래프 한 줄) ──
fig.add_trace(go.Scatter(
    x=x, y=y_sin,
    mode="lines",                # 선으로 (markers=점, lines+markers=둘 다)
    name="sin"                   # 범례에 표시될 이름
))
fig.add_trace(go.Scatter(
    x=x, y=y_cos,
    mode="lines",
    name="cos",
    line=dict(dash="dash")       # 점선으로
))

# ── 제목·축 이름 등 꾸미기 ──
fig.update_layout(
    title="사인과 코사인",
    xaxis_title="x",
    yaxis_title="y",
    template="plotly_white",     # 깔끔한 흰 배경 테마
    width=700, height=400
)

fig.show()                        # 그래프 띄우기!


# %% [Block 2] 02-09-AT Plotly 입문 — 왜 matplotlib이 아니라 Plotly인가 🟢
import pandas as pd
import plotly.express as px

sales = pd.DataFrame({
    "월": ["1월", "2월", "3월", "4월"],
    "매출": [100, 200, 150, 300],
    "비용": [80, 120, 130, 160],
})
fig = px.line(sales, x="월", y=["매출", "비용"], markers=True,   # 열 이름만 지정
              title="월별 매출과 비용", template="plotly_white")
fig.show()


# %% [Block 3] 02-09-AU Plotly 실전 5종 — 머신러닝에서 실제로 쓰는 그래프 🔵
import numpy as np
import plotly.graph_objects as go

# 가짜 학습 기록 (🔗 01-13의 실제 손실 패턴을 흉내)
epochs = np.arange(1, 301)
train_loss = 0.75 * np.exp(-epochs / 60) + 0.004
valid_loss = 0.75 * np.exp(-epochs / 70) + 0.02

fig = go.Figure()
fig.add_trace(go.Scatter(x=epochs, y=train_loss,
                         mode="lines", name="훈련 손실"))
fig.add_trace(go.Scatter(x=epochs, y=valid_loss,
                         mode="lines", name="검증 손실",
                         line=dict(dash="dash")))
fig.update_layout(title="학습 곡선", xaxis_title="에폭",
                  yaxis_title="손실", template="plotly_white")
fig.show()


# %% [Block 4] 02-09-AU Plotly 실전 5종 — 머신러닝에서 실제로 쓰는 그래프 🔵
import numpy as np
import plotly.graph_objects as go

# ── 산점도: 두 변수의 관계 보기 (🔗 01-12 공분산) ──
rng = np.random.default_rng(42)
study = rng.uniform(1, 10, 50)
score = 50 + 5 * study + rng.normal(0, 5, 50)

fig = go.Figure(go.Scatter(
    x=study, y=score,
    mode="markers",                       # 점만 찍기
    marker=dict(size=9, color=score,      # 값에 따라 색 변화
                colorscale="Viridis",
                showscale=True)
))
fig.update_layout(title="공부 시간과 점수",
                  xaxis_title="공부 시간", yaxis_title="점수",
                  template="plotly_white")
fig.show()

# ── 막대: 클래스별 개수 등 ──
fig2 = go.Figure(go.Bar(
    x=["고양이", "강아지", "새"],
    y=[120, 95, 60],
    marker_color=["#3b82f6", "#22c55e", "#eab308"]
))
fig2.update_layout(title="클래스별 데이터 개수",
                   template="plotly_white")
fig2.show()


# %% [Block 5] 02-09-AU Plotly 실전 5종 — 머신러닝에서 실제로 쓰는 그래프 🔵
import numpy as np
import plotly.graph_objects as go

# 혼동행렬 예시 (실제 정답 vs 모델 예측)
confusion = np.array([[45, 3, 2],
                      [5, 40, 5],
                      [1, 4, 45]])
labels = ["고양이", "강아지", "새"]

fig = go.Figure(go.Heatmap(
    z=confusion,
    x=labels, y=labels,
    colorscale="Blues",
    text=confusion,                   # 칸 안에 숫자 표시
    texttemplate="%{text}"
))
fig.update_layout(title="혼동행렬",
                  xaxis_title="모델의 예측", yaxis_title="실제 정답",
                  template="plotly_white")
fig.show()


# %% [Block 6] 02-09-AU Plotly 실전 5종 — 머신러닝에서 실제로 쓰는 그래프 🔵
import numpy as np
import plotly.express as px

# 28×28 가짜 이미지 (실제 MNIST는 04-01-02에서!)
rng = np.random.default_rng(42)
fake_image = rng.random((28, 28))       # 0~1 랜덤 값

print("이미지 shape:", fake_image.shape)
print("픽셀 수     :", 28 * 28, "← 신경망 입력 차원!")

fig = px.imshow(fake_image,
                color_continuous_scale="gray")  # 흑백으로
fig.update_layout(title="28x28 이미지",
                  width=400, height=400)
# fig.show()   ← 코랩에서 실행하면 28×28 흑백 이미지가 표시됩니다
