# -*- coding: utf-8 -*-
"""
03-06-01 이상탐지와 밀도추정 개념 — (Anomaly Detection & Density Estimation) - 정상만 배워두고 "낯선 것"에 경보를 울리는 문지기

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-06장 - 이상탐지 이론/03-06-01 이상탐지와 밀도추정 개념.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 코드
import numpy as np
import matplotlib.pyplot as plt

# default_rng : NumPy의 최신 난수 생성기.
#   42 는 '시드(seed)' — 같은 숫자를 넣으면 몇 번을 돌려도 똑같은 난수가 나옵니다.
#   (실험 결과를 남과 맞춰보려면 반드시 고정!)
rng = np.random.default_rng(42)

# ── 정상 엔진 10,000개 만들기 ─────────────────────────────
# normal(loc=평균, scale=표준편차, size=모양)
#   loc=[14.0, 12.0]  → 열은 평균 14, 진동은 평균 12 근처에서 생성
#   scale=[1.2, 1.5]  → 각각 이 정도씩 흩어짐
#   size=(10000, 2)   → 10,000행 2열 (엔진 1만 개, 특성 2개)
n_good = 10000
good = rng.normal(loc=[14.0, 12.0], scale=[1.2, 1.5], size=(n_good, 2))

# ── 불량 엔진 20개 만들기 ─────────────────────────────────
# ★ 핵심 : 불량은 '한 종류'가 아닙니다. 서로 다른 두 유형을 만듭니다.
#   유형 A : 열은 과하게 높고 진동은 이상하게 낮음 (19.5, 6.0)
#   유형 B : 열은 낮은데 진동만 심함 (9.0, 17.5)
# 이렇게 서로 다른 것이 바로 "불량을 학습할 수 없는" 이유입니다.
bad = np.vstack([                     # vstack : 배열을 세로로 이어붙이기
    rng.normal(loc=[19.5, 6.0],  scale=[0.8, 0.8], size=(10, 2)),
    rng.normal(loc=[9.0,  17.5], scale=[0.8, 0.8], size=(10, 2)),
])
rng.shuffle(bad)                      # 두 유형이 섞이도록 순서 뒤섞기

print("정상 엔진:", good.shape, " 불량 엔진:", bad.shape)
print("정상 평균:", good.mean(axis=0).round(4))

# ── 그림으로 확인 ─────────────────────────────────────────
# alpha=0.3 : 투명도. 점이 겹칠 때 밀도가 눈에 보이게 해줍니다.
plt.figure(figsize=(7, 6))
plt.scatter(good[:, 0], good[:, 1], s=6, alpha=0.3,
            c='tab:red', marker='x', label='normal engines')
plt.scatter(bad[:, 0], bad[:, 1], s=90,
            c='tab:blue', marker='X', edgecolor='k', label='anomalies')
plt.xlabel('$x_1$ : heat generated')
plt.ylabel('$x_2$ : vibration intensity')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
