# -*- coding: utf-8 -*-
"""
00-03연립방정식— _x가 뭔지 맞춰보세요_ 게임

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 00권  비전공자 처음부터 — 수포자 훑어보기/00-03연립방정식— _x가 뭔지 맞춰보세요_ 게임.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 00-03-R3 🔴 구현 원리와 코드 — 공식·lstsq 두 방법 비교
import numpy as np

# ── 데이터: 5명의 키(cm)와 몸무게(kg) ──────────────────────
height = np.array([165, 170, 175, 180, 185], dtype=float)
weight = np.array([58,  65,  70,  75,  82],  dtype=float)

# ── 설계행렬 A 만들기 ─────────────────────────────────────
# 찾고 싶은 직선: weight = a×height + b×1
# → 각 행이 (height, 1)이 되도록 1로 채운 열을 붙입니다 (절편 항!)
A = np.column_stack([height, np.ones(len(height))])
b = weight

print("A shape:", A.shape, "| b shape:", b.shape)   # (5,2) (5,)

# ── 방법 1) 정규방정식 공식을 그대로 구현 (이해용) ──────────
# x̂ = (AᵀA)⁻¹ Aᵀb  ← 방금 유도한 공식 그 자체
x_formula = np.linalg.inv(A.T @ A) @ A.T @ b

# ── 방법 2) lstsq 사용 (실무 권장) ────────────────────────
# rcond=None: 버전 경고 방지 / [0]: 반환 튜플의 첫 원소가 해
x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]

a_hat, b_hat = x_lstsq          # 튜플 언패킹: 기울기, 절편
print("공식 방식 :", np.round(x_formula, 4))
print("lstsq     :", np.round(x_lstsq,   4))
print(f"찾은 직선: 몸무게 = {a_hat:.2f} × 키 + ({b_hat:.2f})")

# ── 잔차와 오차 제곱합 확인 ───────────────────────────────
pred     = A @ x_lstsq          # 예측값 ŷ = Ax̂
residual = b - pred             # 잔차 e = b - Ax̂
sse      = np.sum(residual ** 2)   # 오차 제곱합 S = Σeᵢ²

print("잔차      :", np.round(residual, 3))
print("잔차 합   :", round(residual.sum(), 10), "← 0이어야 정상")
print("오차제곱합:", round(sse, 4))

# ── 직교 조건 검증: Aᵀe = 0 이어야 함 (유도 ①단계) ─────────
print("Aᵀe (≈0?) :", np.round(A.T @ residual, 10))

# ── 예측해보기 ────────────────────────────────────────────
new_height = 168
print(f"키 {new_height}cm 예측 몸무게: {a_hat * new_height + b_hat:.1f}kg")
