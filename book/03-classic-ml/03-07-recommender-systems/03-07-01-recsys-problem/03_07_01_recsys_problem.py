# -*- coding: utf-8 -*-
"""
03-07-01 추천시스템 문제 정의 (Recommender System · Predicting Movie Ratings)

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-07장 - 추천시스템/03-07-01 추천시스템 문제 정의.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 코드 구현 — 별점표를 NumPy로 옮기기🟢 초급
import numpy as np                  # 수치 계산 라이브러리를 np라는 짧은 별명으로 부른다

# ── 1) 별점 행렬 Y 만들기 ───────────────────────────────
# 행(row)   = 영화 5편,  열(column) = 사용자 4명
# 아직 평가하지 않은 칸은 np.nan("Not a Number", 값 없음 표시)으로 둔다.
#   ※ 0으로 두면 안 된다! 0점은 "아주 싫었다"는 뜻이고,
#      nan은 "아예 보지 않았다"는 뜻이라 의미가 완전히 다르다.
Y = np.array([
    [5,      5,      0,      0     ],   # Love at last
    [5,      np.nan, np.nan, 0     ],   # Romance forever
    [np.nan, 4,      0,      np.nan],   # Cute puppies of love
    [0,      0,      5,      4     ],   # Nonstop car chases
    [0,      0,      5,      np.nan],   # Swords vs. karate
], dtype=float)                          # dtype=float : nan은 실수형에만 담을 수 있다

# ── 2) 평가 여부 행렬 R 만들기 ─────────────────────────
# np.isnan(Y)  : 각 칸이 nan이면 True, 아니면 False인 표를 만든다
# ~            : True/False를 뒤집는 기호(물결표, NOT). "nan이 아니다" = "평가했다"
# .astype(int) : True→1, False→0 으로 바꾼다  → 이것이 바로 r(i, j)
R = (~np.isnan(Y)).astype(int)

# ── 3) 기본 통계 출력 ─────────────────────────────────
n_m, n_u = Y.shape                   # .shape는 (행 개수, 열 개수)를 튜플로 돌려준다
print("n_m (영화 수) =", n_m)
print("n_u (사용자 수) =", n_u)
print("R (평가 여부 행렬) =")
print(R)

# ── 4) 사용자별 평가 편수 m^(j) ───────────────────────
# R.sum(axis=0) : axis=0은 "세로 방향으로 더하라"는 뜻 → 열(=사용자)마다 합계
m_j = R.sum(axis=0)
print("사용자별 평가 편수 m^(j) =", m_j)

# ── 5) 희소도(sparsity) 계산 ─────────────────────────
# 전체 칸 수 대비 실제로 채워진 칸의 비율
filled = R.sum()                     # 인자를 안 주면 전체 합
total  = n_m * n_u
print(f"채워진 칸: {filled} / {total} = {filled/total:.1%}")
print(f"물음표 칸: {total - filled}개 → 우리가 예측해야 할 대상")


# %% [Block 2] 실습 — 실습 3 미리보기 — 영화별 평균 별점(nan 제외)
# 실습 3 미리보기 — 영화별 평균 별점(nan 제외)
print(np.round(np.nanmean(Y, axis=1), 3))
