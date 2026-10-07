# -*- coding: utf-8 -*-
"""
04-01-06 합성곱 신경망 (CNN) — 이미지 인식의 혁명

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-01장 딥러닝 이론과 구현/04-01-06 합성곱 신경망 (CNN) — 이미지 인식의 혁명.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-01-06-B 합성곱 연산 — "필터를 슬라이딩하며 내적"
import numpy as np

def conv2d_simple(x, kernel):
    """가장 단순한 2D 합성곱 (패딩 없음, 스트라이드 1).
    비유: '돋보기를 왼쪽 위부터 한 칸씩 밀며 검사'"""
    H, W = x.shape                      # 입력 크기
    KH, KW = kernel.shape                # 필터 크기
    OH = H - KH + 1                     # 출력 높이
    OW = W - KW + 1                     # 출력 너비

    out = np.zeros((OH, OW))
    for i in range(OH):
        for j in range(OW):
            # 겹치는 부분 잘라내기 → 원소별 곱 → 합산
            patch = x[i:i+KH, j:j+KW]   # 입력의 3×3 영역
            out[i, j] = np.sum(patch * kernel)  # 내적!
    return out

# ── 테스트 ──
x = np.array([[1,2,3,0],
              [0,1,2,3],
              [3,0,1,1],
              [2,1,0,2]])
kernel = np.array([[2,0,1],
                   [0,1,2],
                   [1,0,2]])

result = conv2d_simple(x, kernel)
print("합성곱 결과:")
print(result)


# %% [Block 2] 04-01-06-C 패딩 & 스트라이드 — "출력 크기를 조절하는 두 가지 장치"
import numpy as np

def calc_output_size(input_size, filter_size, padding, stride):
    """합성곱 출력 크기 계산 공식."""
    return (input_size + 2 * padding - filter_size) // stride + 1

# ── 예제: 28×28 입력, 3×3 필터 ──
print("28×28 입력, 3×3 필터:")
print(f"  패딩=0, 스트라이드=1 → {calc_output_size(28,3,0,1)}")  # 26
print(f"  패딩=1, 스트라이드=1 → {calc_output_size(28,3,1,1)}")  # 28 (유지!)
print(f"  패딩=0, 스트라이드=2 → {calc_output_size(28,3,0,2)}")  # 13 (절반!)

# ── np.pad로 패딩 적용 ──
x = np.array([[1,2,3],[4,5,6],[7,8,9]])
x_pad = np.pad(x, pad_width=1, constant_values=0)  # 상하좌우 0 한 줄
print(f"\n원본 {x.shape}:\n{x}")
print(f"패딩 후 {x_pad.shape}:\n{x_pad}")


# %% [Block 3] 04-01-06-D 풀링(Pooling) — "요약해서 줄이기"
def max_pool_2x2(x):
    """2×2 맥스 풀링: 2×2 영역에서 최댓값만 남김.
    비유: '교실에서 가장 키 큰 학생만 대표로 남기기'"""
    H, W = x.shape
    OH, OW = H // 2, W // 2
    out = np.zeros((OH, OW))

    for i in range(OH):
        for j in range(OW):
            patch = x[i*2:i*2+2, j*2:j*2+2]  # 2×2 영역
            out[i, j] = np.max(patch)   # 최댓값만!
    return out

# ── 테스트 ──
x = np.array([[1, 3, 2, 1],
              [4, 2, 5, 0],
              [0, 7, 1, 3],
              [6, 2, 8, 4]])
print(f"입력 {x.shape}:\n{x}")
print(f"\n2×2 맥스 풀링 결과:\n{max_pool_2x2(x).astype(int)}")
