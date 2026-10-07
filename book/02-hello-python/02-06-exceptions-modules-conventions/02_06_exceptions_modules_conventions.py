# -*- coding: utf-8 -*-
"""
02-06 🛠️ 6부 — 실무 도구 예외 · 모듈 · 코드 컨벤션

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 02권 🐍 헬로 파이썬 — 파이썬의 탄생부터 NumPy · Pandas  · Plotly까지/02-06 🛠️ 6부 — 실무 도구  예외 · 모듈 · 코드 컨벤션.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 02-06-W 예외 처리 — 프로그램이 죽지 않게 하기
for value in ["10", "abc"]:
    try:
        result = 100 / int(value)      # 위험할 수 있는 코드
        print(result)
    except ValueError as e:      # 형변환 실패
        print("ValueError:", e)
    except ZeroDivisionError:      # 0으로 나누기
        print("0으로 나눌 수 없습니다")
    else:                          # 오류가 없었을 때만
        print("성공")
    finally:                       # 무조건 실행
        print("-- 정리 --")


# %% [Block 2] 02-06-X 모듈 · 패키지 · 파일 입출력
import math                       # ① 통째로
print(round(math.pi, 5), math.sqrt(16))

from datetime import datetime    # ② 필요한 것만
print(type(datetime.now()).__name__)

import numpy as np              # ③ 별명 붙이기 (관례!)
print(np.array([1, 2, 3]))


# %% [Block 3] 02-06-X 모듈 · 패키지 · 파일 입출력 — 쓰기 ("w" = write, 기존 내용 덮어씀)
# ── 쓰기 ("w" = write, 기존 내용 덮어씀) ──
with open("demo.txt", "w", encoding="utf-8") as f:
    f.write("첫 줄\n")          # \n = 줄바꿈
    f.write("둘째 줄\n")
# with 블록을 벗어나면 자동으로 닫힘!

# ── 읽기 ("r" = read) ──
with open("demo.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):   # 파일도 이터레이터! (R단원)
        print(f"  {i}: {line.strip()}")  # strip() = 앞뒤 공백/줄바꿈 제거

# ── NumPy 배열 저장/불러오기 (모델 가중치 저장의 원형) ──
import numpy as np
np.save("weights.npy", np.array([[1., 2.], [3., 4.]]))
print(np.load("weights.npy"))


# %% [Block 4] 02-06-Y PEP8 코드 컨벤션 — AI에게 코드를 맡기려면 필수
def Calc( a,b ):
  x=a+b
  return x

class my_layer:
    def Forward(self,X):
        return X*2
