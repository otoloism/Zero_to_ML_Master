# -*- coding: utf-8 -*-
"""
02-01 🌱 1부 — 파이썬이라는 언어의 이야기

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 02권 🐍 헬로 파이썬 — 파이썬의 탄생부터 NumPy · Pandas  · Plotly까지/02-01 🌱 1부 — 파이썬이라는 언어의 이야기.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 💡 가상환경(venv) — 프로젝트마다 방을 따로 쓰기
print("Hello Python!")         # 화면에 글자 출력
print(2 + 3 * 4)                # 계산기로도 쓸 수 있다

import sys
print("파이썬 버전:", sys.version.split()[0])
