# 01-08 함수와합성함수— 딥러닝의 구조

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `01-08 함수와합성함수— 딥러닝의 구조.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`01_08_functions_composition.py`](01_08_functions_composition.py) | 코드 블록 8개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`01_08_functions_composition.ipynb`](01_08_functions_composition.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 📊 그림 1 — 합성함수 $g(f(x))$: 상자를 통과하며 값이 변한다 — 함수 = 입력을 규칙대로 바꿔 출력하는 상자 | 13 |
| 2 | 1-2 🔵⭐ 신경망 = 함수를 수백 겹 쌓은 것 | 17 |
| 3 | 2-1 🔵⭐ 왜 활성화 함수가 없으면 안 되는가 | 10 |
| 4 | 📊 그림 2 — 세 활성화 함수의 모양 (전부 곡선 = 비선형) | 12 |
| 5 | 3-1 🔵⭐ 연쇄법칙 — 합성함수를 미분하는 단 하나의 규칙 — 연쇄법칙:  dy/dx = (dy/du) × (du/dx) | 12 |
| 6 | 3-1 🔵⭐ 연쇄법칙 — 합성함수를 미분하는 단 하나의 규칙 — 연쇄법칙 결과가 맞는지 '수치미분'으로 검산 | 11 |
| 7 | 📊 그림 4 — 층이 깊어질수록 기울기가 0으로 (0.25ⁿ) | 11 |
| 8 | 4-2 🔵⭐ ReLU라는 해법 — 곱해도 안 줄어드는 미분 | 14 |

## 실행 방법

```bash
cd book/01-ml-math-basics-python/01-08-functions-composition
pip install -r ../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 01_08_functions_composition.py
```

또는 `01_08_functions_composition.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
