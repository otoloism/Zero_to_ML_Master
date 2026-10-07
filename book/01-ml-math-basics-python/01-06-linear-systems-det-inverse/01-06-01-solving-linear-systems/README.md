# 1차식: x - 2y = 0 / 2차식: x^2 + y^2 = 5

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `01-06-01📐 연립방정식의 의미와 풀이법 — 대수와 기하를 잇는 다리.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`01_06_01_solving_linear_systems.py`](01_06_01_solving_linear_systems.py) | 코드 블록 3개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`01_06_01_solving_linear_systems.ipynb`](01_06_01_solving_linear_systems.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 🔵 2단계 — 1차-2차 연립방정식: 무조건 대입법 | 10 |
| 2 | 🔵 3단계 — 해의 개수 ↔ 그래프의 교점: 대수를 기하로 해석하기 | 16 |
| 3 | 🔴 4단계 — 2차-2차 연립방정식: 인수분해가 핵심 | 14 |

## 실행 방법

```bash
cd book/01-ml-math-basics-python/01-06-linear-systems-det-inverse/01-06-01-solving-linear-systems
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 01_06_01_solving_linear_systems.py
```

또는 `01_06_01_solving_linear_systems.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
