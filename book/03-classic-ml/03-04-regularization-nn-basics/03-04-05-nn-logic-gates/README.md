# 03-04-05 신경망으로 논리 게이트 만들기

> (Examples and Intuitions — AND to XNOR) — 직선 하나로 못 자르면, 접어서 두 번 자른다

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-04-05 신경망으로 논리 게이트 만들기.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_04_05_nn_logic_gates.py`](03_04_05_nn_logic_gates.py) | 코드 블록 2개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_04_05_nn_logic_gates.ipynb`](03_04_05_nn_logic_gates.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 구현 원리 | 101 |
| 2 | 📝 해설 | 27 |

## 실행 방법

```bash
cd book/03-classic-ml/03-04-regularization-nn-basics/03-04-05-nn-logic-gates
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_04_05_nn_logic_gates.py
```

또는 `03_04_05_nn_logic_gates.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
