# 04-02-01 밴디트 문제 — 강화학습의 작은 출발점

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-02-01 밴디트 문제 — 강화학습의 작은 출발점.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_02_01_bandit.py`](04_02_01_bandit.py) | 코드 블록 6개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_02_01_bandit.ipynb`](04_02_01_bandit.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-02-01-A 밴디트 문제란? — "어느 슬롯머신이 최고인지 찾기" | 18 |
| 2 | 04-02-01-C 탐욕 전략 (Greedy) — "최고만 고집하는 전략" | 30 |
| 3 | 04-02-01-C 탐욕 전략 (Greedy) — "최고만 고집하는 전략" | 28 |
| 4 | 04-02-01-D ε-탐욕 전략 (ε-Greedy) — "가끔은 모험하자" | 27 |
| 5 | 04-02-01-E UCB 전략 — "덜 시도한 슬롯에 보너스를 주자" | 49 |
| 6 | 04-02-01-F 비정상 문제 — "맛이 바뀌는 뷔페에서 살아남기" | 64 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-02-reinforcement-learning/04-02-01-bandit
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_02_01_bandit.py
```

또는 `04_02_01_bandit.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
