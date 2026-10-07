# 04-02-06 TD법 —SARSA와Q-Learning

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-02-06 TD법 —SARSA와Q-Learning.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_02_06_td_sarsa_qlearning.py`](04_02_06_td_sarsa_qlearning.py) | 코드 블록 4개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_02_06_td_sarsa_qlearning.ipynb`](04_02_06_td_sarsa_qlearning.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-02-06-B TD의 통찰 — "한 걸음의 보상 + 다음 상태의 추정치" | 23 |
| 2 | 04-02-06-C SARSA — "(S, A, R, S', A')의 다섯 글자" | 32 |
| 3 | 04-02-06-D Q-Learning — "최선을 다했다고 가정하고 학습" | 32 |
| 4 | 04-02-06-E 클리프 워킹 — SARSA와 Q-Learning의 차이를 눈으로 보기 — SARSA와 Q-Learning의 차이를 한눈에 | 20 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-02-reinforcement-learning/04-02-06-td-sarsa-qlearning
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_02_06_td_sarsa_qlearning.py
```

또는 `04_02_06_td_sarsa_qlearning.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
