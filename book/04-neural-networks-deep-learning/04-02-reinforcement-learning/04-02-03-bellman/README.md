# 04-02-03 벨만 방정식— 가치 함수의 재귀 구조

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-02-03 벨만 방정식— 가치 함수의 재귀 구조.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_02_03_bellman.py`](04_02_03_bellman.py) | 코드 블록 5개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_02_03_bellman.ipynb`](04_02_03_bellman.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-02-03-B 상태 가치 Vπ(s) — "이 칸에서 시작하면 앞으로 얼마를 벌까?" | 53 |
| 2 | 04-02-03-E 벨만 방정식으로 V(s) 계산하기 — 코드 구현 | 30 |
| 3 | 04-02-03-E 벨만 방정식으로 V(s) 계산하기 — 코드 구현 | 38 |
| 4 | 04-02-03-F 벨만 최적 방정식 — "최고의 선택을 했을 때의 가치" | 45 |
| 5 | 04-02-03-G V*에서 최적 정책 추출 — "가치가 높은 쪽으로 가!" | 31 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-02-reinforcement-learning/04-02-03-bellman
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_02_03_bellman.py
```

또는 `04_02_03_bellman.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: 앞 절 04-02-02 의 코드 실행 결과 (`GridWorld` 등) — 책이 '앞 절에서 만든 것을 그대로 쓴다'고 가정
- 스크립트가 멈춘 지점: `NameError: name 'GridWorld' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 1/5 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
