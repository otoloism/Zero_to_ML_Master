# 04-02-04 동적 프로그래밍 (DP) — 환경 모델을 알 때

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-02-04 동적 프로그래밍 (DP) — 환경 모델을 알 때.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_02_04_dynamic_programming.py`](04_02_04_dynamic_programming.py) | 코드 블록 4개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_02_04_dynamic_programming.ipynb`](04_02_04_dynamic_programming.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-02-04-B 정책 평가 — "이 정책이면 각 칸의 가치는?" | 23 |
| 2 | 04-02-04-C 정책 개선 — "가치가 높은 방향으로 바꾸기" | 21 |
| 3 | 04-02-04-D 정책 반복 — "평가와 개선을 번갈아 반복" | 59 |
| 4 | 04-02-04-E 가치 반복 — "평가와 개선을 한 줄로 합치기" | 25 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-02-reinforcement-learning/04-02-04-dynamic-programming
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_02_04_dynamic_programming.py
```

또는 `04_02_04_dynamic_programming.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: 앞 절 04-02-02 의 코드 실행 결과 (`GridWorld` 등) — 책이 '앞 절에서 만든 것을 그대로 쓴다'고 가정
- 스크립트가 멈춘 지점: `NameError: name 'GridWorld' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 2/4 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
