# 04-03-03 고차미분계산 (25 36단계) —헤시안과 뉴턴법

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-03-03 고차미분계산 (25 36단계) —헤시안과 뉴턴법.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_03_03_higher_order_diff_steps_25_36.py`](04_03_03_higher_order_diff_steps_25_36.py) | 코드 블록 7개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_03_03_higher_order_diff_steps_25_36.ipynb`](04_03_03_higher_order_diff_steps_25_36.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 25~26단계 — 계산 그래프 시각화: Graphviz로 그림 그리기 | 22 |
| 2 | 27단계 — 테일러 급수로 sin 함수 미분 검증 | 21 |
| 3 | 28단계 — 함수 최적화: 경사하강법으로 Rosenbrock 최솟값 찾기 | 17 |
| 4 | 29~31단계 — 뉴턴 방법: 수동 → 자동 2차 미분 | 14 |
| 5 | 32단계 — 고차 미분의 핵심: backward 자체를 그래프로 기록 | 10 |
| 6 | 33단계 — 뉴턴 방법 자동화: 2차 미분도 자동으로! | 21 |
| 7 | 34~35단계 — sin 함수 고차 미분: 미분을 반복하면 돌아옵니다 | 12 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-03-dezero-framework/04-03-03-higher-order-diff-steps-25-36
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_03_03_higher_order_diff_steps_25_36.py
```

또는 `04_03_03_higher_order_diff_steps_25_36.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: 앞 절 04-03-02 의 코드 실행 결과 (`Variable` 등) — 책이 '앞 절에서 만든 것을 그대로 쓴다'고 가정
- 스크립트가 멈춘 지점: `NameError: name 'Variable' is not defined. Did you mean: 'callable'?`
- 블록별 실행(오류가 나도 다음 블록 계속): 0/7 블록 성공

## 추출 시 수정 사항

- Block 0 추가: `import numpy as np` — 책에서는 앞 절에서 import 했거나 뒤 블록에서야 import 하는 모듈이라, 단독 실행을 위해 보충했습니다.

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
