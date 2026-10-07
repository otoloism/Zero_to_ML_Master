# 04-02-07 신경망과 Q 러닝 — 함수 근사로 가는 길

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-02-07 신경망과 Q 러닝 — 함수 근사로 가는 길.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_02_07_nn_q_learning.py`](04_02_07_nn_q_learning.py) | 코드 블록 2개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_02_07_nn_q_learning.ipynb`](04_02_07_nn_q_learning.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-02-07-A self.Q = {} 라는 딕셔너리의 함정 — 각 환경의 상태 수를 비교해 봅시다 | 13 |
| 2 | 04-02-07-C DeZero(04-03장)로 QNet 구현 — 04-01장 도구가 그대로! — DeZero(04-03장) 스타일의 Q 신경망 | 33 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-02-reinforcement-learning/04-02-07-nn-q-learning
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_02_07_nn_q_learning.py
```

또는 `04_02_07_nn_q_learning.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **❌ 실행 오류** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 상세: Python 3.11+ 의 정수→문자열 변환 자릿수 제한(4300자리) 때문에 `len(str(256**(84*84)))` 에서 `ValueError` — 책 집필 환경(구버전 Python)과의 차이. `sys.set_int_max_str_digits(0)` 를 먼저 실행하면 동작합니다.
- 블록별 실행(오류가 나도 다음 블록 계속): 0/2 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
