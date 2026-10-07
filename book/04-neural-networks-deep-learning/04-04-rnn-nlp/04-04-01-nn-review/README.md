# 04-04-01 신경망 복습

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-04-01 신경망 복습.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_04_01_nn_review.py`](04_04_01_nn_review.py) | 코드 블록 6개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_04_01_nn_review.ipynb`](04_04_01_nn_review.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-04-01-A 벡터와 행렬 — 데이터를 숫자로 담는 그릇 | 18 |
| 2 | 04-04-01-B 순전파 — 입력에서 출력까지의 여행 | 37 |
| 3 | 04-04-01-C 손실 함수 — "얼마나 틀렸는지" 채점하기 | 18 |
| 4 | 04-04-01-D 역전파 — 오답 노트로 실력 올리기 | 22 |
| 5 | 04-04-01-E 미니배치 학습 — 매일 조금씩 공부하기 — 학습 루프의 5단계 패턴 (모든 신경망 학습의 기본!) | 10 |
| 6 | 04-04-01-F 04-04장의 도구 — TwoLayerNet과 Trainer | 23 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-04-rnn-nlp/04-04-01-nn-review
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_04_01_nn_review.py
```

또는 `04_04_01_nn_review.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 상세: `max_epoch` 이(가) 이 절 안에서 정의되지 않습니다 — 데이터·변수·클래스를 미리 준비했다고 가정한 설명용 코드 조각입니다.
- 스크립트가 멈춘 지점: `NameError: name 'max_epoch' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 5/6 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
