# 04-04-02 자연어와 단어의 분산 표현 — 의미를벡터로

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-04-02 자연어와 단어의 분산 표현 — 의미를벡터로.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_04_02_distributed_representation.py`](04_04_02_distributed_representation.py) | 코드 블록 5개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_04_02_distributed_representation.ipynb`](04_04_02_distributed_representation.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-04-02-A 원-핫 벡터의 결정적 문제 — "모든 단어가 똑같이 멀다" | 11 |
| 2 | 04-04-02-B 분포 가설 — "친구를 보면 그 사람을 알 수 있다" | 33 |
| 3 | 04-04-02-C 코사인 유사도 — 나침반 방향이 얼마나 같은가? | 14 |
| 4 | 04-04-02-D PPMI — "단순히 자주 나온다고 중요한 건 아니다" | 17 |
| 5 | 04-04-02-E SVD로 차원 축소 — "300페이지를 3페이지로 요약" — SVD 실행: W = U × S × V^T | 7 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-04-rnn-nlp/04-04-02-distributed-representation
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_04_02_distributed_representation.py
```

또는 `04_04_02_distributed_representation.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
