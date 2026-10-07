# 04-04-07 RNN을 사용한 문장 생성 —seq2seq

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-04-07 RNN을 사용한 문장 생성 —seq2seq.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_04_07_seq2seq.py`](04_04_07_seq2seq.py) | 코드 블록 2개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_04_07_seq2seq.ipynb`](04_04_07_seq2seq.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-04-07-A Toy 문제: 덧셈 문자열 번역 ("57+5" → "62") | 18 |
| 2 | 04-04-07-B Reverse 트릭의 효과 — Reverse 트릭: 입력 문자열을 뒤집는다 | 12 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-04-rnn-nlp/04-04-07-seq2seq
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_04_07_seq2seq.py
```

또는 `04_04_07_seq2seq.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
