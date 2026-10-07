# 04-04-03 word2vec— 추론 기반 단어임베딩

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-04-03 word2vec— 추론 기반 단어임베딩.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_04_03_word2vec.py`](04_04_03_word2vec.py) | 코드 블록 2개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_04_03_word2vec.ipynb`](04_04_03_word2vec.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-04-03-A CBOW 모델 — 주변 단어로 빈칸 맞추기 | 31 |
| 2 | 04-04-03-C Skip-gram과 "왕 - 남자 + 여자 = 여왕" — 학습이 완료된 임베딩이 있다고 가정 | 9 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-04-rnn-nlp/04-04-03-word2vec
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_04_03_word2vec.py
```

또는 `04_04_03_word2vec.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
