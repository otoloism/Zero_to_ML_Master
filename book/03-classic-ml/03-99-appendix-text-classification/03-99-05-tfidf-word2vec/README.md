# max_features: 상위 5000개 단어만 사용 (메모리 절약)

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-99-05 🔢 TF-IDF와 Word2Vec으로 텍스트 분류하기 — 단어를 숫자로.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_99_05_tfidf_word2vec.py`](03_99_05_tfidf_word2vec.py) | 코드 블록 2개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_99_05_tfidf_word2vec.ipynb`](03_99_05_tfidf_word2vec.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | ⑥ 구현 관점 | 23 |
| 2 | ⑥ 구현 관점 | 35 |

## 실행 방법

```bash
cd book/03-classic-ml/03-99-appendix-text-classification/03-99-05-tfidf-word2vec
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_99_05_tfidf_word2vec.py
```

또는 `03_99_05_tfidf_word2vec.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: 앞 절 03-99-01 의 코드 실행 결과 (`X_train` 등) — 책이 '앞 절에서 만든 것을 그대로 쓴다'고 가정
- 스크립트가 멈춘 지점: `NameError: name 'X_train' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 0/2 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
