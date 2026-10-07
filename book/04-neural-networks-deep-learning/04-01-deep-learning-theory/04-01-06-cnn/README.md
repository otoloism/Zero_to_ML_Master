# 04-01-06 합성곱 신경망 (CNN) — 이미지 인식의 혁명

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-01-06 합성곱 신경망 (CNN) — 이미지 인식의 혁명.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_01_06_cnn.py`](04_01_06_cnn.py) | 코드 블록 3개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_01_06_cnn.ipynb`](04_01_06_cnn.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-01-06-B 합성곱 연산 — "필터를 슬라이딩하며 내적" | 30 |
| 2 | 04-01-06-C 패딩 & 스트라이드 — "출력 크기를 조절하는 두 가지 장치" | 17 |
| 3 | 04-01-06-D 풀링(Pooling) — "요약해서 줄이기" | 20 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-01-deep-learning-theory/04-01-06-cnn
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_01_06_cnn.py
```

또는 `04_01_06_cnn.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
