# #03 벡터란? 장바구니로 이해하는 벡터·내적·코사인 유사도 (NumPy 실습)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/03-vectors/03_vectors.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #03의 실습 코드입니다.
> 원문: 책 **00-01 「벡터 — 장바구니 속 숫자의 줄서기」**

벡터를 "자리가 정해진 숫자 목록(장바구니)"으로 이해하고, NumPy로 덧셈·뺄셈, 스칼라 곱, 크기(Norm)와 정규화, 내적과 코사인 유사도(영화 추천의 원리), 그리고 브로드캐스트·벡터화의 속도 차이까지 직접 실행해 봅니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | [https://medium.com/p/9a7f826c3b2e](https://medium.com/p/9a7f826c3b2e) |
| 📘 책 원문 페이지 | [00-01 벡터 — 장바구니 속 숫자의 줄서기](https://wikidocs.net/439839) |
| ✍️ 블로그 글 | [https://wikidocs.net/blog/@mldict/32647/](https://wikidocs.net/blog/@mldict/32647/) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`03_vectors.py`](03_vectors.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`03_vectors.ipynb`](03_vectors.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`numpy`) |

## 🧭 코드 순서

1. 1단계 — 벡터란 무엇인가?
2. 2단계 — 벡터의 덧셈과 뺄셈
3. 3단계 — 스칼라 곱
4. 4단계 — 벡터의 크기(Norm)
5. 5단계 — 내적(Dot Product)과 코사인 유사도
6. 심화 🔴 — 브로드캐스트와 벡터화

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 03_vectors.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
