# #12 벡터 완전기초 — 화살표로 이해하는 노름·내적·코사인 유사도

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/12-vector-basics/12_vector_basics.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #12의 실습 코드입니다.
> 원문: 책 **01-03 「벡터 — 방향과 크기를 가진 화살표」**

스칼라와 벡터의 차이(shape), 덧셈·스칼라곱·원소별 곱·내적·노름을 NumPy로 확인하고, 문서 벡터의 내적과 코사인 유사도가 정반대 결론을 내는 예, 그리고 "왕 − 남자 + 여자 ≈ 여왕" 단어 벡터 연산까지 실행합니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | **TODO: Medium 게시 후 URL 입력** (`https://medium.com/p/...`) |
| 📘 책 원문 페이지 | [01-03 벡터 — 방향과 크기를 가진 화살표](https://wikidocs.net/439747) |
| ✍️ 블로그 글 | [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/) (이 편의 블로그 글은 아직 없음) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`12_vector_basics.py`](12_vector_basics.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`12_vector_basics.ipynb`](12_vector_basics.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`numpy`) |

## 🧭 코드 순서

1. 1단계 — "북쪽으로 3km" vs "그냥 3km": 스칼라와 벡터
2. 6단계 — 벡터 연산과 내적
3. 7단계 — 코사인 유사도: 길이는 잊고 방향만
4. 8단계 — ML에서 벡터: 단어 임베딩 연산

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 12_vector_basics.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
