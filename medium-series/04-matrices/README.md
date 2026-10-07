# #04 행렬 기초 — shape·행렬 곱셈·역행렬을 성적표로 이해하기

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/04-matrices/04_matrices.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #04의 실습 코드입니다.
> 원문: 책 **00-02 「행렬 — 성적표는 곧 행렬이다」**

성적표를 행렬로 보고 shape("행이 먼저!")를 확인한 뒤, 흑백 이미지도 숫자 행렬이라는 걸 눈으로 보고, 행렬 곱셈 공식 Cᵢⱼ = Σₖ AᵢₖBₖⱼ 를 삼중 반복문으로 직접 구현해 `@` 와 비교하고, 역행렬과 `np.linalg.solve` 로 Ax = b 를 풉니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | **TODO: Medium 게시 후 URL 입력** (`https://medium.com/p/...`) |
| 📘 책 원문 페이지 | [00-02 행렬 — 성적표는 곧 행렬이다](https://wikidocs.net/439843) |
| ✍️ 블로그 글 | [https://wikidocs.net/blog/@mldict/32648/](https://wikidocs.net/blog/@mldict/32648/) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`04_matrices.py`](04_matrices.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`04_matrices.ipynb`](04_matrices.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`numpy`) |

## 🧭 코드 순서

1. 00-02-D 📏 행렬의 크기(shape) — "몇 행 몇 열인가요?"
2. 00-02-E 🌍 우리 주변의 행렬들 — 사진도 숫자 표
3. 00-02-Q 🔁 행렬 곱셈 — "행과 열을 짝지어서 내적한다"
4. 00-02-Y 🤔 역행렬 — "행렬에는 나눗셈이 없다"

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 04_matrices.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
