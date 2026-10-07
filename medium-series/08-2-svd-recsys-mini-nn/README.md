# #08 · 2부 SVD 추천 시스템부터 미니 신경망까지 — 수포자 수학 6개 장을 하나로 (2부)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/08-2-svd-recsys-mini-nn/08_2_svd_recsys_mini_nn.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #08 · 2부의 실습 코드입니다.
> 원문: 책 **00-06 「SVD·PCA — 300페이지 책을 3페이지로 (2부)」**

사용자×영화 평점 행렬을 SVD 랭크-k 근사로 채워 빈 평점을 예측하고(사용자 평균 중심화 개선 포함), 벡터·행렬·미분·확률을 모두 써서 XOR을 학습하는 2층 신경망을 NumPy만으로 구현한 뒤 역전파 기울기를 수치미분으로 검증합니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | **TODO: Medium 게시 후 URL 입력** (`https://medium.com/p/...`) |
| 📘 책 원문 페이지 | [00-06 SVD·PCA — 300페이지 책을 3페이지로 (2부)](https://wikidocs.net/439842) |
| ✍️ 블로그 글 | [https://wikidocs.net/blog/@mldict/32652/](https://wikidocs.net/blog/@mldict/32652/) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`08_2_svd_recsys_mini_nn.py`](08_2_svd_recsys_mini_nn.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`08_2_svd_recsys_mini_nn.ipynb`](08_2_svd_recsys_mini_nn.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`numpy`) |

## 🧭 코드 순서

1. 00-06-D · 4단계 — SVD 추천 시스템
2. 00-06-E · 5단계 — 미니 신경망으로 XOR 학습

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 08_2_svd_recsys_mini_nn.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
