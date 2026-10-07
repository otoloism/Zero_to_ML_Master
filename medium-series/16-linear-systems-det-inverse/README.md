# #16 연립방정식·행렬식·역행렬 한 번에 — det(A) = 0이 말해 주는 것

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/16-linear-systems-det-inverse/16_linear_systems_det_inverse.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #16의 실습 코드입니다.
> 원문: 책 **01-06 「연립방정식·행렬식·역행렬」**

해가 유일한 행렬과 평행한 두 직선(해 없음)을 나타내는 행렬의 행렬식을 비교하고, det(A)=0 인 행렬의 역행렬을 구하려 하면 LinAlgError 가 나는 것을 확인합니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | **TODO: Medium 게시 후 URL 입력** (`https://medium.com/p/...`) |
| 📘 책 원문 페이지 | [01-06 연립방정식·행렬식·역행렬](https://wikidocs.net/439752) |
| ✍️ 블로그 글 | [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/) (이 편의 블로그 글은 아직 없음) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`16_linear_systems_det_inverse.py`](16_linear_systems_det_inverse.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`16_linear_systems_det_inverse.ipynb`](16_linear_systems_det_inverse.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`numpy`) |

## 🧭 코드 순서

1. 01-06-E — 코드로 확인하기: det(A)=0 이면 역행렬이 없다

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 16_linear_systems_det_inverse.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
