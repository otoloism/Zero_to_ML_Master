# #20 고유값·고유벡터·대각화·SVD — Av = λv 한 줄로 시작하는 행렬 분해

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/20-eigen-diagonalization-svd/20_eigen_diagonalization_svd.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #20의 실습 코드입니다.
> 원문: 책 **01-07 「고유값·고유벡터·대각화·SVD」**

np.linalg.eig 로 고유값·고유벡터를 구해 Av = λv 를 검증하고, A = PDP⁻¹ 대각화로 원래 행렬을 재구성한 뒤, 정사각행렬이 아닌 2×3 행렬에 SVD 를 적용합니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | **TODO: Medium 게시 후 URL 입력** (`https://medium.com/p/...`) |
| 📘 책 원문 페이지 | [01-07 고유값·고유벡터·대각화·SVD](https://wikidocs.net/439755) |
| ✍️ 블로그 글 | [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/) (이 편의 블로그 글은 아직 없음) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`20_eigen_diagonalization_svd.py`](20_eigen_diagonalization_svd.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`20_eigen_diagonalization_svd.ipynb`](20_eigen_diagonalization_svd.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`numpy`) |

## 🧭 코드 순서

1. 01-07-F — 코드로 확인하기: 고유값·대각화·SVD

## ⚠️ 참고

- **1. 01-07-F — 코드로 확인하기: 고유값·대각화·SVD** — NumPy 2.5 이상에서는 `np.linalg.eig` 가 항상 복소수 배열을 돌려주므로 값 뒤에 `+0.j` 가 붙어 출력될 수 있습니다(값은 같음). 또 고유값의 순서와 고유벡터의 부호(±)는 NumPy/LAPACK 환경에 따라 책과 다를 수 있습니다 — 고유벡터는 방향만 같으면 됩니다. 이 노트북은 NumPy 2.4.4로 실행했습니다.

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 20_eigen_diagonalization_svd.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
