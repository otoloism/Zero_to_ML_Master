# #08 · 1부 고유값·SVD·PCA 쉽게 이해하기 — 300페이지 책을 3페이지로 (1부)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/08-1-eigen-svd-pca/08_1_eigen_svd_pca.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #08 · 1부의 실습 코드입니다.
> 원문: 책 **00-06 「SVD·PCA — 300페이지 책을 3페이지로 (1부)」**

고유값·고유벡터를 np.linalg.eig 로 구해 Av = λv 와 det(A − λI) = 0 을 검증하고, SVD로 행렬을 랭크-k 근사해 에너지 비율과 에카르트-영 오차를 확인합니다. 키·몸무게 데이터에서 공분산 고유분해와 SVD 두 경로로 PCA를 수행하고 2D→1D 압축·복원 오차를 이론값과 비교합니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | **TODO: Medium 게시 후 URL 입력** (`https://medium.com/p/...`) |
| 📘 책 원문 페이지 | [00-06 SVD·PCA — 300페이지 책을 3페이지로 (1부)](https://wikidocs.net/439842) |
| ✍️ 블로그 글 | [https://wikidocs.net/blog/@mldict/32652/](https://wikidocs.net/blog/@mldict/32652/) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`08_1_eigen_svd_pca.py`](08_1_eigen_svd_pca.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`08_1_eigen_svd_pca.ipynb`](08_1_eigen_svd_pca.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`numpy`) |

## 🧭 코드 순서

1. 00-06-A · 1단계 — 고유벡터와 고유값
2. 00-06-B · 2단계 — SVD와 랭크-k 근사
3. 00-06-C · 3단계 — PCA: 공분산 고유분해 vs SVD

## ⚠️ 참고

- **1. 00-06-A · 1단계 — 고유벡터와 고유값** — NumPy 2.5 이상에서는 `np.linalg.eig` 가 항상 복소수 배열을 돌려주므로 값 뒤에 `+0.j` 가 붙어 출력될 수 있습니다(값은 같음). 또 고유값의 순서와 고유벡터의 부호(±)는 NumPy/LAPACK 환경에 따라 책과 다를 수 있습니다 — 고유벡터는 방향만 같으면 됩니다. 이 노트북은 NumPy 2.4.4로 실행했습니다.

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 08_1_eigen_svd_pca.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
