# Zero to 머신러닝 딥러닝 Master

이 책은 WikiDocs와 연동된 GitHub 리포지토리입니다.

## Medium 시리즈 코드

책 **[Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)** 를 Medium 시리즈로 연재하며, 각 글의 실습 코드를 [`medium-series/`](medium-series/) 폴더에 정리합니다. 폴더마다 실행 스크립트(`.py`), 실행 결과가 담긴 노트북(`.ipynb`), `requirements.txt`, 설명 `README.md` 가 있습니다.

| # | 주제 | 코드 | Medium 글 | 책 원문 |
|---|---|---|---|---|
| 03 | 벡터 — 덧셈·Norm·내적·코사인 유사도 | [03-vectors](medium-series/03-vectors/) | [읽기](https://medium.com/p/9a7f826c3b2e) | [00-01](https://wikidocs.net/439839) |
| 04 | 행렬 — shape·행렬 곱셈·역행렬 | [04-matrices](medium-series/04-matrices/) | TODO: 게시 후 URL | [00-02](https://wikidocs.net/439843) |
| 05 | 최소제곱법과 정규방정식 | [05-least-squares](medium-series/05-least-squares/) | TODO: 게시 후 URL | [00-03](https://wikidocs.net/439841) |
| 06 · 1부 | 미분 — 수치미분·연쇄법칙·그래디언트 | [06-1-derivatives](medium-series/06-1-derivatives/) | TODO: 게시 후 URL | [00-04](https://wikidocs.net/439840) |
| 06 · 2부 | 경사하강법 — 학습률·선형회귀·자동미분 | [06-2-gradient-descent](medium-series/06-2-gradient-descent/) | TODO: 게시 후 URL | [00-04](https://wikidocs.net/439840) |
| 07 · 1부 | 확률·조건부 확률·베이즈 정리 | [07-1-probability-bayes](medium-series/07-1-probability-bayes/) | TODO: 게시 후 URL | [00-05](https://wikidocs.net/439838) |
| 07 · 2부 | 정규분포·소프트맥스·교차 엔트로피 | [07-2-normal-softmax-ce](medium-series/07-2-normal-softmax-ce/) | TODO: 게시 후 URL | [00-05](https://wikidocs.net/439838) |
| 08 · 1부 | 고유값·SVD·PCA | [08-1-eigen-svd-pca](medium-series/08-1-eigen-svd-pca/) | TODO: 게시 후 URL | [00-06](https://wikidocs.net/439842) |
| 08 · 2부 | SVD 추천 시스템·미니 신경망(XOR) | [08-2-svd-recsys-mini-nn](medium-series/08-2-svd-recsys-mini-nn/) | TODO: 게시 후 URL | [00-06](https://wikidocs.net/439842) |
| 10 | 왜 수학인가 — 환경 확인·20줄 맛보기 | [10-why-math](medium-series/10-why-math/) | TODO: 게시 후 URL | [01-01](https://wikidocs.net/439746) |
| 12 | 벡터 완전기초 — 노름·내적·코사인 유사도 | [12-vector-basics](medium-series/12-vector-basics/) | TODO: 게시 후 URL | [01-03](https://wikidocs.net/439747) |
| 13 | 행렬의 두 얼굴 — 표와 변환, AB ≠ BA | [13-matrix-basics](medium-series/13-matrix-basics/) | TODO: 게시 후 URL | [01-04](https://wikidocs.net/439753) |
| 14 | 내적·외적·유사도 행렬·어텐션 | [14-dot-cross-outer-similarity](medium-series/14-dot-cross-outer-similarity/) | TODO: 게시 후 URL | [01-05](https://wikidocs.net/439750) |
| 15 | 텐서 — shape·reshape·PyTorch 자동미분 | [15-tensors](medium-series/15-tensors/) | TODO: 게시 후 URL | [01-05-01](https://wikidocs.net/439758) |
| 16 | 연립방정식·행렬식·역행렬 개요 | [16-linear-systems-det-inverse](medium-series/16-linear-systems-det-inverse/) | TODO: 게시 후 URL | [01-06](https://wikidocs.net/439752) |
| 17 | 연립방정식 풀이법 (SymPy) | [17-solving-systems](medium-series/17-solving-systems/) | TODO: 게시 후 URL | [01-06-01](https://wikidocs.net/439759) |
| 18 | 행렬식(Determinant) | [18-determinant](medium-series/18-determinant/) | TODO: 게시 후 URL | [01-06-02](https://wikidocs.net/439761) |
| 19 | 역행렬 — 가우스-조던·수반행렬·성질 | [19-inverse-matrix](medium-series/19-inverse-matrix/) | TODO: 게시 후 URL | [01-06-03](https://wikidocs.net/439760) |
| 20 | 고유값·고유벡터·대각화·SVD | [20-eigen-diagonalization-svd](medium-series/20-eigen-diagonalization-svd/) | TODO: 게시 후 URL | [01-07](https://wikidocs.net/439755) |

- 📘 책 (WikiDocs): https://wikidocs.net/book/21464
- 🏠 홈페이지: https://mldict.net
- ✍️ 기술 블로그: https://wikidocs.net/blog/@mldict/


## 📘 책 단원별 소스코드 (`book/`)

책 **[Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)** 원고의 코드 블록을 단원(절)별로 추출해 [`book/`](book/) 폴더에 정리했습니다. 코드가 있는 절 157개마다 실행 스크립트(`.py`), 블록당 셀 1개인 노트북(`.ipynb`), 설명 `README.md` 가 있고, 공통 패키지는 [`book/requirements.txt`](book/requirements.txt) 에 있습니다. CPU 검증 결과 105개 절이 그대로 실행되며, 나머지는 외부 데이터·API 키·앞 절 코드가 필요한 경우로 각 README 에 이유를 적었습니다.

| 권 | 제목 | 코드가 있는 절 | 코드 블록 |
|---|---|---|---|
| 00 | [비전공자 처음부터 — 수포자 훑어보기](book/00-math-overview-for-beginners/) | 6 | 31 |
| 01 | [수포자들을 위한 머신러닝 수학 완전기초와 파이썬 맛보기](book/01-ml-math-basics-python/) | 17 | 176 |
| 02 | [🐍 헬로 파이썬 — 파이썬의 탄생부터 NumPy · Pandas · Plotly까지](book/02-hello-python/) | 9 | 140 |
| 03 | [초기(전통적) 머신러닝으로 기초 다지기](book/03-classic-ml/) | 62 | 353 |
| 04 | [신경망과 딥러닝 이론](book/04-neural-networks-deep-learning/) | 63 | 263 |

빠른 시작: `pip install -r book/requirements.txt` 후 원하는 절 폴더에서 `python <파일>.py` — 자세한 내용은 [`book/README.md`](book/README.md).

## 사용 방법

1. `TOC.md` 파일에서 목차 구조를 정의하세요.
2. `pages/` 디렉토리에 마크다운 파일을 추가하세요.
3. 변경사항을 push하면 WikiDocs에 자동으로 반영됩니다.

## 페이지 정렬 규칙 (중요!)

WikiDocs는 페이지를 **제목 알파벳순**으로 자동 정렬합니다.
원하는 순서를 유지하려면 제목에 **번호를 붙이세요**:

```markdown
# TOC.md 예시
* [01. 시작하기](pages/01-getting-started.md)
  * [01-1. 설치](pages/01-1-install.md)
  * [01-2. 환경설정](pages/01-2-config.md)
* [02. 기본 문법](pages/02-basics.md)
* [03. 심화 학습](pages/03-advanced.md)
```

## 이미지 사용

이미지는 `assets/` 디렉토리에 저장하고 상대 경로로 참조하세요:

```markdown
![이미지 설명](./assets/example.png)
```
