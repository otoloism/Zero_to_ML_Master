# #13 행렬의 두 얼굴 — 데이터를 담는 표, 공간을 바꾸는 장치 (행렬 곱셈이 쉬워지는 법)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/13-matrix-basics/13_matrix_basics.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #13의 실습 코드입니다.
> 원문: 책 **01-04 「행렬 — 숫자를 표로 정리하기」**

행렬을 "데이터를 담는 표"와 "벡터를 바꾸는 함수" 두 얼굴로 보고, 행렬×벡터를 행 내적·열 가중합 두 관점으로 계산합니다. AB ≠ BA 를 회전·전단 변환으로 확인하고, 연립방정식·역행렬·전치, 그리고 신경망 순전파 XW 의 shape 설계까지 실행합니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | **TODO: Medium 게시 후 URL 입력** (`https://medium.com/p/...`) |
| 📘 책 원문 페이지 | [01-04 행렬 — 숫자를 표로 정리하기](https://wikidocs.net/439753) |
| ✍️ 블로그 글 | [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/) (이 편의 블로그 글은 아직 없음) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`13_matrix_basics.py`](13_matrix_basics.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`13_matrix_basics.ipynb`](13_matrix_basics.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`numpy`) |

## 🧭 코드 순서

1. 1단계 — 첫 번째 얼굴: 행렬은 표다
2. 1.5단계 — 행렬 = 함수
3. 3단계 — 덧셈과 스칼라곱 (원소별 곱 vs 행렬 곱)
4. 읽는 법 ② — 행렬×벡터의 두 관점
5. 6단계 — 왜 AB ≠ BA 인가
6. 7단계 — 연립방정식·역행렬·전치
7. 8단계 — ML에서의 행렬: 순전파와 배치 shape

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 13_matrix_basics.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
