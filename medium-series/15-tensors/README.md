# #15 텐서(Tensor)란 무엇인가 — 스칼라·벡터·행렬에서 다차원 배열까지, 숫자 상자의 차원 여행

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/15-tensors/15_tensors.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #15의 실습 코드입니다.
> 원문: 책 **01-05-01 「텐서(Tensor)란 뭘까 — 숫자 상자의 차원 여행」**

0차원(스칼라)부터 3차원 텐서까지 NumPy로 shape·ndim 을 확인하고, reshape 로 원소는 그대로 모양만 바꿉니다. PyTorch 텐서에 requires_grad 를 켜고 backward() 로 기울기를 구해 텐서가 계산 그래프의 노드가 되는 것을 봅니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | **TODO: Medium 게시 후 URL 입력** (`https://medium.com/p/...`) |
| 📘 책 원문 페이지 | [01-05-01 텐서(Tensor)란 뭘까 — 숫자 상자의 차원 여행](https://wikidocs.net/439758) |
| ✍️ 블로그 글 | [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/) (이 편의 블로그 글은 아직 없음) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`15_tensors.py`](15_tensors.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`15_tensors.ipynb`](15_tensors.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`numpy`, `torch`) |

## 🧭 코드 순서

1. 01-05-01-C — 코드로 확인하기: 0~3차원 텐서
2. 01-05-01-C — reshape: 모양만 바꾸기
3. 01-05-01-D — 프레임워크 관점: PyTorch 텐서와 자동미분

## ⚠️ 참고

- **3. 01-05-01-D — 프레임워크 관점: PyTorch 텐서와 자동미분** — `torch` 가 필요합니다. Colab에는 기본 설치되어 있습니다.

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 15_tensors.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
