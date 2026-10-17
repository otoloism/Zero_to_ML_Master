# #26 · 2부 NumPy 60줄이 PyTorch 15줄로 — 신경망 수학 오해 바로잡기와 GPT·CNN·확산 모델 속 같은 수학 (2부)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/26-2-numpy-to-pytorch/26_2_numpy_to_pytorch.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #26 · 2부의 실습 코드입니다.
> 원문: 책 **01-13 「ML 수학 연결 총정리 — 모든 것을 이어서! (2부)」**

1부에서 NumPy 60줄로 손수 짠 XOR 신경망을 PyTorch 15줄로 옮깁니다. nn.Sequential(층 쌓기 = 합성함수), nn.CrossEntropyLoss(softmax + 교차 엔트로피), loss.backward()(손으로 짠 역전파 5줄), optimizer.step()(W -= lr * gW)이 각각 무엇을 대신하는지 확인하고, 학습 후 4문제를 모두 맞히는지 출력해 봅니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | **TODO: Medium 게시 후 URL 입력** (`https://medium.com/p/...`) |
| 📘 책 원문 페이지 | [01-13 ML 수학 연결 총정리 — 모든 것을 이어서! (2부)](https://wikidocs.net/439749) |
| ✍️ 블로그 글 | [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/) (이 편의 블로그 글은 아직 없음) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`26_2_numpy_to_pytorch.py`](26_2_numpy_to_pytorch.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`26_2_numpy_to_pytorch.ipynb`](26_2_numpy_to_pytorch.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`numpy`, `torch`) |

## 🧭 코드 순서

1. 01-13-J 10단계 — 같은 신경망을 PyTorch로

## ⚠️ 참고

- **1. 01-13-J 10단계 — 같은 신경망을 PyTorch로** — 책의 PyTorch 코드는 출력이 없어, 재현용 시드 고정(앞)과 최종 손실·예측 확인 출력(뒤)을 **추가**했습니다(표시된 부분).

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 26_2_numpy_to_pytorch.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
