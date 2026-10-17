# #26 · 1부 ML 수학 총정리 — 순전파·손실·역전파·경사하강법 5단계를 손계산과 NumPy XOR 신경망으로 (1부)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/26-1-xor-mini-neural-net/26_1_xor_mini_neural_net.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #26 · 1부의 실습 코드입니다.
> 원문: 책 **01-13 「ML 수학 연결 총정리 — 모든 것을 이어서! (1부)」**

순전파(xW + b → ReLU → softmax) → 교차 엔트로피 손실 → 역전파(예측 − 정답에서 출발하는 책임 배분) → 경사하강 업데이트 → 300 에폭 반복, 다섯 단계를 NumPy만으로 직접 구현한 2층 미니 신경망으로 XOR 문제를 풀고(평균 손실 0.7535 → 0.0037), 학습률을 3.0으로 키우면 ReLU가 전부 죽어 학습이 멈추는 것까지 확인합니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | **TODO: Medium 게시 후 URL 입력** (`https://medium.com/p/...`) |
| 📘 책 원문 페이지 | [01-13 ML 수학 연결 총정리 — 모든 것을 이어서! (1부)](https://wikidocs.net/439749) |
| ✍️ 블로그 글 | [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/) (이 편의 블로그 글은 아직 없음) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`26_1_xor_mini_neural_net.py`](26_1_xor_mini_neural_net.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`26_1_xor_mini_neural_net.ipynb`](26_1_xor_mini_neural_net.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`numpy`) |

## 🧭 코드 순서

1. 01-13-G 7단계 — 미니 신경망으로 XOR 풀기 (NumPy)

## ⚠️ 참고

- **1. 01-13-G 7단계 — 미니 신경망으로 XOR 풀기 (NumPy)** — 책의 「🧪 직접 확인해보기 — 학습률을 3.0으로 바꾸면?」은 결과만 있어, 같은 신경망을 lr=3.0으로 다시 학습시키는 코드를 **추가**했습니다(표시된 부분).

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 26_1_xor_mini_neural_net.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
