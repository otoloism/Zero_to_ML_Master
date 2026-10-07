# 04-06-06 신경망 —PyTorch기초

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-06-06 신경망 —PyTorch기초.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_06_06_pytorch_basics.py`](04_06_06_pytorch_basics.py) | 코드 블록 6개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_06_06_pytorch_basics.ipynb`](04_06_06_pytorch_basics.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-06-06-B 🟢 초급 ⭐ · PyTorch 텐서(Tensor) | 18 |
| 2 | 04-06-06-C 🔵 중급 ⭐ · 자동 미분(Autograd) | 12 |
| 3 | 04-06-06-D 🔵 중급 · 선형 회귀를 PyTorch로 | 26 |
| 4 | 04-06-06-E 🔵 중급 · 옵티마이저 (torch.optim.Adam) | 15 |
| 5 | 04-06-06-F 🔵 중급 ⭐ · nn.Module — 신경망을 클래스로 | 25 |
| 6 | 04-06-06-G 🔴 심화 🚀 · torchvision과 MNIST 분류 전체 파이프라인 | 31 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-06-generative-models/04-06-06-pytorch-basics
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_06_06_pytorch_basics.py
```

또는 `04_06_06_pytorch_basics.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: 인터넷 연결 (torchvision 이 MNIST 데이터셋을 자동 다운로드)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
