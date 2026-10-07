# 03-02-04 NumPy/PyTorch 오토그라드·옵티마이저 실습

> 미분을 대신 계산해주는 자동 계산기

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-02-04 Numpy PyTorch 오토그라드·옵티마이저 실습.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_02_04_numpy_pytorch_autograd.py`](03_02_04_numpy_pytorch_autograd.py) | 코드 블록 14개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_02_04_numpy_pytorch_autograd.ipynb`](03_02_04_numpy_pytorch_autograd.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | linear_regression_numpy.py | 59 |
| 2 | sanity_check.py — 내 구현이 맞는지 검산하기 | 11 |
| 3 | autograd.py | 32 |
| 4 | optimizer.py | 28 |
| 5 | loss_fn.py | 25 |
| 6 | manual_model.py | 32 |
| 7 | train_with_model.py | 29 |
| 8 | nested_and_sequential.py | 19 |
| 9 | train_step.py | 41 |
| 10 | use_train_step.py | 15 |
| 11 | custom_dataset.py | 41 |
| 12 | dataloader.py | 33 |
| 13 | random_split.py | 22 |
| 14 | 다. 구현 문제 — no_grad() : 검증에서는 기울기가 필요 없으므로 녹화를 끕니다. | 17 |

## 실행 방법

```bash
cd book/03-classic-ml/03-02-parameter-learning/03-02-04-numpy-pytorch-autograd
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_02_04_numpy_pytorch_autograd.py
```

또는 `03_02_04_numpy_pytorch_autograd.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 상세: `x_train` 이(가) 이 절 안에서 정의되지 않습니다 — 데이터·변수·클래스를 미리 준비했다고 가정한 설명용 코드 조각입니다.
- 스크립트가 멈춘 지점: `NameError: name 'x_train' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 2/14 블록 성공

## 추출 시 수정 사항

- Block 0 추가: `import torch` — 책에서는 앞 절에서 import 했거나 뒤 블록에서야 import 하는 모듈이라, 단독 실행을 위해 보충했습니다.

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
