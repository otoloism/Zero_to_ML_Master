# 03-02-01 경사하강법 개념과 절차

> 안개 낀 산에서 발밑 기울기만 믿고 내려가는 등산객

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-02-01 경사하강법 개념과 절차.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_02_01_gradient_descent_concept.py`](03_02_01_gradient_descent_concept.py) | 코드 블록 3개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_02_01_gradient_descent_concept.ipynb`](03_02_01_gradient_descent_concept.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | cost_function.py | 36 |
| 2 | simultaneous_update.py | 32 |
| 3 | 다. 구현 문제 | 3 |

## 실행 방법

```bash
cd book/03-classic-ml/03-02-parameter-learning/03-02-01-gradient-descent-concept
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_02_01_gradient_descent_concept.py
```

또는 `03_02_01_gradient_descent_concept.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 상세: `theta0` 이(가) 이 절의 뒤쪽 블록에서야 정의됩니다 (책 서술 순서상 앞 블록은 설명용 조각).
- 스크립트가 멈춘 지점: `NameError: name 'theta0' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 1/3 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
