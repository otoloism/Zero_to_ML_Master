# 참고 — 우리가 NumPy로 손수 만든 코드 A의 결과: 2.009904

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `00-04미분·경사하강법— 안개 낀 산에서 내려오기.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`00_04_derivative_gradient_descent.py`](00_04_derivative_gradient_descent.py) | 코드 블록 9개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`00_04_derivative_gradient_descent.ipynb`](00_04_derivative_gradient_descent.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 📍 1단계 — 미분(Derivative)이란 무엇인가? | 25 |
| 2 | 📍 2단계 — 미분 공식과 연쇄법칙 | 53 |
| 3 | 📍 2단계 — 미분 공식과 연쇄법칙 | 33 |
| 4 | 📍 3단계 — 편미분과 그래디언트 | 48 |
| 5 | 📍 5단계 — 코드로 구현하고 실전에 적용하기 | 36 |
| 6 | 📍 5단계 — 코드로 구현하고 실전에 적용하기 — 4단계에서 유도한 조건:  \|1 - 2η\| < 1  ⟺  0 < η < 1 | 15 |
| 7 | 📍 5단계 — 코드로 구현하고 실전에 적용하기 | 14 |
| 8 | 📍 5단계 — 코드로 구현하고 실전에 적용하기 | 31 |
| 9 | 📍 6단계 — 프레임워크 연결: 자동미분 — ① PyTorch | 42 |

## 실행 방법

```bash
cd book/00-math-overview-for-beginners/00-04-derivative-gradient-descent
pip install -r ../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 00_04_derivative_gradient_descent.py
```

또는 `00_04_derivative_gradient_descent.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`, 2026-10-17 재검증)
- 상세: `Function` 이(가) 이 절 안에서 정의되지 않습니다 — 데이터·변수·클래스를 미리 준비했다고 가정한 설명용 코드 조각입니다.
- 스크립트가 멈춘 지점: `NameError: name 'Function' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 7/9 블록 성공

## 추출 시 수정 사항

- 위키독스 페이지([wikidocs.net/439840](https://wikidocs.net/439840))의 코드 상자를 그대로 옮겼습니다 — 블록 1개 = 코드 상자 1개, 순서·내용 동일 (2026-10-17 페이지 기준 정렬). 출력 결과·mermaid 다이어그램·글자 표(ASCII) 상자는 코드 상자로 세지 않았습니다.
- 예전에 넣었던 보충 import(`Block 0 [추가]`), 주석 처리, 오타 수정은 페이지와 1:1로 맞추기 위해 되돌렸습니다.

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
