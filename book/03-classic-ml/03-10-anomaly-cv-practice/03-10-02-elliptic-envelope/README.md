# 03-10-02 EllipticEnvelope — 타원 울타리

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-10-02 EllipticEnvelope — 타원 울타리.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_10_02_elliptic_envelope.py`](03_10_02_elliptic_envelope.py) | 코드 블록 11개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_10_02_elliptic_envelope.ipynb`](03_10_02_elliptic_envelope.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | ▸ 숫자로 확인하기 | 32 |
| 2 | ▸ 실험 — MLE와 MCD는 얼마나 다른가 | 33 |
| 3 | ▸ 실험 — MLE와 MCD는 얼마나 다른가 | 23 |
| 4 | ▸ 실험 — MLE와 MCD는 얼마나 다른가 | 27 |
| 5 | ▸ 정확도 재현 — 강의 자료 13p — list(...)로 감싸는 이유 : 넘파이 배열에는 .count() 메서드가 없기 때문입니다. | 6 |
| 6 | ▸ 결과 시각화 — 강의 자료 12p | 26 |
| 7 | ▸ 결과 시각화 — 강의 자료 12p | 21 |
| 8 | ▸ 결정적 증거 — 아무것도 없는 자리를 물어보기 | 34 |
| 9 | ▸ 수학적으로 왜 이런 일이 벌어지는가 | 43 |
| 10 | ▸ 프레임워크 비교 | 19 |
| 11 | ▸ 프레임워크 비교 | 28 |

## 실행 방법

```bash
cd book/03-classic-ml/03-10-anomaly-cv-practice/03-10-02-elliptic-envelope
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_10_02_elliptic_envelope.py
```

또는 `03_10_02_elliptic_envelope.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: 앞 절 03-10-01 의 코드 실행 결과 (`outliers` 등) — 책이 '앞 절에서 만든 것을 그대로 쓴다'고 가정
- 스크립트가 멈춘 지점: `NameError: name 'outliers' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 3/11 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
