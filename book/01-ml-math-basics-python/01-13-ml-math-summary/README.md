# 01-13🤖 ML 수학 연결 총정리 — 모든 것을 이어서!

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `01-13🤖 ML 수학 연결 총정리 — 모든 것을 이어서!.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`01_13_ml_math_summary.py`](01_13_ml_math_summary.py) | 코드 블록 7개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`01_13_ml_math_summary.ipynb`](01_13_ml_math_summary.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 01-13-G 7단계 — 전체 코드: 미니 신경망으로 XOR 풀기 | 77 |
| 2 | 01-13-J 10단계 — NumPy에서 PyTorch로: 60줄이 15줄이 된다 | 20 |
| 3 | [추가·저장소 전용] 실습 8 풀이 — 죽은 뉴런 세기 (책 본문에는 문장만 실림, 10단계에서 X·T 가 torch 텐서로 바뀌었으므로 NumPy 용 X·T 를 다시 정의) | 21 |
| 4 | [추가·저장소 전용] 🐞 디버깅 팁 — 죽은 뉴런 수 한 줄 (본문 인라인 코드의 실행 예시) | 5 |
| 5 | 01-13 연습문제 Q14 — ReLU vs LeakyReLU vs He 초기화 (NumPy, lr 0.5 / 3.0) | 50 |
| 6 | 01-13 연습문제 Q14 — PyTorch 버전 | 31 |
| 7 | [추가·저장소 전용] 🔁 PyTorch에서는 처방도 한 줄 — LeakyReLU · He(Kaiming) 초기화 (10단계 모델에 적용하는 예시) | 12 |

> Block 1–2 는 기존 그대로입니다(위키독스 본문의 `L12-L88`, `L92-L111` 줄 번호 링크 유지). 2026-10-17 위키독스 개정(죽은 ReLU · 학습률 3.0 실패 · LeakyReLU/He 초기화 · 연습문제 Q14)에 맞춘 코드는 **파일 끝**에 Block 3–7 로 덧붙였습니다. Block 5·6 은 본문 Q14 코드 상자 그대로이고, `[추가·저장소 전용]` 표시가 붙은 Block 3·4·7 은 본문에 코드 상자가 없는 내용(실습 8 풀이, 본문 인라인 한 줄 코드의 실행 예시)입니다. Block 2 가 `X`, `T` 를 torch 텐서로 덮어쓰므로 Block 3 맨 앞에서 NumPy 용 `X`, `T` 를 다시 정의합니다.

## 실행 방법

```bash
cd book/01-ml-math-basics-python/01-13-ml-math-summary
pip install -r ../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 01_13_ml_math_summary.py
```

또는 `01_13_ml_math_summary.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`, torch CPU, 2026-10-17 재검증 — 약 11초)
- 블록별: 7/7 블록 성공. 참고 출력(seed 고정): 실습 8 `lr=0.3` 손실 0.0037·죽은 뉴런 3/8, `lr=3.0` 손실 1.5861·죽은 뉴런 8/8. Q14 는 시드 10개 비교라 실행에 몇 초 걸립니다.

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
