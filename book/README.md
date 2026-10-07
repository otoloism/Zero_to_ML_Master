# 📘 책 「Zero to 머신러닝 딥러닝 Master」 단원별 소스코드

[WikiDocs 책](https://wikidocs.net/book/21464)의 원고(마크다운 189개)에서 **코드 블록을 단원(절)별로 추출**해 정리한 폴더입니다.
각 절 폴더에는 다음 세 파일이 있습니다.

- `NN_NN_NN_<slug>.py` — 책의 코드 블록을 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 으로 구분, VS Code/Jupytext 에서 셀로 인식)
- `NN_NN_NN_<slug>.ipynb` — 같은 코드를 **블록당 셀 1개**로 담은 주피터 노트북
- `README.md` — 절 제목, 코드 블록 목록, 실행 방법, 검증 결과

코드는 책 원문 그대로입니다. 출력 결과·수식·의사코드·다이어그램(mermaid) 블록은 제외했고, 추출 과정에서 생긴 문제만 고쳤습니다(각 절 README 의 "추출 시 수정 사항" 참고).

## 빠른 시작

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU 버전 PyTorch
pip install -r book/requirements.txt
cd book/03-classic-ml/03-02-parameter-learning/03-02-04-numpy-pytorch-autograd
python 03_02_04_numpy_pytorch_autograd.py
```

## 권(卷)별 목록

| 권 | 제목 | 코드가 있는 절 | 코드 블록 | ✅ OK | ⏭️ 외부 요구 | 🧩 문맥 필요 | ❌ 오류 |
|---|---|---|---|---|---|---|---|
| 00 | [비전공자 처음부터 — 수포자 훑어보기](00-math-overview-for-beginners/) | 6 | 31 | 5 | 0 | 1 | 0 |
| 01 | [수포자들을 위한 머신러닝 수학 완전기초와 파이썬 맛보기](01-ml-math-basics-python/) | 17 | 176 | 14 | 0 | 3 | 0 |
| 02 | [🐍 헬로 파이썬 — 파이썬의 탄생부터 NumPy · Pandas · Plotly까지](02-hello-python/) | 9 | 140 | 8 | 0 | 0 | 1 |
| 03 | [초기(전통적) 머신러닝으로 기초 다지기](03-classic-ml/) | 62 | 353 | 46 | 3 | 13 | 0 |
| 04 | [신경망과 딥러닝 이론](04-neural-networks-deep-learning/) | 63 | 263 | 32 | 12 | 18 | 1 |
| | **합계** | **157** | **963** | **105** | **15** | **35** | **2** |

원고 189개 중 코드 블록이 없는 32개는 서문·권/장 소개·이론 절입니다. 그중 하위 절에도 코드가 없는 9개(「000 - 목차 - 이 책의 사용법!」, 03-01-01~03, 04-09장 전체)는 폴더를 만들지 않고 권 README 표에 **코드 없음**으로 표시했습니다.

## 검증 방법과 상태 표시

모든 `.py` 를 CPU 환경에서 `MPLBACKEND=Agg`(그림 창 없음), Plotly 브라우저 열기 비활성화, 제한 시간 120초(03-09-06 만 300초)로 실행했습니다.

| 표시 | 의미 |
|---|---|
| ✅ 실행 OK | 스크립트가 끝까지 오류 없이 실행됨 |
| ⏭️ 외부 요구사항 | 외부 데이터 파일(Kaggle·MovieLens 등), API 키(OpenAI·Anthropic), 대형/특수 패키지(spaCy·KoNLPy·gym·TensorFlow 등), NLTK 데이터가 필요해 검증 환경에서는 실행하지 않음 — 데이터를 지어내지 않았습니다 |
| 🧩 문맥 필요(조각) | 책이 앞 절에서 만든 변수·클래스(예: DeZero `Variable`)를 그대로 쓴다고 가정하거나, 설명용 짧은 코드 조각이라 단독 실행되지 않음 |
| ❌ 실행 오류 | 위에 해당하지 않는 오류 (의도된 오류 예시, 최신 Python 과의 차이 등 — 절 README 에 이유 기재) |

각 절 README 에는 스크립트가 멈춘 지점과, 오류가 나도 다음 블록을 계속 실행했을 때 몇 개 블록이 성공했는지도 적어 두었습니다. 모든 `.ipynb` 는 `nbformat` 검증을 통과했습니다.

### 검증 환경

Python 3.12.15 · numpy 2.5.3 · pandas 3.0.6 · matplotlib 3.11.2 · plotly 7.1.0 · scipy 1.18.1 · scikit-learn 1.9.1 · seaborn 0.13.2 · sympy 1.14.0 · torch 2.14.1+cpu · torchvision 0.29.1+cpu · xgboost 3.4.1 · lightgbm 4.7.0

## 폴더 이름 규칙

`book/<권 번호>-<영문 슬러그>/<장 번호>-<슬러그>/<절 번호>-<슬러그>/` — 책 번호(예: `03-02-04`)를 그대로 앞에 붙였습니다. 상위 원고(예: `01-05`, `03-05장`)에도 코드가 있으면 그 폴더에 해당 코드와 하위 절 폴더가 함께 있습니다.
