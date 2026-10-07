# -*- coding: utf-8 -*-
"""
03-06-04 이상탐지 시스템의 구현과 평가 — (Building an Anomaly Detection System) - 정확도 99.5%가 사실은 0점인 이유

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-06장 - 이상탐지 이론/03-06-04 이상탐지 시스템의 구현과 평가.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 코드
import numpy as np

# ═══ ① 데이터 준비 — 강의의 권장 분할 그대로 ═══════════════
rng = np.random.default_rng(42)
good = rng.normal(loc=[14.0, 12.0], scale=[1.2, 1.5], size=(10000, 2))
bad = np.vstack([
    rng.normal(loc=[19.5, 6.0],  scale=[0.8, 0.8], size=(10, 2)),
    rng.normal(loc=[9.0,  17.5], scale=[0.8, 0.8], size=(10, 2)),
])
rng.shuffle(bad)

X_train = good[:6000]                       # 정상 6,000 — 라벨 없음

# np.vstack : 배열을 세로로 이어붙임 (정상 2,000 + 불량 10 = 2,010행)
X_cv = np.vstack([good[6000:8000], bad[:10]])
# np.r_ : 1차원 배열을 이어붙이는 짧은 문법.
#   앞 2,000개는 정상(0), 뒤 10개는 불량(1) 이라는 정답표를 만듭니다.
y_cv = np.r_[np.zeros(2000, int), np.ones(10, int)]

X_test = np.vstack([good[8000:10000], bad[10:]])
y_test = np.r_[np.zeros(2000, int), np.ones(10, int)]

# ═══ ② 모델 학습 (훈련셋의 정상 데이터만 사용) ══════════════
mu, sigma2 = X_train.mean(axis=0), X_train.var(axis=0)

def gaussian_p(X, mu, sigma2):
    """독립 가우시안 결합밀도 p(x) = ∏ⱼ p(xⱼ)"""
    coef = 1.0 / np.sqrt(2 * np.pi * sigma2)
    return np.prod(coef * np.exp(-((X - mu) ** 2) / (2 * sigma2)), axis=1)

p_cv = gaussian_p(X_cv, mu, sigma2)
p_test = gaussian_p(X_test, mu, sigma2)

# ═══ ③ 지표 계산 함수 ═══════════════════════════════════
def evaluate(epsilon, p, y):
    """
    주어진 ε 에서 precision / recall / F1 을 계산한다.

    비유:
    그물코 크기를 정하고 한 번 던져본 뒤,
    "물고기를 몇 마리 건졌고 나뭇가지를 몇 개 건졌나"를 세는 작업.
    """
    # (p < epsilon) 은 True/False 배열. astype(int)로 1/0 으로 바꿉니다.
    #   → 예측 라벨 : 확률이 문턱보다 낮으면 '이상(1)'
    pred = (p < epsilon).astype(int)

    # & 는 '그리고'(원소별 AND). 괄호를 꼭 써야 합니다 — 연산자 우선순위 때문!
    #   .sum() 은 True의 개수를 세어줍니다 (True=1, False=0 이므로)
    tp = ((pred == 1) & (y == 1)).sum()   # 불량을 제대로 잡음
    fp = ((pred == 1) & (y == 0)).sum()   # 정상을 잘못 잡음
    fn = ((pred == 0) & (y == 1)).sum()   # 불량을 놓침

    # [0으로 나누기 방어] 아무것도 못 잡으면 tp=0 → 분모가 0이 될 수 있음
    if tp == 0:
        return 0.0, 0.0, 0.0, (tp, fp, fn)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)  # 조화평균
    return f1, precision, recall, (tp, fp, fn)

# ═══ ④ ε 탐색 — CV 세트에서 F1 이 최대가 되는 값 ═══════════
# linspace(최솟값, 최댓값, 1000) : 후보 ε 을 1,000개 균등하게 만듭니다.
#   [1:] 로 맨 앞(최솟값)은 제외 — 그 값에서는 아무것도 안 잡히기 때문
best_f1, best_eps = 0.0, None
for eps in np.linspace(p_cv.min(), p_cv.max(), 1000)[1:]:
    f1, _, _, _ = evaluate(eps, p_cv, y_cv)
    if f1 > best_f1:
        best_f1, best_eps = f1, eps

print("=== CV 에서 고른 ε ===")
f1, prec, rec, (tp, fp, fn) = evaluate(best_eps, p_cv, y_cv)
print(f"ε = {best_eps:.6g}")
print(f"F1={f1:.4f}  precision={prec:.4f}  recall={rec:.4f}")
print(f"TP={tp}  FP={fp}  FN={fn}")

# ═══ ⑤ 최종 성능 보고 — Test 세트에서 딱 한 번 ═══════════
print("\n=== Test 세트 최종 성능 ===")
f1, prec, rec, (tp, fp, fn) = evaluate(best_eps, p_test, y_test)
print(f"F1={f1:.4f}  precision={prec:.4f}  recall={rec:.4f}")
print(f"TP={tp}  FP={fp}  FN={fn}")
