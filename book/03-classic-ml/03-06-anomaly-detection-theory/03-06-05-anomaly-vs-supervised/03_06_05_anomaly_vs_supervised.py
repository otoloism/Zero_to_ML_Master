# -*- coding: utf-8 -*-
"""
03-06-05 이상탐지 vs 지도학습 — (Anomaly Detection vs Supervised Learning) - 수배 전단을 외울 것인가, 주민 얼굴을 외울 것인가

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-06장 - 이상탐지 이론/03-06-05 이상탐지 vs 지도학습.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 코드
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

rng = np.random.default_rng(0)

def make_data(n_bad_train, n_types=2):
    """
    훈련용 불량 개수(n_bad_train)와 불량 유형 수(n_types)를 바꿔가며
    데이터를 만드는 함수.

    비유:
    "고장 사례를 몇 건이나 봤는가"와
    "고장 종류가 몇 가지인가"를 조절하는 실험 손잡이 두 개.
    """
    # 불량 유형별 중심 좌표 (유형이 늘수록 사방으로 흩어짐)
    centers = [[19.5, 6.0], [9.0, 17.5],
               [20.0, 18.0], [8.0, 5.0]][:n_types]

    good = rng.normal(loc=[14.0, 12.0], scale=[1.2, 1.5], size=(8000, 2))

    # 각 유형에서 골고루 뽑아 불량 집합을 만듭니다.
    #   // 는 '몫만 남기는 나눗셈'(정수 나눗셈). 예: 7 // 2 == 3
    per = max(1, (n_bad_train + 200) // n_types)
    bad = np.vstack([
        rng.normal(loc=c, scale=[0.8, 0.8], size=(per, 2)) for c in centers
    ])
    rng.shuffle(bad)
    return good, bad

def gaussian_p(X, mu, sigma2):
    coef = 1.0 / np.sqrt(2 * np.pi * sigma2)
    return np.prod(coef * np.exp(-((X - mu) ** 2) / (2 * sigma2)), axis=1)

for n_bad in [5, 20, 100, 500]:
    good, bad = make_data(n_bad)

    # ── 공통 테스트셋 : 정상 2,000 + 불량 100 ─────────────
    X_test = np.vstack([good[6000:8000], bad[-100:]])
    y_test = np.r_[np.zeros(2000, int), np.ones(100, int)]

    # ── ① 이상탐지 : 정상만으로 학습 (불량 개수와 무관!) ────
    X_tr_anom = good[:6000]
    mu, sigma2 = X_tr_anom.mean(axis=0), X_tr_anom.var(axis=0)
    pred_anom = (gaussian_p(X_test, mu, sigma2) < 1e-4).astype(int)
    f1_anom = f1_score(y_test, pred_anom)

    # ── ② 지도학습 : 정상 + '가진 만큼의' 불량으로 학습 ────
    X_tr_sup = np.vstack([good[:6000], bad[:n_bad]])
    y_tr_sup = np.r_[np.zeros(6000, int), np.ones(n_bad, int)]

    # class_weight='balanced' : 불균형 보정. 이걸 안 하면
    #   "전부 정상"이라고만 답하는 모델이 나옵니다.
    clf = LogisticRegression(max_iter=2000, class_weight='balanced')
    clf.fit(X_tr_sup, y_tr_sup)
    f1_sup = f1_score(y_test, clf.predict(X_test))

    print(f"훈련 불량 {n_bad:4d}개 → 이상탐지 F1={f1_anom:.4f} / 지도학습 F1={f1_sup:.4f}")
