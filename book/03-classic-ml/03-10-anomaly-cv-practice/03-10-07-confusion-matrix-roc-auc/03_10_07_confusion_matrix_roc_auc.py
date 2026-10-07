# -*- coding: utf-8 -*-
"""
03-10-07 혼동행렬과 ROC-AUC

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-10장 - 이상탐지·교차검증 실습/03-10-07 혼동행렬과 ROC-AUC.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] ▸ 학습 목표
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score

# ── 상황 설정 ─────────────────────────────────────────────
# 1000명을 검진했는데 실제 암 환자는 10명(1%)입니다.
#   0 = 정상, 1 = 암
y_true = np.array([0] * 990 + [1] * 10)

# ── 아무 일도 안 하는 모델 ────────────────────────────────
# "전부 정상입니다"라고만 외치는 모델. 학습도 안 했습니다.
y_pred_lazy = np.zeros(1000, dtype=int)

print("【아무 일도 안 하는 모델】")
print("  정확도 :", accuracy_score(y_true, y_pred_lazy))
print("  → 99%! 훌륭해 보입니다.")
print()
print("  그런데 암 환자 10명 중 몇 명을 찾았을까요?")
print("  재현율 :", recall_score(y_true, y_pred_lazy, zero_division=0))
print("  → 0명입니다. 단 한 명도 못 찾았습니다.")
print()
print("★ 정확도 99%짜리 모델이 '암 환자를 전원 놓치는' 모델입니다.")


# %% [Block 2] ▸ 1종 오류와 2종 오류
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

# ── 유방암 데이터 ────────────────────────────────────────
#   569개 샘플, 특성 30개
#   원래 라벨: 0 = malignant(악성), 1 = benign(양성/정상)
d = load_breast_cancer()
print("클래스 이름 :", d.target_names)
print("클래스 개수 :", np.bincount(d.target), " (0=악성, 1=정상)")
print()

# ★ 중요 : 우리가 '찾으려는 것'은 악성 종양입니다.
#   그런데 원본 데이터는 악성이 0으로 되어 있습니다.
#   지표를 올바로 읽으려면 악성을 1(Positive)로 뒤집어야 합니다.
X = d.data
y = 1 - d.target        # 0↔1 뒤집기 → 1 = 악성(찾을 대상)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# make_pipeline : 스케일러와 모델을 하나로 묶습니다 (누수 방지)
# max_iter=5000 : 로지스틱 회귀가 수렴할 때까지 반복 횟수를 넉넉히
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
model.fit(X_tr, y_tr)

pred = model.predict(X_te)
proba = model.predict_proba(X_te)[:, 1]   # 악성일 확률

cm = confusion_matrix(y_te, pred)
print("혼동행렬 (행=실제, 열=예측)")
print(cm)
print()
# .ravel() : 2차원 배열을 1차원으로 펴 줍니다. 순서는 TN, FP, FN, TP
tn, fp, fn, tp = cm.ravel()
print(f"  TN(정상을 정상이라 함) : {tn:>3}   FP(정상을 악성이라 함, 헛경보) : {fp:>3}")
print(f"  FN(악성을 정상이라 함, 놓침) : {fn:>3}   TP(악성을 악성이라 함) : {tp:>3}")


# %% [Block 3] ▸ F1은 왜 산술평균이 아니라 조화평균인가 — 정밀도 1.0, 재현율 0.01인 극단적 모델을 생각해 봅시다.
# 정밀도 1.0, 재현율 0.01인 극단적 모델을 생각해 봅시다.
#   → "가장 확실한 것 하나만 찍고 나머지는 다 포기"하는 모델
P, R = 1.0, 0.01

arith = (P + R) / 2                  # 산술평균
harm = 2 * P * R / (P + R)           # 조화평균 = F1

print(f"정밀도 {P}, 재현율 {R} 인 모델")
print(f"  산술평균 : {arith:.4f}  ← 0.5나 됩니다. 절반은 하는 것처럼 보입니다.")
print(f"  조화평균 : {harm:.4f}  ← 0.02. 쓸모없다는 게 정직하게 드러납니다.")
print()
print("다양한 조합에서 비교")
print(f"{'정밀도':>8} | {'재현율':>8} | {'산술평균':>9} | {'F1(조화)':>9}")
print("-" * 44)
for p, r in [(1.0, 0.01), (0.9, 0.1), (0.7, 0.5), (0.6, 0.6), (0.5, 0.7), (0.95, 0.95)]:
    print(f"{p:>8.2f} | {r:>8.2f} | {(p+r)/2:>9.4f} | {2*p*r/(p+r):>9.4f}")
print()
print("★ 조화평균은 '둘 다 잘해야' 높습니다. 한쪽만 잘하면 봐주지 않습니다.")
print("  그래서 F1은 '균형'을 재는 지표입니다.")


# %% [Block 4] ▸ F1은 왜 산술평균이 아니라 조화평균인가
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)

print("【악성(1)을 Positive로 놓았을 때】")
print(f"  정확도  : {accuracy_score(y_te, pred):.4f}")
print(f"  정밀도  : {precision_score(y_te, pred):.4f}"
      f"   = TP/(TP+FP) = {tp}/({tp}+{fp})")
print(f"  재현율  : {recall_score(y_te, pred):.4f}"
      f"   = TP/(TP+FN) = {tp}/({tp}+{fn})")
print(f"  F1      : {f1_score(y_te, pred):.4f}")
print()
print("classification_report — 클래스별로 한 번에 보기")
print(classification_report(y_te, pred,
                            target_names=['정상(0)', '악성(1)'], digits=4))


# %% [Block 5] ▸ F-베타 — 균형을 직접 조절하기
from sklearn.metrics import fbeta_score

print(f"{'β':>6} | {'점수':>8} | 의미")
print("-" * 46)
for b in [0.25, 0.5, 1.0, 2.0, 4.0]:
    s = fbeta_score(y_te, pred, beta=b)
    if b < 1:
        m = "정밀도 중시 (스팸 필터형)"
    elif b == 1:
        m = "동등 (= F1)"
    else:
        m = "재현율 중시 (암 진단형)"
    print(f"{b:>6} | {s:>8.4f} | {m}")

print()
print("이 모델은 정밀도와 재현율이 거의 같아서 β를 바꿔도 점수가 비슷합니다.")
print("→ 균형이 잘 잡힌 모델이라는 뜻입니다.")


# %% [Block 6] ▸ F-베타 — 균형을 직접 조절하기
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

print(f"{'임계값 t':>9} | {'TN':>4} {'FP':>4} {'FN':>4} {'TP':>4} | "
      f"{'정밀도':>8} | {'재현율':>8} | {'F1':>8}")
print("-" * 72)

for t in [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]:
    # proba(악성일 확률)가 t 이상이면 1(악성)로 판정
    #   .astype(int) : True/False 를 1/0 으로 바꿉니다.
    p_t = (proba >= t).astype(int)
    tn_, fp_, fn_, tp_ = confusion_matrix(y_te, p_t).ravel()
    pr = precision_score(y_te, p_t, zero_division=0)
    rc = recall_score(y_te, p_t)
    f1 = f1_score(y_te, p_t)
    star = "  ← predict()의 기본값" if t == 0.5 else ""
    print(f"{t:>9.2f} | {tn_:>4} {fp_:>4} {fn_:>4} {tp_:>4} | "
          f"{pr:>8.4f} | {rc:>8.4f} | {f1:>8.4f}{star}")

print()
print("★ 임계값을 내리면(왼쪽 위→) 재현율 ↑ 정밀도 ↓")
print("★ 임계값을 올리면(아래로→) 정밀도 ↑ 재현율 ↓")
print("★ 모델은 하나도 안 바뀌었습니다. 자르는 위치만 바뀌었습니다.")


# %% [Block 7] ▸ 곡선이 그려지는 과정
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, roc_auc_score

# ── 일부러 '약한 모델'을 만듭니다 ──────────────────────────
#   특성 30개를 다 쓰면 거의 완벽해서(AUC 0.998) 곡선 모양이 안 보입니다.
#   구별력이 약한 특성 2개만 골라 곡선의 모양을 관찰합니다.
d = load_breast_cancer()
names = list(d.feature_names)
idx = [names.index('mean texture'), names.index('mean smoothness')]

Xw = d.data[:, idx]
yw = 1 - d.target                       # 악성 = 1

Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(
    Xw, yw, test_size=0.3, random_state=42, stratify=yw)

weak = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
weak.fit(Xw_tr, yw_tr)
pw = weak.predict_proba(Xw_te)[:, 1]

# roc_curve(정답, 확률) → (FPR 배열, TPR 배열, 임계값 배열)
fpr, tpr, thr = roc_curve(yw_te, pw)

print("ROC 곡선을 이루는 점의 개수 :", len(fpr))
print()
print(f"{'임계값':>10} | {'FPR(헛경보율)':>13} | {'TPR(재현율)':>12}")
print("-" * 42)
# np.linspace(0, 끝, 10) : 0부터 끝까지 10개 지점을 균등하게 고릅니다.
for i in np.linspace(0, len(fpr) - 1, 10).astype(int):
    t = thr[i]
    ts = "  inf" if np.isinf(t) else f"{t:.4f}"
    print(f"{ts:>10} | {fpr[i]:>13.4f} | {tpr[i]:>12.4f}")

print()
print("★ 임계값을 높은 곳(inf)에서 낮은 곳(0)으로 내리면서")
print("  (FPR, TPR) 점을 찍어 나간 것이 ROC 곡선입니다.")
print("  왼쪽 아래 (0,0)에서 출발해 오른쪽 위 (1,1)에 도착합니다.")
print()
print("AUC :", round(float(roc_auc_score(yw_te, pw)), 4))


# %% [Block 8] ▸ 직접 곡선을 재현해 보기
import numpy as np

# roc_curve가 어떻게 계산되는지 직접 구현해 봅니다.
def my_roc(y_true, scores):
    """확률 점수를 내림차순으로 훑으며 (FPR, TPR) 궤적을 만듭니다."""
    # 점수가 높은 순서로 정렬 (argsort는 오름차순이라 [::-1]로 뒤집습니다)
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]

    P = y_true.sum()            # 실제 양성 총 개수
    N = len(y_true) - P         # 실제 음성 총 개수

    tp = fp = 0
    fpr_list, tpr_list = [0.0], [0.0]
    for yi in y_sorted:
        if yi == 1:
            tp += 1             # 진짜 양성을 하나 더 건졌다 → 위로
        else:
            fp += 1             # 헛경보 하나 → 오른쪽으로
        tpr_list.append(tp / P)
        fpr_list.append(fp / N)
    return np.array(fpr_list), np.array(tpr_list)

my_fpr, my_tpr = my_roc(yw_te, pw)

# 사다리꼴 공식으로 곡선 아래 면적을 계산합니다.
#   np.trapezoid : 사다리꼴 적분 (numpy 2.0부터 이름이 trapz→trapezoid)
try:
    my_auc = np.trapezoid(my_tpr, my_fpr)
except AttributeError:
    my_auc = np.trapz(my_tpr, my_fpr)

print("직접 계산한 AUC   :", round(float(my_auc), 4))
print("사이킷런 AUC      :", round(float(roc_auc_score(yw_te, pw)), 4))
print("두 값이 같은가?   :", abs(my_auc - roc_auc_score(yw_te, pw)) < 1e-6)
print()
print("★ ROC 곡선은 '점수 높은 순으로 하나씩 꺼내며'")
print("  정답이면 위로, 오답이면 오른쪽으로 한 칸씩 간 궤적입니다.")


# %% [Block 9] ▸ 직접 곡선을 재현해 보기
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thr = roc_curve(yw_te, pw)
auc = roc_auc_score(yw_te, pw)

plt.figure(figsize=(6, 6))
# 곡선 그리기
plt.plot(fpr, tpr, lw=2.5, label=f'ROC curve (AUC = {auc:.4f})')
# 무작위 분류기 대각선
plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()


# %% [Block 10] ▸ AUC의 확률적 해석 — 가장 중요한 사실
import numpy as np
from sklearn.metrics import roc_auc_score

# ── 확률적 해석을 직접 검증해 봅시다 ──────────────────────
pos_scores = pw[yw_te == 1]     # 악성 샘플들의 점수
neg_scores = pw[yw_te == 0]     # 정상 샘플들의 점수

print("악성 샘플 :", len(pos_scores), "개")
print("정상 샘플 :", len(neg_scores), "개")
print()

# 모든 (악성, 정상) 쌍을 만들어 비교합니다.
#   pos_scores[:, None] → 세로 벡터 (n_pos, 1)
#   neg_scores[None, :] → 가로 벡터 (1, n_neg)
#   빼면 브로드캐스팅으로 모든 쌍의 차이 (n_pos, n_neg)
diff = pos_scores[:, None] - neg_scores[None, :]

win = (diff > 0).mean()          # 악성이 이긴 비율
tie = (diff == 0).mean()         # 동점 비율

print("모든 (악성, 정상) 쌍의 개수 :", diff.size)
print("악성의 점수가 더 높은 쌍 비율 :", round(float(win), 6))
print("동점인 쌍 비율                :", round(float(tie), 6))
print()
# 동점은 절반씩 나눠 갖는 것이 AUC의 정의입니다.
manual = win + tie / 2
print("직접 계산한 확률 (동점은 0.5로) :", round(float(manual), 6))
print("사이킷런 roc_auc_score          :", round(float(roc_auc_score(yw_te, pw)), 6))
print()
print("★ 두 값이 정확히 같습니다.")
print("  AUC는 '아무 악성과 아무 정상을 뽑았을 때 악성 점수가 더 높을 확률'입니다.")


# %% [Block 11] ▸ 이상탐지 세 모델을 AUC로 다시 비교하기
import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

# 03-10-01절의 실험 데이터를 다시 만듭니다.
rng = np.random.RandomState(42)
Xtr = 0.2 * rng.randn(1000, 2); Xtr = np.r_[Xtr + 3, Xtr]
Xte = 0.2 * rng.randn(200, 2); Xte = np.r_[Xte + 3, Xte]
outl = rng.uniform(low=-1, high=5, size=(50, 2))

# 검증용 정상 400개 + 이상치 50개를 합쳐 정답표를 만듭니다.
X_eval = np.r_[Xte, outl]
y_eval = np.r_[np.zeros(len(Xte)), np.ones(len(outl))]   # 1 = 이상치(Positive)

models = [
    ("EllipticEnvelope", EllipticEnvelope(contamination=0.1, random_state=42)),
    ("LocalOutlierFactor", LocalOutlierFactor(n_neighbors=20, novelty=True,
                                              contamination=0.1)),
    ("IsolationForest", IsolationForest(contamination=0.1, random_state=42)),
]

print(f"{'모델':>20} | {'이상치 검출율':>12} | {'ROC-AUC':>9} | {'PR-AUC':>8}")
print("-" * 60)
for name, m in models:
    m.fit(Xtr)
    # 검출율 (contamination=0.1 임계값 기준)
    det = (m.predict(outl) == -1).mean()
    # 이상 점수 : score_samples는 '클수록 정상'이므로 마이너스를 붙입니다.
    s = -m.score_samples(X_eval)
    auc = roc_auc_score(y_eval, s)
    ap = average_precision_score(y_eval, s)
    print(f"{name:>20} | {det:>12.4f} | {auc:>9.4f} | {ap:>8.4f}")

print()
print("★★ 순위가 뒤집혔습니다 ★★")
print("  검출율 기준 : IsolationForest(0.98) > LOF(0.96) > EllipticEnvelope(0.82)")
print("  AUC   기준 : LOF(0.9811) > IsolationForest(0.9800) > EllipticEnvelope(0.8795)")
print()
print("  검출율은 'contamination=0.1'이라는 특정 임계값 하나에서의 성적표입니다.")
print("  AUC는 '모든 임계값을 통틀어' 순위를 얼마나 잘 매겼는지를 봅니다.")
print("  → 03-10-04절에서 'IsolationForest가 최고'라고 성급히 결론 내리지 말라고")
print("    했던 이유가 바로 이것입니다.")


# %% [Block 12] ▸ ROC-AUC가 실패하는 경우 — PR 곡선
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

rngX = np.random.RandomState(0)

# ── 극단적 불균형 상황을 만듭니다 ──────────────────────────
# 정상 9990개, 이상 10개 (0.1%)
n_neg, n_pos = 9990, 10
y_imb = np.r_[np.zeros(n_neg), np.ones(n_pos)]

# 그럭저럭 하는 모델 : 이상 샘플의 점수가 평균적으로 조금 높음
s_imb = np.r_[rngX.randn(n_neg), rngX.randn(n_pos) + 2.2]

auc = roc_auc_score(y_imb, s_imb)
ap = average_precision_score(y_imb, s_imb)

print("클래스 비율 : 정상", n_neg, ": 이상", n_pos,
      f" ({n_pos/(n_pos+n_neg):.2%})")
print()
print("ROC-AUC :", round(float(auc), 4), " ← 꽤 좋아 보입니다")
print("PR-AUC  :", round(float(ap), 4), " ← 현실은 훨씬 냉정합니다")
print()

# 상위 100개를 이상이라 지목했을 때 실제 성적
top100 = np.argsort(s_imb)[-100:]
hit = int((top100 >= n_neg).sum())
print(f"점수 상위 100개를 이상이라 지목하면 : 진짜 이상 {hit}개 / 100개")
print(f"  → 정밀도 {hit/100:.2f}  (100번 경보 중 {100-hit}번이 헛경보)")
print()
print("★ ROC-AUC가 높아도 실무에서는 헛경보에 파묻힐 수 있습니다.")
print("  분모가 '실제 음성 전체'라 FP 몇 개가 FPR에 거의 영향을 못 주기 때문입니다.")
print("  → 양성이 1% 미만이면 PR 곡선(average_precision_score)을 함께 보세요.")


# %% [Block 13] ▸ 프레임워크 비교
import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix

# ── 실습 1 정답 : 재현율 0.99 이상 중 최고 정밀도 ──────────
prec, rec, thresholds = precision_recall_curve(y_te, proba)

# prec, rec은 thresholds보다 원소가 1개 많습니다(마지막에 (1,0) 추가).
#   비교하려면 앞쪽 len(thresholds)개만 씁니다.
ok = rec[:-1] >= 0.99
if ok.any():
    best_i = np.argmax(np.where(ok, prec[:-1], -1))
    print("【실습 1】 재현율 0.99 이상을 만족하는 임계값")
    print(f"  임계값 : {thresholds[best_i]:.6f}")
    print(f"  정밀도 : {prec[best_i]:.4f}")
    print(f"  재현율 : {rec[best_i]:.4f}")
    p_best = (proba >= thresholds[best_i]).astype(int)
    tn_, fp_, fn_, tp_ = confusion_matrix(y_te, p_best).ravel()
    print(f"  혼동행렬 : TN={tn_} FP={fp_} FN={fn_} TP={tp_}")
    print(f"  → 악성 {tp_+fn_}명 중 {tp_}명을 찾고 {fn_}명을 놓칩니다.")
    print(f"     대신 정상 {tn_+fp_}명 중 {fp_}명이 헛경보를 받습니다.")
print()

# ── 실습 2 정답 : F1이 최대가 되는 임계값 ─────────────────
f1s = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
bi = int(np.argmax(f1s))
print("【실습 2】 F1이 최대가 되는 지점")
print(f"  임계값 : {thresholds[bi]:.6f}")
print(f"  정밀도 : {prec[bi]:.4f}   재현율 : {rec[bi]:.4f}   F1 : {f1s[bi]:.4f}")
print(f"  (참고) 기본 임계값 0.5에서의 F1 : "
      f"{2*precision_score(y_te,pred)*recall_score(y_te,pred)/(precision_score(y_te,pred)+recall_score(y_te,pred)):.4f}")


# %% [Block 14] ▸ 프레임워크 비교
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

# ── 실습 3 정답 : Positive를 뒤집으면 ──────────────────────
print("【실습 3】 무엇을 Positive로 놓느냐에 따라")
print(f"{'Positive':>14} | {'정밀도':>8} | {'재현율':>8} | {'ROC-AUC':>9}")
print("-" * 48)
print(f"{'악성(1=악성)':>14} | {precision_score(y_te, pred):>8.4f} | "
      f"{recall_score(y_te, pred):>8.4f} | {roc_auc_score(y_te, proba):>9.4f}")
# 라벨과 예측을 모두 뒤집으면 = 정상을 Positive로 본 것
print(f"{'정상(1=정상)':>14} | {precision_score(1-y_te, 1-pred):>8.4f} | "
      f"{recall_score(1-y_te, 1-pred):>8.4f} | "
      f"{roc_auc_score(1-y_te, 1-proba):>9.4f}")
print()
print("→ 정밀도·재현율은 완전히 다른 값이 됩니다.")
print("  AUC는 같습니다 (양쪽을 대칭적으로 보는 지표라서).")
print()

# ── 실습 4 정답 : 교차검증으로 AUC 재기 ───────────────────
print("【실습 4】 5-폴드 교차검증 ROC-AUC")
cv_auc = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True,
                                                          random_state=42),
                         scoring='roc_auc')
print("  폴드별 AUC :", np.round(cv_auc, 4))
print(f"  평균 ± 표준편차 : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
print()
print("  → 단일 홀드아웃 AUC 하나만 보고 모델을 고르면 안 됩니다.")
