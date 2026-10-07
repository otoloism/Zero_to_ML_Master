# -*- coding: utf-8 -*-
"""
03-10-02 EllipticEnvelope — 타원 울타리

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-10장 - 이상탐지·교차검증 실습/03-10-02 EllipticEnvelope — 타원 울타리.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] ▸ 숫자로 확인하기
import numpy as np

# 데이터가 이런 공분산으로 퍼져 있다고 합시다.
#   [[4, 3],
#    [3, 4]]
#   → 대각선 원소 4, 4  : x1과 x2 각각의 분산(퍼짐)
#   → 비대각 원소 3, 3  : x1이 커질 때 x2도 같이 커진다는 뜻(양의 상관)
Sigma = np.array([[4.0, 3.0],
                  [3.0, 4.0]])

# np.linalg.inv : 역행렬을 구합니다. (linalg = linear algebra, 선형대수)
Sigma_inv = np.linalg.inv(Sigma)

# 중심(평균)은 원점 (0, 0)이라고 두겠습니다.
point_A = np.array([3.0,  3.0])   # 퍼진 방향(대각선)으로 멀어진 점
point_B = np.array([3.0, -3.0])   # 좁은 방향(반대 대각선)으로 멀어진 점

for name, p in [("A (퍼진 방향)", point_A), ("B (좁은 방향)", point_B)]:
    # 유클리드 거리 : 그냥 피타고라스. √(x1² + x2²)
    #   p @ p  는 내적(dot product). 여기서는 x1² + x2² 와 같습니다.
    euclid = np.sqrt(p @ p)

    # 마할라노비스 거리 : 가운데에 Σ⁻¹를 끼워 넣습니다.
    #   p @ Sigma_inv @ p  →  pᵀ Σ⁻¹ p
    maha = np.sqrt(p @ Sigma_inv @ p)

    print(f"{name:>14} 좌표 {p}  유클리드 {euclid:.4f}   마할라노비스 {maha:.4f}")

print()
print("→ 유클리드는 두 점을 '똑같이 멀다'고 말합니다. (둘 다 4.2426)")
print("→ 마할라노비스는 B가 A보다 약 2.6배 멀다고 말합니다.")
print("   이것이 우리 눈이 보는 것과 일치합니다.")


# %% [Block 2] ▸ 실험 — MLE와 MCD는 얼마나 다른가
import numpy as np
from sklearn.covariance import EmpiricalCovariance, MinCovDet

rng2 = np.random.RandomState(7)

# ① 깨끗한 정상 데이터 120개
#    multivariate_normal(평균, 공분산, 개수) : 다변량 정규분포에서 뽑기
#    참값: 중심 [0,0],  공분산 [[4,3],[3,4]]  (대각선 방향으로 길쭉한 모양)
clean = rng2.multivariate_normal([0, 0], [[4, 3], [3, 4]], 120)

# ② 이상치 30개 — 넓은 범위에 균등하게 흩뿌립니다.
bad = rng2.uniform(-8, 8, (30, 2))

# ③ 둘을 섞습니다. 전체 150개 중 20%가 오염된 상태입니다.
X_mix = np.r_[clean, bad]

# ── 추정 방법 두 가지 ──────────────────────────────────────
# EmpiricalCovariance : 그냥 전부 다 써서 평균·공분산을 구합니다 (= MLE, 최대가능도추정)
mle = EmpiricalCovariance().fit(X_mix)
# MinCovDet : 가장 촘촘한 부분집합만 골라 구합니다 (= MCD, 강건 추정)
mcd = MinCovDet(random_state=42).fit(X_mix)

print("참값        중심 [0. 0.]   공분산 [[4. 3.] [3. 4.]]")
print("MLE  추정   중심", np.round(mle.location_, 3),
      "  공분산", np.round(mle.covariance_, 3).tolist())
print("MCD  추정   중심", np.round(mcd.location_, 3),
      "  공분산", np.round(mcd.covariance_, 3).tolist())
print()
print("★ 주목할 곳 : 공분산의 '비대각 원소'(원래 3이어야 함)")
print("   MLE :", round(mle.covariance_[0, 1], 3),
      "→ 이상치가 상관관계를 거의 지워 버렸습니다!")
print("   MCD :", round(mcd.covariance_[0, 1], 3),
      "→ 원래 모양을 거의 그대로 살렸습니다.")


# %% [Block 3] ▸ 실험 — MLE와 MCD는 얼마나 다른가
import numpy as np
from sklearn.covariance import EmpiricalCovariance, MinCovDet

rng2 = np.random.RandomState(7)
clean = rng2.multivariate_normal([0, 0], [[4, 3], [3, 4]], 120)
bad = rng2.uniform(-8, 8, (30, 2))
X_mix = np.r_[clean, bad]
# 앞 120개가 정상, 뒤 30개가 이상치라는 사실을 우리는 알고 있습니다(정답표).

for name, est in [("MLE(전부 사용)", EmpiricalCovariance().fit(X_mix)),
                  ("MCD(강건 추정)", MinCovDet(random_state=42).fit(X_mix))]:
    # mahalanobis()는 '제곱된' 거리를 돌려줍니다.
    d2 = est.mahalanobis(X_mix)

    # np.argsort : 값을 작은 순으로 정렬했을 때의 '원래 위치 번호'를 돌려줍니다.
    #   [-30:] → 뒤에서 30개 = 거리가 가장 먼 30개의 위치
    far30 = np.argsort(d2)[-30:]

    # 위치 번호가 120 이상이면 진짜 이상치입니다.
    hit = (far30 >= 120).sum()

    print(f"{name} : 가장 먼 30개 중 진짜 이상치 {hit}개  "
          f"(검출율 {hit/30:.1%})")


# %% [Block 4] ▸ 실험 — MLE와 MCD는 얼마나 다른가
from sklearn.covariance import EllipticEnvelope

# .copy() : 원본을 건드리지 않으려고 사본을 만듭니다.
#   (뒤에서 X_outliers에 열을 추가할 건데, 원본 outliers는 그대로 두고 싶기 때문)
#   🔗 파이썬의 '참조 복사 vs 값 복사' — 02권절 메모리 단원
X_outliers = outliers.copy()

# ── 모델 만들기 ────────────────────────────────────────────
#  contamination=0.1 : 훈련 데이터 중 하위 10%를 이상으로 보겠다는 임계값 설정
#  random_state=42   : MCD가 부분집합을 랜덤하게 탐색하므로 씨앗을 고정합니다
clf = EllipticEnvelope(contamination=0.1, random_state=42)

# ── 학습 ──────────────────────────────────────────────────
#  X_train에는 '정상만' 들어 있습니다 → 노벨티 탐지 구도
#  이 한 줄 안에서 벌어지는 일:
#    ① MCD로 중심 μ와 공분산 Σ를 강건하게 추정
#    ② 모든 훈련 데이터의 마할라노비스 거리를 계산
#    ③ 그 거리의 상위 10% 지점을 임계값 τ로 저장 (offset_)
clf.fit(X_train)

# ── 판정 ──────────────────────────────────────────────────
y_pred_train    = clf.predict(X_train)      # 훈련 데이터
y_pred_test     = clf.predict(X_test)       # 새 정상 손님  → +1이 정답
y_pred_outliers = clf.predict(X_outliers)   # 이상치        → -1이 정답

print("predict가 돌려주는 값의 종류 :", np.unique(y_pred_test))
print("검증 데이터 앞 10개 판정      :", y_pred_test[:10])


# %% [Block 5] ▸ 정확도 재현 — 강의 자료 13p — list(...)로 감싸는 이유 : 넘파이 배열에는 .count() 메서드가 없기 때문입니다.
# list(...)로 감싸는 이유 : 넘파이 배열에는 .count() 메서드가 없기 때문입니다.
#   (파이썬 리스트에만 있는 기능입니다)
print("테스트 데이터셋에서 정확도:",
      list(y_pred_test).count(1) / y_pred_test.shape[0])
print("이상치 데이터셋에서 정확도:",
      list(y_pred_outliers).count(-1) / y_pred_outliers.shape[0])


# %% [Block 6] ▸ 결과 시각화 — 강의 자료 12p
import matplotlib.pyplot as plt

# .assign(y = ...) : DataFrame에 'y'라는 새 열을 붙인 사본을 돌려줍니다.
#   원본을 바꾸지 않고 새 DataFrame을 만드는 '체이닝 친화적' 방식입니다.
X_outliers = X_outliers.assign(y=y_pred_outliers)

# ① 훈련 데이터(정상) — 흰 동그라미
plt.scatter(X_train.x1, X_train.x2,
            c='white', s=20*4, edgecolor='k',
            label="training observations")

# ② 이상치 중 '이상(-1)'으로 제대로 잡힌 것 — 빨간 점
#   .loc[조건, ['열이름']] : 조건에 맞는 행의 특정 열만 뽑습니다.
plt.scatter(X_outliers.loc[X_outliers.y == -1, ['x1']],
            X_outliers.loc[X_outliers.y == -1, ['x2']],
            c='red', s=20*4, edgecolor='k',
            label="detected outliers")

# ③ 이상치인데 '정상(+1)'으로 놓친 것 — 초록 점
plt.scatter(X_outliers.loc[X_outliers.y == 1, ['x1']],
            X_outliers.loc[X_outliers.y == 1, ['x2']],
            c='green', s=20*4, edgecolor='k',
            label="detected regular obs")

plt.legend(loc='upper right')
plt.show()


# %% [Block 7] ▸ 결과 시각화 — 강의 자료 12p
import numpy as np

# 학습이 끝난 모델은 자신이 추정한 중심과 공분산을 속성으로 갖고 있습니다.
#   (이름 끝의 밑줄 _ 은 사이킷런의 관례입니다: "학습 후에 생기는 속성"이라는 표시)
print("타원의 중심 (location_)   :", np.round(clf.location_, 4))
print()
print("공분산 행렬 (covariance_) :")
print(np.round(clf.covariance_, 4))
print()

# np.linalg.eigh : 대칭행렬의 고유값·고유벡터를 구합니다.
#   (eig가 아니라 eigh인 이유 : h = Hermitian, 대칭행렬 전용이라 더 빠르고 정확합니다)
eigvals, eigvecs = np.linalg.eigh(clf.covariance_)

print("고유값        :", np.round(eigvals, 4))
print("→ 타원 반지름 :", np.round(np.sqrt(eigvals), 4), "(√고유값)")
print("→ 긴 축 / 짧은 축 비율 :",
      round(np.sqrt(eigvals[1] / eigvals[0]), 1), "배")
print()
print("가장 긴 축의 방향 (고유벡터) :", np.round(eigvecs[:, -1], 4))
print("→ (0.71, 0.71) 은 45도 대각선 방향을 뜻합니다.")


# %% [Block 8] ▸ 결정적 증거 — 아무것도 없는 자리를 물어보기
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

# 세 지점을 시험지에 올려 봅니다.
probe = np.array([
    [1.5, 1.5],   # ① 두 덩어리 사이의 텅 빈 공간 — 상식적으로는 '이상'
    [3.0, 3.0],   # ② 오른쪽 위 덩어리 한복판    — 명백히 '정상'
    [0.0, 0.0],   # ③ 왼쪽 아래 덩어리 한복판    — 명백히 '정상'
    [0.0, 3.0],   # ④ 완전히 딴 곳              — 명백히 '이상'
])

# 세 모델을 같은 데이터로 학습시켜 비교합니다.
models = {
    "EllipticEnvelope": EllipticEnvelope(contamination=0.1, random_state=42),
    "LocalOutlierFactor": LocalOutlierFactor(n_neighbors=20, novelty=True,
                                             contamination=0.1),
    "IsolationForest": IsolationForest(contamination=0.1, random_state=42),
}

labels = ["(1.5,1.5) 빈 공간", "(3.0,3.0) 덩어리 안",
          "(0.0,0.0) 덩어리 안", "(0.0,3.0) 딴 곳"]

# X_train.values : DataFrame을 넘파이 배열로 (열 이름 경고를 피하기 위함)
Xtr = X_train.values

print(f"{'':>20}", " | ".join(f"{l:>17}" for l in labels))
print("-" * 100)
for name, m in models.items():
    m.fit(Xtr)
    pred = m.predict(probe)
    # +1이면 '정상', -1이면 '이상'으로 글자를 바꿔 출력합니다.
    cells = ["      정상(+1)" if v == 1 else "      이상(-1)" for v in pred]
    print(f"{name:>20}", " | ".join(f"{c:>17}" for c in cells))


# %% [Block 9] ▸ 수학적으로 왜 이런 일이 벌어지는가
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

rng3 = np.random.RandomState(42)

# ── 이번엔 정상 데이터를 '한 덩어리'로만 만듭니다 ──────────────
#   앞에서는 np.r_[X+3, X] 로 두 덩어리를 만들었지만,
#   여기서는 처음부터 2000개를 (3,3) 근처 한 곳에 몰아 넣습니다.
#   ★ 개수(2000개)와 퍼짐(0.2)은 그대로, '덩어리 개수'만 2 → 1로 바꿉니다.
X_one = 0.2 * rng3.randn(2000, 2) + 3

# ★ 이상치는 앞에서 쓰던 것을 '그대로' 씁니다.
#   시험 문제를 똑같이 두고 교과서만 바꿔야 공정한 비교가 됩니다.

print(f"{'모델':>20} | {'두 덩어리 훈련':>15} | {'한 덩어리 훈련':>15} | 변화")
print("-" * 74)

pairs = [
    ("EllipticEnvelope", lambda: EllipticEnvelope(contamination=0.1, random_state=42)),
    ("LocalOutlierFactor", lambda: LocalOutlierFactor(n_neighbors=20, novelty=True,
                                                      contamination=0.1)),
    ("IsolationForest", lambda: IsolationForest(contamination=0.1, random_state=42)),
]

for name, make in pairs:
    # (가) 두 덩어리 데이터로 학습 — 앞에서 쓰던 것
    m1 = make().fit(X_train.values)
    acc_two = (m1.predict(outliers.values) == -1).mean()

    # (나) 한 덩어리 데이터로 학습 — EllipticEnvelope의 가정이 성립하는 경우
    m2 = make().fit(X_one)
    acc_one = (m2.predict(outliers.values) == -1).mean()

    diff = acc_one - acc_two
    mark = f"+{diff:.2f} ▲▲▲" if diff > 0.05 else f"{diff:+.2f}"
    print(f"{name:>20} | {acc_two:>15.4f} | {acc_one:>15.4f} | {mark}")

print()
print("EllipticEnvelope : 0.82 → 0.96 으로 껑충 뜁니다.")
print("나머지 둘은 거의 그대로입니다 (원래도 잘하고 있었으니까요).")
print("모델을 바꾼 게 아니라, 데이터가 모델의 가정에 맞아떨어진 것입니다.")


# %% [Block 10] ▸ 프레임워크 비교
import numpy as np

# ── 실습 2 정답 : decision_function의 정체 밝히기 ──────────────
d2 = clf.mahalanobis(X_test.values)          # 제곱된 마할라노비스 거리
df = clf.decision_function(X_test.values)    # 사이킷런이 주는 점수
offset = clf.offset_                          # 임계값 (음수로 저장되어 있음)

print("offset_ :", round(offset, 4))
print()
print("앞 5개 비교")
print("  마할라노비스 제곱  :", np.round(d2[:5], 4))
print("  decision_function :", np.round(df[:5], 4))
print("  -d2 - offset      :", np.round(-d2[:5] - offset, 4))
print()
# np.allclose : 두 배열이 (소수점 오차 범위 안에서) 같은지 확인합니다.
print("두 값이 같은가? :", np.allclose(df, -d2 - offset))
print()
print("→ decision_function = -(마할라노비스 제곱) - offset_")
print("   거리에 마이너스를 붙여 '클수록 정상'으로 방향을 뒤집은 것입니다.")


# %% [Block 11] ▸ 프레임워크 비교
import numpy as np
from sklearn.mixture import GaussianMixture

# ── 실습 4 정답 : 정규분포 두 개를 동시에 추정하기 ──────────────
#  n_components=2 : "정상 데이터는 정규분포 2개가 섞인 것"이라고 알려 줍니다.
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X_train.values)

# score_samples : 로그 확률밀도 log p(x)를 돌려줍니다. 클수록 흔한 자리.
log_p_train = gmm.score_samples(X_train.values)

# np.percentile(배열, 10) : 아래에서 10% 지점의 값
#   → contamination=0.1 과 똑같은 방식으로 임계값을 정합니다.
tau = np.percentile(log_p_train, 10)

print("추정된 두 중심 :")
print(np.round(gmm.means_, 4))
print()
print("→ (0,0)과 (3,3)을 정확히 찾아냈습니다!")
print("   EllipticEnvelope가 (1.51, 1.50) 하나만 찾았던 것과 대조됩니다.")
print()

acc_out = (gmm.score_samples(outliers.values) < tau).mean()
acc_test = (gmm.score_samples(X_test.values) >= tau).mean()
print("GaussianMixture 이상치 검출율 :", round(acc_out, 4),
      "  (EllipticEnvelope는 0.82였습니다)")
print("GaussianMixture 정상 통과율   :", round(acc_test, 4),
      "  (EllipticEnvelope는 0.9075였습니다)")
