# -*- coding: utf-8 -*-
"""
Zero to 머신러닝 딥러닝 Master — Medium 시리즈 #08 · 1부
고유값·SVD·PCA 쉽게 이해하기 — 300페이지 책을 3페이지로 (1부)

원문(책): 00-06 「SVD·PCA — 300페이지 책을 3페이지로 (1부)」 https://wikidocs.net/439842
Medium : TODO (게시 후 추가)
책     : https://wikidocs.net/book/21464
홈페이지: https://mldict.net
블로그 : https://wikidocs.net/blog/@mldict/

실행: pip install -r requirements.txt && python 08_1_eigen_svd_pca.py
※ 아래 코드는 책 본문의 코드 블록을 순서대로 옮긴 것입니다(내용 변경 없음).
"""

#======================================================================
# [1] 00-06-A · 1단계 — 고유벡터와 고유값
#     eig 결과로 Av = λv 를 검증하고, 특성방정식 det(A − λI)=0 과 손계산 고유벡터를 비교합니다.
#     ⚠️ NumPy 2.5 이상에서는 np.linalg.eig 가 항상 복소수 배열을 돌려주므로 값 뒤에 +0.j 가 붙어 출력될 수 있습니다(값은 같음). 또 고유값의 순서와 고유벡터의 부호(±)는 NumPy/LAPACK 환경에 따라 책과 다를 수 있습니다 — 고유벡터는 방향만 같으면 됩니다. 이 노트북은 NumPy 2.4.4로 실행했습니다.
#======================================================================
print("\n" + "=" * 60)
print("[1] 00-06-A · 1단계 — 고유벡터와 고유값")
print("=" * 60)
import numpy as np

A = np.array([[4., 1.],
              [2., 3.]])

# ══════ ① 라이브러리로 고유값·고유벡터 구하기 ══════
# eig()는 (고유값 배열, 고유벡터 행렬)을 반환합니다.
# ⚠️ 고유벡터 행렬은 '각 열'이 하나의 고유벡터입니다 (행이 아님!)
eigenvalues, eigenvectors = np.linalg.eig(A)

print("고유값:", eigenvalues)
print("고유벡터(열 단위 저장):\n", eigenvectors.round(6))

# ══════ ② 정의 Av = λv 가 성립하는지 검증 ══════
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]        # i번째 '열' = i번째 고유벡터
    lam = eigenvalues[i]
    left = A @ v                  # Av  (00-02 행렬×벡터)
    right = lam * v                # λv  (00-01 스칼라배)
    ok = np.allclose(left, right) # 부동소수점이라 '거의 같은지'로 비교
    print(f"  λ={lam:.1f}: Av={left.round(4)}, λv={right.round(4)} → {ok}")

# ══════ ③ 특성방정식 det(A - λI) = 0 을 직접 확인 ══════
# 손으로 푼 λ=5, 2 를 넣으면 행렬식이 0이 되어야 합니다
I = np.eye(2)                      # 2×2 단위행렬 (00-02)
for lam in [5.0, 2.0, 3.0]:  # 3은 고유값이 아님 → 0이 안 나와야 정상
    det = np.linalg.det(A - lam * I)
    tag = "고유값 ✅" if abs(det) < 1e-9 else "고유값 아님 ❌"
    print(f"  det(A - {lam}I) = {det:+.6f}  → {tag}")

# ══════ ④ 고유벡터는 '방향'만 같으면 된다 ══════
v_hand = np.array([1., 1.])           # 손으로 구한 것
v_lib = eigenvectors[:, 0]           # 라이브러리 결과 (길이 1로 정규화됨)
print(f"\n손계산 (1,1)을 00-01의 정규화 적용: {(v_hand/np.linalg.norm(v_hand)).round(6)}")
print(f"라이브러리 결과              : {v_lib.round(6)}")


#======================================================================
# [2] 00-06-B · 2단계 — SVD와 랭크-k 근사
#     특잇값·랭크·에너지 비율을 보고, A_k = U_k Σ_k V_kᵀ 근사의 오차를 이론값과 비교합니다.
#======================================================================
print("\n" + "=" * 60)
print("[2] 00-06-B · 2단계 — SVD와 랭크-k 근사")
print("=" * 60)
import numpy as np
np.set_printoptions(suppress=True)   # 1e-16 같은 지수 표기 끄기

# 각 행이 '앞 행 + 1'인 규칙적인 행렬 → 사실상 패턴이 2개뿐
A = np.array([[1., 2., 3., 4.],
              [2., 3., 4., 5.],
              [3., 4., 5., 6.],
              [4., 5., 6., 7.]])

# ══════ ① SVD 분해 ══════
# ⚠️ S는 '1차원 배열'로 반환됩니다 (대각행렬 아님!)
U, S, Vt = np.linalg.svd(A)

print("특잇값 S :", S.round(4))
print("행렬 랭크 :", np.linalg.matrix_rank(A), "← 0이 아닌 특잇값의 개수")

# ══════ ② 에너지 비율로 k 정하기 ══════
# energy(k) = (상위 k개 σ²의 합) / (전체 σ²의 합)
energy = np.cumsum(S**2) / np.sum(S**2)   # cumsum: 누적 합
print("\n--- k별 에너지 비율과 오차 ---")
for k in [1, 2, 3]:
    # ★핵심 한 줄★ 수식 A_k = U[:,:k] Σ_k Vt[:k,:] 그대로
    A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

    # 00-01의 노름을 행렬로 확장한 '프로베니우스 노름'으로 오차 측정
    err = np.linalg.norm(A - A_k)            # 기본값이 'fro'

    # 이론값 √(σ_{k+1}² + ...) 과 비교 (에카르트-영 정리)
    theory = np.sqrt(np.sum(S[k:]**2))
    print(f"  k={k}: 에너지 {energy[k-1]:.4f} | 실제오차 {err:.6f}"
          f" | 이론오차 {theory:.6f}")

# ══════ ③ k=1과 k=2 근사 행렬 비교 ══════
A_1 = U[:, :1] @ np.diag(S[:1]) @ Vt[:1, :]
A_2 = U[:, :2] @ np.diag(S[:2]) @ Vt[:2, :]
print("\nk=1 근사 (패턴 1개만):\n", A_1.round(2))
print("\nk=2 근사 (패턴 2개):\n", A_2.round(2))
print("\n원본과 k=2가 같은가?", np.allclose(A, A_2))

# ══════ ④ '층의 합' 형태로도 같은지 확인 ══════
# A_k = Σ σ_i · u_i v_iᵀ  (외적의 가중합)
A_layers = sum(S[i] * np.outer(U[:, i], Vt[i, :]) for i in range(2))
print("층의 합 방식과 슬라이싱 방식이 같은가?", np.allclose(A_2, A_layers))


#======================================================================
# [3] 00-06-C · 3단계 — PCA: 공분산 고유분해 vs SVD
#     중심화 → 공분산 → eigh 경로와 SVD 경로가 같은 고유값을 주는지 확인하고, 2D→1D 압축 후 복원합니다 (seed 42).
#======================================================================
print("\n" + "=" * 60)
print("[3] 00-06-C · 3단계 — PCA: 공분산 고유분해 vs SVD")
print("=" * 60)
import numpy as np
np.random.seed(42)

# ══════ 데이터 생성: 키와 상관된 몸무게 (00-05 정규분포 활용) ══════
n = 100
height = np.random.normal(170, 8, n)                    # N(170, 8²)
weight = 0.8 * height - 60 + np.random.normal(0, 3, n)  # 키에 노이즈 추가
X = np.column_stack([height, weight])                    # (100, 2)

# ══════ 경로 A) 공분산 행렬 → 고유값 분해 ══════
# ① 중심화 — ⚠️ 절대 빠뜨리면 안 되는 단계!
Xc = X - X.mean(axis=0)

# ② 공분산 행렬 C = XcᵀXc/(n-1)  ← 00-03의 AᵀA와 같은 형태
C = np.cov(Xc.T)                     # ⚠️ 전치 필수! (변수가 행이 되어야 함)
print("공분산 행렬:\n", C.round(4))

# ③ 고유값 분해 — 대칭행렬이므로 eigh (1단계에서 배운 규칙!)
eigenvalues, eigenvectors = np.linalg.eigh(C)

# ④ eigh는 '오름차순'으로 주므로 내림차순으로 뒤집기
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]      # 열 단위로 재정렬

ratio = eigenvalues / eigenvalues.sum()      # 설명 분산 비율
print(f"\n고유값 : {eigenvalues.round(4)}")
print(f"설명비율: PC1 {ratio[0]:.1%} / PC2 {ratio[1]:.1%}")
print(f"PC1 방향: {eigenvectors[:, 0].round(4)}")

# ══════ 경로 B) SVD로 바로 (수식 ⑤ 검증!) ══════
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
lambda_from_svd = S**2 / (n - 1)          # λ = σ²/(n-1)

print(f"\n[검증] 공분산 경로 고유값: {eigenvalues.round(4)}")
print(f"[검증] SVD 경로  σ²/(n-1): {lambda_from_svd.round(4)}")
print(f"[검증] 두 방법이 일치하는가? {np.allclose(eigenvalues, lambda_from_svd)}")

# ══════ 차원 축소: 2D → 1D 투영 후 복원 ══════
k = 1
V_k = eigenvectors[:, :k]            # PC1만 사용 (2×1)
Z = Xc @ V_k                         # 투영: (100,2)@(2,1) → (100,1)
X_restored = Z @ V_k.T + X.mean(axis=0)   # 복원 (평균 다시 더하기!)

recon_error = np.linalg.norm(X - X_restored)
theory_error = np.sqrt((n - 1) * eigenvalues[1])   # 버린 고유값으로 예측
print(f"\n2D→1D 압축 후 복원 오차: {recon_error:.4f}")
print(f"이론 예측 √((n-1)·λ₂)  : {theory_error:.4f}")
