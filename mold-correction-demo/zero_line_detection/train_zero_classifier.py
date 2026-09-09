"""제로라인 후보 분류기 — 학습/검증 하네스 (스모크테스트용, 아직 배포 아님).

[지금 이게 왜 스모크테스트인가]
부품이 3개(64XX2/67XX6/71XX2)뿐이라 어떤 수치를 내도 통계적으로 못 믿는다.
leave-one-part-out 검증도 폴드당 부품 1개짜리라 우연에 크게 흔들린다.
이 스크립트의 목적은 "숫자가 좋다"가 아니라 "배관이 돌아간다" —
회의록에서 요청한 사례(약 10건)가 들어와 부품이 늘면 그대로 재사용해서
바로 의미 있는 결과를 볼 수 있게 해두는 것이다.

[모델을 이렇게 가볍게 만든 이유]
scikit-learn 도 안 쓴다. 특징 10개, 표본 38개짜리 데이터에 복잡한 모델을
쓰면 라이브러리 선택 자체가 과적합이다. numpy 로짜리 로지스틱 회귀
(L2 정규화, 경사하강)면 충분하고 뭘 하는지 100% 눈으로 볼 수 있다.

[아직 프로덕션에 안 붙인 이유]
n=3 부품짜리 모델은 지금 규칙 기반(zero_points.cluster_zero_points /
connect_strongest_pair)보다 못할 가능성이 높다. 사례가 15~20부품 이상
쌓이기 전엔 서버 응답에 이 모델 점수를 섞지 않는다. 여기서는 가중치를
zero_line_detection/ml_data/classifier_weights.json 에 저장만 해둔다.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from zero_line_detection.ml_dataset import DATASET_DIR, FEATURE_NAMES

CSV_PATH = DATASET_DIR / "candidate_dataset.csv"
WEIGHTS_PATH = DATASET_DIR / "classifier_weights.json"

MIN_PARTS_TO_TRUST = 15  # 이 미만이면 결과를 프로덕션에 쓰면 안 된다


def load_dataset(csv_path: Path = CSV_PATH) -> list:
    rows = []
    with csv_path.open(encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            row = {
                "part_no": raw["part_no"], "loop": raw["loop"],
                "x": float(raw["x"]), "y": float(raw["y"]),
                "label": int(raw["label"]),
            }
            for name in FEATURE_NAMES:
                row[name] = float(raw[name])
            rows.append(row)
    return rows


def _matrix(rows: list) -> tuple:
    X = np.array([[r[f] for f in FEATURE_NAMES] for r in rows], dtype=float)
    y = np.array([r["label"] for r in rows], dtype=float)
    return X, y


def standardize(X: np.ndarray, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std


def train_logreg(X: np.ndarray, y: np.ndarray, l2: float = 1.0,
                  lr: float = 0.2, iters: int = 3000):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad_w = X.T @ (p - y) / n + l2 * w / n
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    z = X @ w + b
    return 1.0 / (1.0 + np.exp(-z))


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    """양성 하나를 무작위로 뽑으면 음성보다 높은 점수를 받을 확률.

    scikit-learn 없이 순위-합 공식으로 계산한다(Mann-Whitney U 와 동치).
    """
    n_pos, n_neg = int(y_true.sum()), int((1 - y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float(
        (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


def leave_one_part_out(rows: list) -> list:
    parts = sorted({r["part_no"] for r in rows})
    results = []
    for held_out in parts:
        train_rows = [r for r in rows if r["part_no"] != held_out]
        test_rows = [r for r in rows if r["part_no"] == held_out]
        Xtr, ytr = _matrix(train_rows)
        Xte, yte = _matrix(test_rows)
        if ytr.sum() == 0 or ytr.sum() == len(ytr):
            print(f"  [{held_out}] 학습 폴드에 한쪽 라벨만 있어 건너뜁니다.")
            continue

        Xtr_s, mean, std = standardize(Xtr)
        Xte_s, _, _ = standardize(Xte, mean, std)
        w, b = train_logreg(Xtr_s, ytr)
        proba = predict_proba(Xte_s, w, b)
        auc = auc_score(yte, proba)
        pred = (proba >= 0.5).astype(int)
        tp = int(((pred == 1) & (yte == 1)).sum())
        fp = int(((pred == 1) & (yte == 0)).sum())
        fn = int(((pred == 0) & (yte == 1)).sum())
        tn = int(((pred == 0) & (yte == 0)).sum())
        print(f"  [{held_out}] n={len(yte)} (양성 {int(yte.sum())}) "
              f"AUC={auc:.2f}  TP={tp} FP={fp} FN={fn} TN={tn}")
        results.append({
            "part_no": held_out, "n": len(yte), "n_pos": int(yte.sum()),
            "auc": auc, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return results


def fit_final(rows: list) -> dict:
    X, y = _matrix(rows)
    X_s, mean, std = standardize(X)
    w, b = train_logreg(X_s, y)
    return {
        "feature_names": FEATURE_NAMES,
        "mean": mean.tolist(), "std": std.tolist(),
        "coef": w.tolist(), "intercept": float(b),
        "trained_on_parts": sorted({r["part_no"] for r in rows}),
        "note": (
            "스모크테스트 산출물. 부품 수가 MIN_PARTS_TO_TRUST 미만이면 "
            "프로덕션 랭킹에 쓰지 말 것."
        ),
    }


def main() -> int:
    rows = load_dataset()
    n_parts = len({r["part_no"] for r in rows})
    print(f"총 {len(rows)}개 후보, {n_parts}개 부품 로드")

    if n_parts < MIN_PARTS_TO_TRUST:
        print(
            f"\n[경고] 부품이 {n_parts}개뿐입니다(신뢰 기준 {MIN_PARTS_TO_TRUST}개). "
            "아래 수치는 코드가 도는지 확인하는 스모크테스트일 뿐 통계적으로 "
            "의미가 없습니다. 회의록에서 요청한 사례(약 10건)가 들어와 부품이 "
            "늘어난 뒤 다시 돌려야 신뢰할 수 있습니다."
        )

    print("\n=== Leave-one-part-out 검증 ===")
    leave_one_part_out(rows)

    print("\n=== 전체 데이터로 최종 학습 (가중치만 저장, 서버에는 미연결) ===")
    weights = fit_final(rows)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_PATH.write_text(
        json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"가중치 저장: {WEIGHTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
