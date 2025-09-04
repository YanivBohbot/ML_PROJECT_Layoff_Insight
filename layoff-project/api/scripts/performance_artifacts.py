import os
import json
import numpy as np
from typing import Sequence, Optional

# Correct imports (avoid the mixups):
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize

# Headless plotting
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
metrics__path = os.path.join(BASE_DIR, "metrics")


def save_performance_artifacts(
    y_test,
    y_pred_test,
    classes: Sequence[int],
    class_labels: Sequence[str],  # ["Low..","Medium..","High..","Unknown.."]
    best_model,
) -> dict:
    # ---- scalar + per-class metrics
    accuracy = accuracy_score(y_test, y_pred_test)
    macro_f1 = f1_score(y_test, y_pred_test, average="macro")
    prec, rec, f1v, support = precision_recall_fscore_support(
        y_test, y_pred_test, labels=classes, zero_division=0
    )

    metrics = {
        "best_model": best_model,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class": [
            {
                "class_id": int(c),
                "label": class_labels[int(c)],
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for c, p, r, f, s in zip(classes, prec, rec, f1v, support)
        ],
    }

    # Save metrics.json
    with open(os.path.join(metrics__path, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # ---- confusion matrix (NPY + PNG)
    cm = confusion_matrix(y_test, y_pred_test, labels=classes)
    np.save(os.path.join(metrics__path, "confusion_matrix.npy"), cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[class_labels[int(c)] for c in classes]
    )
    fig = disp.plot(values_format="d", cmap="Blues", colorbar=True).figure_
    fig.tight_layout()
    fig.savefig(os.path.join(metrics__path, "confusion_matrix.png"))
    plt.close(fig)
    print(f"✅ Saved metrics.json and confusion matrix to {metrics__path}")
