#Explainer V2
# explainer.py
import numpy as np
import scipy.sparse as sp
from typing import Dict, Any, List


def _lr_from_calibrated(clf):
    """
    Return the underlying LogisticRegression from CalibratedClassifierCV
    (or a calibrated pipeline) so we can read coef_ and intercept_ safely.
    """
    lr = None
    # CalibratedClassifierCV stores per-fold estimators
    if hasattr(clf, "calibrated_classifiers_") and clf.calibrated_classifiers_:
        lr = clf.calibrated_classifiers_[0].estimator
    # Some versions expose base_estimator_
    if lr is None and hasattr(clf, "base_estimator_"):
        lr = clf.base_estimator_
    if lr is None or not hasattr(lr, "coef_") or not hasattr(lr, "intercept_"):
        raise RuntimeError("Could not access linear coefficients/intercept from calibrated classifier.")
    return lr


def _safe_feature_names(pipe) -> List[str]:
    ct = pipe.named_steps["features"]
    try:
        return list(ct.get_feature_names_out())
    except Exception:
        names = []
        try:
            # stitch manually (word + char + section one-hot)
            w = list(ct.named_transformers_["txt_word"].get_feature_names_out())
            w = [f"txt_word__{t}" for t in w]
            c = list(ct.named_transformers_["txt_char"].get_feature_names_out())
            c = [f"txt_char__{t}" for t in c]
            secs = ct.transformers_[2][2]  # column list of section one-hot
            secs = [f"sec_hot__{s}" for s in secs]
            names = w + c + secs
        except Exception:
            pass
        if not names:
            # last resort: generic names
            names = [f"f{i}" for i in range(pipe.named_steps["features"].transformers_[0][1].shape[1])]
        return names


def _aggregate(tokens: List[str], contrib: np.ndarray, top_k=8):
    groups = {"words": [], "char_ngrams": [], "sections": []}
    for t, v in zip(tokens, contrib):
        if t.startswith("txt_word__"):
            groups["words"].append((t.replace("txt_word__", ""), float(v)))
        elif t.startswith("txt_char__"):
            groups["char_ngrams"].append((t.replace("txt_char__", ""), float(v)))
        elif t.startswith("sec_hot__") or t.startswith("sec_"):
            groups["sections"].append((t.split("__")[-1], float(v)))

    def top2(arr):
        arr = sorted(arr, key=lambda x: x[1], reverse=True)
        pos = arr[:top_k]
        neg = sorted(arr, key=lambda x: x[1])[:top_k]
        return pos, neg

    pos_w, neg_w = top2(groups["words"])
    pos_c, neg_c = top2(groups["char_ngrams"])
    pos_s, neg_s = top2(groups["sections"])
    return {
        "pos": {"words": pos_w, "char_ngrams": pos_c, "sections": pos_s},
        "neg": {"words": neg_w, "char_ngrams": neg_c, "sections": neg_s},
    }


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def explain_single(pipe, X_row, top_k: int = 8) -> Dict[str, Any]:
    """
    Deterministic, fast attribution for a linear model:
    contribution = feature_value * coefficient (per-feature).
    We now use margin = x·coef + intercept so the sign is aligned with the classifier.
    """
    Xt = pipe.named_steps["features"].transform(X_row)
    x = Xt.tocsr() if sp.issparse(Xt) else np.asarray(Xt)

    lr = _lr_from_calibrated(pipe.named_steps["clf"])
    coef = lr.coef_.ravel()                # direction for class 1
    intercept = float(lr.intercept_.ravel()[0])

    # contributions exclude intercept (it’s a global bias, not tied to a token)
    if sp.issparse(x):
        margin = float(x.dot(coef)[0] + intercept)
        contrib = (x.multiply(coef)).toarray().ravel()
    else:
        margin = float((x * coef).sum() + intercept)
        contrib = (x * coef).ravel()

    names = _safe_feature_names(pipe)[: len(contrib)]
    grouped = _aggregate(names, contrib, top_k=top_k)

    # proba for class 1 (whatever your class 1 is in training)
    proba_class1 = _sigmoid(margin)

    return {
        "margin": margin,
        "proba_class1": proba_class1,
        "grouped": grouped,
    }
