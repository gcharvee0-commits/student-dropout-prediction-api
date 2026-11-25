"""engine.py - the prediction logic (no web code), so it can be tested without FastAPI."""
from pathlib import Path
import joblib, numpy as np, pandas as pd

BUNDLE = joblib.load(Path(__file__).parent / "model_bundle.joblib")
LABELS = BUNDLE["labels"]
NICE = {"prev_qual_pct": "12th / previous-qualification %", "admission_pct": "Admission score %",
        "admission_missing": "Admission score missing", "hostel": "Studying away from home",
        "first_gen": "First-generation student", "scholarship": "Scholarship holder",
        "fee_dues": "Fee dues pending", "s1_registered": "Semester 1 subjects registered",
        "s1_passed": "Semester 1 subjects passed", "s1_pass_rate": "Semester 1 pass rate",
        "s1_sgpa": "Semester 1 SGPA", "s1_no_reg": "Not registered in semester 1",
        "s2_registered": "Semester 2 subjects registered", "s2_passed": "Semester 2 subjects passed",
        "s2_pass_rate": "Semester 2 pass rate", "s2_sgpa": "Semester 2 SGPA",
        "s2_no_reg": "Not registered in semester 2", "delta_pass_rate": "Change in pass rate (sem 2 vs 1)",
        "delta_sgpa": "Change in SGPA (sem 2 vs 1)", "age": "Age at enrollment"}
# Used by the model, but never shown as a "reason" to a counsellor (sensitive / not actionable).
HIDDEN = {"gender_male", "special_needs", "international"} | \
         {f"{w}_occ_{g}" for w in ("mother", "father") for g in range(1, 10)}
OPTIONAL_NEW = ["age_at_enrollment", "gender", "special_needs", "international"]


def checkpoint_for(d):
    if d.get("semester1") is None:
        if d.get("semester2") is not None:
            raise ValueError("semester2 was given without semester1")
        return "A"
    return "C" if d.get("semester2") is not None else "B"


def _num(d, key):
    v = d.get(key)
    return np.nan if v is None else float(v)


def build_row(d, cp):
    f = {"prev_qual_pct": d["previous_qualification_percent"],
         "admission_pct": d.get("admission_percent") if d.get("admission_percent") is not None else np.nan,
         "admission_missing": float(d.get("admission_percent") is None),
         "hostel": float(d["hostel"]), "first_gen": float(d["first_generation"]),
         "scholarship": float(d["scholarship"]),
         # --- new in v2: all optional; a missing value is filled with the training average ---
         "age": _num(d, "age_at_enrollment"),
         "gender_male": np.nan if d.get("gender") is None else float(d["gender"] == "male"),
         "special_needs": _num(d, "special_needs"),
         "international": _num(d, "international")}
    for who in ("mother", "father"):
        grp = d.get(f"{who}_occupation_group") or 0
        for g in range(1, 10):
            # unknown / not given -> NaN, so it is filled with the training average (neutral).
            # (All-zeros would be a rare pattern that pushes the score far from the average.)
            f[f"{who}_occ_{g}"] = float(grp == g) if grp else np.nan
    if cp in ("B", "C"):
        f["fee_dues"] = float(d.get("fee_dues", 0))
    for tag, key in (("s1", "semester1"), ("s2", "semester2")):
        s = d.get(key)
        if s is None:
            continue
        reg, passed = float(s["registered"]), float(s["passed"])
        f[f"{tag}_registered"], f[f"{tag}_passed"] = reg, passed
        f[f"{tag}_pass_rate"] = passed / reg if reg > 0 else 0.0
        f[f"{tag}_sgpa"] = float(s["sgpa"])
        f[f"{tag}_no_reg"] = float(reg == 0)
    if cp == "C":
        f["delta_pass_rate"] = f["s2_pass_rate"] - f["s1_pass_rate"]
        f["delta_sgpa"] = f["s2_sgpa"] - f["s1_sgpa"]
    return pd.DataFrame([f])[BUNDLE["models"][cp]["columns"]]


def predict(d):
    cp = checkpoint_for(d)
    m = BUNDLE["models"][cp]
    row = build_row(d, cp)
    pipe = m["pipeline"]
    proba = pipe.predict_proba(row)[0]
    p_drop = float(proba[0])
    # why: standardised value x Dropout coefficient (logistic regression is directly readable)
    z = pipe[:-1].transform(row)[0]
    contrib = pipe.named_steps["model"].coef_[0] * z
    cols = m["columns"]
    order = [i for i in np.argsort(-np.abs(contrib)) if cols[i] not in HIDDEN][:3]
    factors = [{"factor": NICE[cols[i]], "effect": "raises risk" if contrib[i] > 0 else "lowers risk"}
               for i in order]
    defaulted = [k for k in OPTIONAL_NEW if d.get(k) is None]
    return {"checkpoint": cp, "predicted_outcome": LABELS[int(proba.argmax())],
            "probabilities": {l: round(float(p), 4) for l, p in zip(LABELS, proba)},
            "dropout_probability": round(p_drop, 4), "review_threshold": m["threshold"],
            "flag_for_counsellor_review": bool(p_drop >= m["threshold"]),
            "top_factors": factors,
            "optional_fields_not_provided": defaulted,
            "model_version": BUNDLE.get("version", "v1"),
            "note": ("Model validated on first-year data only (UCI 697). Advisory, not a verdict; "
                     "a counsellor reviews every flag. Gender, special-needs, nationality and parents' "
                     "occupation are used by the model but are not listed as top factors.")}
