import joblib, numpy as np, pandas as pd, os
from features import build_features
from physics import TARGETS, solve, euler_buckling_load
from train import SIGN

_BUNDLE = None
def load():
    global _BUNDLE
    if _BUNDLE is None: _BUNDLE = joblib.load(os.path.join(os.path.dirname(__file__), "models", "surrogate.joblib"))
    return _BUNDLE

def predict(inp: dict):
    """inp keys: section, load_type, bc, L, b, h, d, E, nu, Sy, F.  Returns (values, 90% interval, analytical reference)."""
    B = load(); df = pd.DataFrame([inp]); X = build_features(df)[B["columns"]]
    val, ci = {}, {}
    for t in TARGETS:
        off = SIGN[t] * np.log10(inp["F"])
        val[t] = float(10 ** (B["models"][t].predict(X)[0] + off))
        raw = np.sort(B["intervals"][t].predict(X)[0]); qh = B.get("conf_q", {}).get(t, 0.0)
        lo, hi = 10 ** (raw[0] - qh + off), 10 ** (raw[1] + qh + off); ci[t] = (float(min(lo, val[t])), float(max(hi, val[t])))
    ref = solve(inp["section"], inp["load_type"], inp["bc"], inp["L"], inp.get("b") or 0, inp.get("h") or 0, inp.get("d") or 0,
                inp["E"], inp["nu"], inp["Sy"], inp["F"])
    ref = {k: float(v) for k, v in ref.items()}
    warn = []
    if inp["load_type"] == "compressive":
        pcr = float(euler_buckling_load(inp["section"], inp["bc"], inp["L"], inp.get("b") or 0, inp.get("h") or 0, inp.get("d") or 0, inp["E"]))
        if inp["F"] > pcr: warn.append(f"Applied load exceeds Euler buckling load (~{pcr:.0f} N): linear static results are not meaningful.")
    if val["factor_of_safety"] < 1: warn.append("Predicted FoS < 1: yielding expected.")
    return val, ci, ref, warn