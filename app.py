import streamlit as st, pandas as pd, json, os, time
from physics import LOADS, SECTIONS, VALID_BC, TARGETS
from predictor import predict
st.set_page_config(page_title="ML-FEA Surrogate", layout="wide")
st.title("ML surrogate for ANSYS Static Structural (simple bars & shafts)")
st.caption("XGBoost surrogate trained on parametric FEA-style data. Valid only inside the training envelope shown in the sidebar.")

with st.sidebar:
    st.header("Inputs (mm, N, MPa)")
    sec = st.selectbox("Section", SECTIONS); lt = st.selectbox("Load type", LOADS)
    bc = st.selectbox("Boundary condition", VALID_BC[lt])
    L = st.number_input("Length L [mm]", 100.0, 1500.0, 500.0)
    if sec == "rectangular":
        b = st.number_input("Breadth b [mm]", 10.0, 100.0, 30.0); h = st.number_input("Height h [mm]", 10.0, 100.0, 50.0); d = 0.0
    else:
        d = st.number_input("Diameter d [mm]", 10.0, 100.0, 40.0); b = h = 0.0
    mat = st.selectbox("Material preset", ["Structural steel", "Aluminium 6061-T6", "Titanium Ti-6Al-4V", "Grey cast iron", "Custom"])
    P = {"Structural steel": (200e3, .30, 250), "Aluminium 6061-T6": (68.9e3, .33, 276), "Titanium Ti-6Al-4V": (113.8e3, .342, 880),
         "Grey cast iron": (110e3, .28, 200), "Custom": (200e3, .30, 250)}[mat]
    E = st.number_input("Young's modulus E [MPa]", 60e3, 220e3, float(P[0])); nu = st.number_input("Poisson ratio", .25, .35, float(P[1]))
    Sy = st.number_input("Yield strength Sy [MPa]", 100.0, 1000.0, float(P[2]))
    unit = "N*mm" if lt == "torsion" else "N"
    lo, hi = {"torsion": (1e4, 1e7), "bending": (50, 1e4)}.get(lt, (100, 1e5))
    F = st.number_input(f"Applied {'torque' if lt=='torsion' else 'load'} [{unit}]", float(lo), float(hi), float(min(max(1000 if lt != 'torsion' else 2e5, lo), hi)))
    st.caption("Envelope: L/min-dim >= 5 (slender members); loads within range shown.")

inp = dict(section=sec, load_type=lt, bc=bc, L=L, b=b, h=h, d=d, E=E, nu=nu, Sy=Sy, F=F)
if sec == "rectangular" and L < 5 * max(b, h) or sec == "circular" and L < 5 * d:
    st.warning("Outside training envelope: L should be >= 5x the largest cross-section dimension.")
if not os.path.exists("models/surrogate.joblib"): st.error("Run generate_data.py and train.py first."); st.stop()
t0 = time.perf_counter(); val, ci, ref, warn = predict(inp); ms = (time.perf_counter() - t0) * 1e3
for w in warn: st.warning(w)
st.subheader(f"Prediction  ({ms:.1f} ms)")
names = {"max_deformation_mm": "Max deformation [mm]", "von_mises_MPa": "von Mises [MPa]", "max_principal_MPa": "Max |principal| [MPa]", "factor_of_safety": "Factor of safety"}
cols = st.columns(4)
for c, t in zip(cols, TARGETS): c.metric(names[t], f"{val[t]:.4g}", help=f"90% interval: {ci[t][0]:.4g} - {ci[t][1]:.4g}")
R = json.load(open("reports/metrics.json")) if os.path.exists("reports/metrics.json") else {}
case = f"{lt}/{bc}/{sec}"
rows = [dict(Quantity=names[t], **{"ML prediction": val[t], "90% lo": ci[t][0], "90% hi": ci[t][1]},
             **{"Beam-theory check (not ANSYS)": ref[t]},
             **{"Typical ML-vs-ANSYS error %": R.get("by_case_MAPE_pct", {}).get(t, {}).get(case)}) for t in TARGETS]
st.dataframe(pd.DataFrame(rows).style.format({c: "{:.4g}" for c in ["ML prediction", "90% lo", "90% hi", "Beam-theory check (not ANSYS)"]} | {"Typical ML-vs-ANSYS error %": "{:.1f}"}, na_rep="-"), hide_index=True)
st.caption("The ML model is trained on the data source in data/. The beam-theory column is only a sanity check: ANSYS values differ from textbook formulas near supports and depend on the stress-extraction rule (see README).")

st.subheader("Model quality on held-out test set")
if R:
    st.dataframe(pd.DataFrame([{"Target": t, **{k: v for k, v in R[t]["test"].items()}, "CV R2 (log)": R[t]["cv5_R2_log"]["mean"], "90% coverage": R[t]["interval90_coverage"]} for t in TARGETS]), hide_index=True)
    tabs = st.tabs([names[t] for t in TARGETS])
    for tb, t in zip(tabs, TARGETS): tb.image(f"reports/residuals_{t}.png")
    st.subheader("Physics-based validation"); st.json(R["physics_validation"])