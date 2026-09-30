import numpy as np, pandas as pd
from physics import LOADS, BCS, SECTIONS, section_props

def build_features(df, physics=True):
    """Raw inputs (+log transforms, one-hots) and optional physics-informed features."""
    s = df.section.values; lt = df.load_type.values; bc = df.bc.values
    b, h, d = (df[c].fillna(0).values.astype(float) for c in ("b", "h", "d"))
    L, E, nu, Sy, F = (df[c].values.astype(float) for c in ("L", "E", "nu", "Sy", "F"))
    X = {"is_circular": (s == "circular").astype(float)}
    for k in LOADS: X["load_" + k] = (lt == k).astype(float)
    for k in BCS: X["bc_" + k] = (bc == k).astype(float)
    for k, v in dict(L=L, b=b, h=h, d=d, E=E, Sy=Sy).items(): X["log_" + k] = np.log10(np.maximum(v, 1e-9))
    X["nu"] = nu
    if physics:
        F = np.ones_like(L)                     # load-free features: model predicts response per unit load
        p = section_props(s, b, h, d); G = E / (2 * (1 + nu))
        ax = np.isin(lt, ["tensile", "compressive"]); bend = lt == "bending"
        sig = np.select([ax, bend], [F / p["A"], F * L * p["c"] / p["I"]], F / p["W"])
        dfl = np.select([ax, bend], [F * L / (p["A"] * E), F * L**3 / (E * p["I"])], F * L * p["r_out"] / (G * p["J"]))
        char = np.where(s == "circular", d, np.minimum(np.where(b > 0, b, 1e9), np.where(h > 0, h, 1e9)))
        for k in ("A", "I", "J", "W"): X["log_" + k] = np.log10(p[k])
        X["slenderness"] = np.log10(L / char)
        X["log_nom_stress"] = np.log10(sig); X["log_nom_defl"] = np.log10(dfl)
        X["log_nom_fos"] = np.log10(Sy / sig)
        X["log_nom_strain_E"] = np.log10(sig / E)
    return pd.DataFrame(X)