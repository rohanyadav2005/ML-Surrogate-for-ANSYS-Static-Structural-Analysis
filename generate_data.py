"""Automated parametric DOE.  --source analytical (default stand-in) | ansys (PyMAPDL, see ansys_adapter.py)."""
import argparse, numpy as np, pandas as pd
from scipy.stats import qmc
from physics import *

def sample(n, seed=0):
    rng = np.random.default_rng(seed)
    per = n // (len(SECTIONS) * 5)      # combos: 2 sections x 5 (load,bc) cases (torsion/axial: one BC)
    cases = [("tensile", "fixed-free"), ("compressive", "fixed-free"), ("torsion", "fixed-free"),
             ("bending", "fixed-free"), ("bending", "simply-supported"), ("bending", "fixed-fixed")]
    rows = []
    for sec in SECTIONS:
        for k, (lt, bc) in enumerate(cases):
            m = per * 5 // 6 + 1
            u = qmc.Sobol(9, scramble=True, seed=seed + k + 17 * len(rows)).random(m)   # space-filling DOE
            lg = lambda lo, hi, x: 10 ** (np.log10(lo) + x * (np.log10(hi) - np.log10(lo)))
            b, h, d = lg(10, 100, u[:, 1]), lg(10, 100, u[:, 2]), lg(10, 100, u[:, 1])
            Lmin = 5 * (d if sec == "circular" else np.maximum(b, h))
            L = np.maximum(lg(100, 1500, u[:, 0]), Lmin)
            F = {"tensile": lg(1e2, 1e5, u[:, 6]), "compressive": lg(1e2, 1e5, u[:, 6]),
                 "bending": lg(5e1, 1e4, u[:, 6]), "torsion": lg(1e4, 1e7, u[:, 6])}[lt]
            rows.append(pd.DataFrame(dict(
                section=sec, load_type=lt, bc=bc, L=L, b=b, h=h, d=d,
                E=60e3 + u[:, 3] * 160e3, nu=0.25 + u[:, 4] * 0.10, Sy=100 + u[:, 5] * 900, F=F)))
    df = pd.concat(rows, ignore_index=True)
    df.loc[df.section == "circular", ["b", "h"]] = np.nan
    df.loc[df.section == "rectangular", "d"] = np.nan
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24000); ap.add_argument("--source", default="analytical")
    ap.add_argument("--noise", type=float, default=0.0, help="optional relative scatter emulating mesh error (0 = none)")
    ap.add_argument("--out", default="data/dataset.csv"); ap.add_argument("--ansys_exe", default=None); a = ap.parse_args()
    df = sample(a.n)
    if a.source == "ansys":
        from ansys_adapter import run_ansys_batch
        res = run_ansys_batch(df, exe=a.ansys_exe)
    else:
        res = solve(df.section.values, df.load_type.values, df.bc.values, df.L.values,
                    df.b.fillna(0).values, df.h.fillna(0).values, df.d.fillna(0).values,
                    df.E.values, df.nu.values, df.Sy.values, df.F.values)
        res = pd.DataFrame(res)
        if a.noise > 0:
            rng = np.random.default_rng(1)
            for t in TARGETS[:3]: res[t] *= 1 + a.noise * rng.standard_normal(len(res))
            res["factor_of_safety"] = df.Sy / res["von_mises_MPa"]
    import os; os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.concat([df, res], axis=1).to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(df)} rows, source={a.source}, noise={a.noise}")