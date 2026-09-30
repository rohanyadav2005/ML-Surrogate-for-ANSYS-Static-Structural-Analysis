"""Preprocess -> features -> tune -> k-fold CV -> test metrics -> uncertainty -> residuals -> physics validation."""
import json, os, argparse, warnings, numpy as np, pandas as pd, joblib, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from physics import TARGETS
from features import build_features
warnings.filterwarnings("ignore")

SIGN = dict(max_deformation_mm=1, von_mises_MPa=1, max_principal_MPa=1, factor_of_safety=-1)  # linear statics: response ~ F^SIGN
UNITS = dict(max_deformation_mm="mm", von_mises_MPa="MPa", max_principal_MPa="MPa", factor_of_safety="-")
SPACE = dict(n_estimators=[300, 500, 800], max_depth=[4, 6, 8, 10], learning_rate=[0.03, 0.06, 0.1],
             subsample=[0.7, 0.9, 1.0], colsample_bytree=[0.6, 0.8, 1.0], min_child_weight=[1, 3, 5], reg_lambda=[0.1, 1, 5])

def metrics(y, p):
    ape = np.abs(p - y) / np.maximum(np.abs(y), 1e-12) * 100
    return dict(MAE=mean_absolute_error(y, p), RMSE=mean_squared_error(y, p) ** .5, R2=r2_score(y, p),
                MAPE_pct=ape.mean(), median_APE_pct=np.median(ape), p95_APE_pct=np.percentile(ape, 95))

def main(a):
    os.makedirs("models", exist_ok=True); os.makedirs("reports", exist_ok=True)
    df = pd.read_csv(a.data).dropna(subset=TARGETS)
    if a.max_defl_frac > 0:                       # small-displacement validity of linear statics
        keep = df.max_deformation_mm <= a.max_defl_frac * df.L
        print(f"dropped {int((~keep).sum())}/{len(df)} cases with deformation > {a.max_defl_frac:.0%} of length"); df = df[keep]
    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df.load_type + df.bc)
    Ytr, Yte = tr[TARGETS].values, te[TARGETS].values
    Xtr, Xte = build_features(tr), build_features(te)
    Xtr_raw, Xte_raw = build_features(tr, False), build_features(te, False)
    cols = list(Xtr.columns); models, lo_hi, best, report, conf = {}, {}, {}, {}, {}
    sub = np.random.default_rng(0).choice(len(tr), min(6000, len(tr)), replace=False)

    for j, t in enumerate(TARGETS):
        off_tr, off_te = SIGN[t] * np.log10(tr.F.values), SIGN[t] * np.log10(te.F.values)
        y = np.log10(Ytr[:, j]) - off_tr             # response per unit load (log10): exact load scaling built in
        rs = RandomizedSearchCV(XGBRegressor(tree_method="hist", n_jobs=-1, random_state=0), SPACE, n_iter=a.iters,
                                cv=3, scoring="neg_mean_squared_error", random_state=0, n_jobs=1)
        rs.fit(Xtr.iloc[sub], y[sub]); best[t] = rs.best_params_
        m = XGBRegressor(tree_method="hist", n_jobs=-1, random_state=0, **best[t])
        cvp = cross_val_predict(m, Xtr, y, cv=KFold(5, shuffle=True, random_state=0)) + off_tr
        cv_r2 = r2_score(np.log10(Ytr[:, j]), cvp); cv_ape = np.median(np.abs(10 ** cvp / Ytr[:, j] - 1) * 100)
        m.fit(Xtr, y); models[t] = m
        # uncertainty: 5%/95% quantile models -> 90% prediction interval
        q = XGBRegressor(tree_method="hist", objective="reg:quantileerror", quantile_alpha=np.array([.05, .95]),
                         n_jobs=-1, random_state=0, **{**best[t], "n_estimators": min(best[t]["n_estimators"], 400)})
        idx = np.random.default_rng(3).permutation(len(Xtr)); cal, fit = idx[:int(.15 * len(idx))], idx[int(.15 * len(idx)):]
        q.fit(Xtr.iloc[fit], y[fit]); lo_hi[t] = q                     # conformalised quantile regression (CQR)
        qc = np.sort(q.predict(Xtr.iloc[cal]), axis=1); sc = np.maximum(qc[:, 0] - y[cal], y[cal] - qc[:, 1])
        qhat = float(np.sort(sc)[min(int(np.ceil((len(cal) + 1) * .9)), len(sc)) - 1]); conf[t] = qhat
        pred = 10 ** (m.predict(Xte) + off_te); qq = np.sort(q.predict(Xte), axis=1); qi = 10 ** (np.column_stack([qq[:, 0] - qhat, qq[:, 1] + qhat]) + off_te[:, None])
        cover = float(np.mean((Yte[:, j] >= qi[:, 0]) & (Yte[:, j] <= qi[:, 1])))
        # ablation: no physics-informed features
        r = XGBRegressor(tree_method="hist", n_jobs=-1, random_state=0, **best[t]).fit(Xtr_raw, y)
        abl = metrics(Yte[:, j], 10 ** (r.predict(Xte_raw) + off_te))
        report[t] = dict(units=UNITS[t], best_params={k: (v.item() if hasattr(v, "item") else v) for k, v in best[t].items()},
                         cv5_R2_log=dict(mean=cv_r2, std=0.0), cv5_median_APE_pct=cv_ape, test=metrics(Yte[:, j], pred),
                         test_log10_R2=r2_score(np.log10(Yte[:, j]), np.log10(pred)),
                         interval90_coverage=cover, ablation_raw_features_only=abl)
        # residual analysis
        res = pred - Yte[:, j]; rel = res / Yte[:, j] * 100
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        ax[0].loglog(Yte[:, j], pred, ".", ms=2, alpha=.4); lim = [Yte[:, j].min(), Yte[:, j].max()]
        ax[0].plot(lim, lim, "r-", lw=1); ax[0].set(xlabel=f"Reference {t}", ylabel="ML prediction", title="Parity")
        ax[1].semilogx(Yte[:, j], rel, ".", ms=2, alpha=.4); ax[1].axhline(0, c="r")
        ax[1].set(xlabel="Reference", ylabel="Residual (%)", title="Relative residual vs reference", ylim=(-10, 10))
        ax[2].hist(np.clip(rel, -10, 10), 80); ax[2].set(xlabel="Residual (%)", title="Residual distribution")
        plt.tight_layout(); plt.savefig(f"reports/residuals_{t}.png", dpi=110); plt.close()
        T = report[t]
        print(f"{t:20s} CV R2(log)={cv_r2:.4f}  test R2(log)={T['test_log10_R2']:.4f}  R2(phys)={T['test']['R2']:.3f}  "
              f"medAPE={T['test']['median_APE_pct']:.2f}%  MAPE={T['test']['MAPE_pct']:.2f}%  p95={T['test']['p95_APE_pct']:.1f}%  cov90={cover:.2f}  [raw-only MAPE={abl['MAPE_pct']:.1f}%]")

    # per-case error table + physics validation
    P = pd.DataFrame(_pred(models, te), index=te.index)
    out = te.copy()
    for t in TARGETS: out["ref_" + t] = te[t]; out["ml_" + t] = P[t]; out["err_pct_" + t] = (P[t] - te[t]) / te[t] * 100
    out.drop(columns=TARGETS).to_csv("reports/test_predictions_vs_reference.csv", index=False)
    report["by_case_MAPE_pct"] = {t: out.assign(k=out.load_type + "/" + out.bc + "/" + out.section).groupby("k")["err_pct_" + t]
                                  .apply(lambda s: s.abs().mean()).round(3).to_dict() for t in TARGETS}
    report["physics_validation"] = physics_validation(models, te)
    json.dump(report, open("reports/metrics.json", "w"), indent=2, default=float)
    joblib.dump(dict(models=models, intervals=lo_hi, conf_q=conf, columns=cols), "models/surrogate.joblib")
    print(json.dumps(report["physics_validation"], indent=2))

def _pred(models, d):
    X = build_features(d); return {t: 10 ** (models[t].predict(X) + SIGN[t] * np.log10(d.F.values)) for t in TARGETS}

def physics_validation(models, te, n=2000):
    d = te.sample(min(n, len(te)), random_state=1).reset_index(drop=True); base = _pred(models, d)
    v = {}
    d2 = d.assign(F=d.F * 2); p2 = _pred(models, d2)                  # 1) linearity in load
    v["load_linearity_median_dev_pct"] = {t: float(np.median(np.abs(p2[t] / base[t] / (0.5 if t == "factor_of_safety" else 2) - 1) * 100)) for t in TARGETS}
    v["fos_consistency_median_err_pct"] = float(np.median(np.abs(base["factor_of_safety"] - d.Sy / base["von_mises_MPa"]) / (d.Sy / base["von_mises_MPa"]) * 100))
    ratio = base["von_mises_MPa"] / base["max_principal_MPa"].clip(1e-9)   # 2) stress-state relation
    th = np.where(d.load_type == "torsion", np.sqrt(3), 1.0); m = d.load_type != "compressive"
    v["vonMises_to_principal_ratio_median_err_pct"] = float(np.median(np.abs(ratio[m] / th[m] - 1) * 100))
    d3 = d.assign(E=d.E * 1.2); p3 = _pred(models, d3)                  # 3) stiffer material deflects less
    v["monotonic_E_up_deflection_down_pass_rate"] = float(np.mean(p3["max_deformation_mm"] < base["max_deformation_mm"]))
    bd = d[d.load_type == "bending"].reset_index(drop=True)             # 4) BC stiffness ordering
    if len(bd):
        pr = {bc: _pred(models, bd.assign(bc=bc))["max_deformation_mm"] for bc in ["fixed-free", "simply-supported", "fixed-fixed"]}
        v["bending_BC_order_ff>ss>fixedfixed_pass_rate"] = float(np.mean((pr["fixed-free"] > pr["simply-supported"]) & (pr["simply-supported"] > pr["fixed-fixed"])))
    return v

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/dataset.csv"); ap.add_argument("--iters", type=int, default=10); ap.add_argument("--max_defl_frac", type=float, default=0.1)
    main(ap.parse_args())