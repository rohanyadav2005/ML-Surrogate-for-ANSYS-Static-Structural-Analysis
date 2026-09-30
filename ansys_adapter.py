"""Drive ANSYS Mechanical APDL (tested target: 2021 R2) in BATCH mode - no PyMAPDL needed.
One ANSYS launch solves a whole chunk of cases in an APDL *DO loop; results are appended to res.csv.

    python ansys_adapter.py --test            # 7 cases, prints ANSYS vs closed-form   <-- run this FIRST
    python generate_data.py --source ansys --n 600

Extraction rules (kept identical for every case, and they are what the ML model learns):
  * deformation = max total displacement (bending with midspan load: over nodes with y<=0, skips the load-point dent)
  * stresses    = max nodal (averaged) value over the INTERIOR only, z in [D, L-D] with D = largest section size
                  (skips end-face/support artefacts, Saint-Venant); midspan-load bending additionally uses y<=0.
                  Consequence: cantilever bending stress is read at z=D, i.e. slightly below the theoretical root value.
  * max_principal = max(|S1|,|S3|);  FoS = Sy / vonMises (computed here, ANSYS does not need Sy)
End faces are rigid (CERIG to a pilot node with a zero-mass MASS21) for tension, compression, torsion, cantilever
load and simply-supported ends.  NOTE: this script could not be run by its author (no ANSYS in the build sandbox)."""
import os, sys, shutil, subprocess, argparse, numpy as np, pandas as pd

DEFAULT_EXE = r"X:\ansys\ANSYS Inc\v212\ansys\bin\winx64\ANSYS212.exe"
LT = {"tensile": 1, "compressive": 2, "bending": 3, "torsion": 4}
BC = {"fixed-free": 1, "simply-supported": 2, "fixed-fixed": 3}

APDL = r"""/NOPR
*DIM,C,ARRAY,__N__,12
*VREAD,C(1,1),cases,txt,,JIK,12,__N__
(12F20.8)
*DO,IC,1,__N__
P_ID=C(IC,1)
P_SEC=C(IC,2)
P_LT=C(IC,3)
P_BC=C(IC,4)
P_L=C(IC,5)
P_B=C(IC,6)
P_H=C(IC,7)
P_D=C(IC,8)
P_E=C(IC,9)
P_NU=C(IC,10)
P_F=C(IC,12)
P_UM=-1
P_VM=-1
P_S1=-1
P_S3=-1
FINISH
/PREP7
ALLSEL,ALL
DDELE,ALL,ALL
FDELE,ALL,ALL
CEDELE,ALL
VCLEAR,ALL
VDELE,ALL,,,1
EDELE,ALL
NDELE,ALL
ET,1,SOLID186
ET,2,MASS21
R,1,1E-6,1E-6,1E-6,1E-6,1E-6,1E-6
MPDELE,ALL,1
MP,EX,1,P_E
MP,PRXY,1,P_NU
*IF,P_SEC,EQ,0,THEN
BLOCK,-P_B/2,P_B/2,-P_H/2,P_H/2,0,P_L
P_DM=MIN(P_B,P_H)
P_DX=MAX(P_B,P_H)
P_YT=P_H/2
*ELSE
CYLIND,0,P_D/2,0,P_L
P_DM=P_D
P_DX=P_D
P_YT=P_D/2
*ENDIF
P_ES=P_DM/__DIV__
P_NZ=MIN(150,MAX(8,NINT(P_L/(P_DM/2))))
P_NZ=2*NINT(P_NZ/2)
ESIZE,P_ES
LSEL,S,LENGTH,,P_L*0.999,P_L*1.001
LESIZE,ALL,,,P_NZ
*IF,P_SEC,EQ,0,THEN
P_NB=2*MAX(1,NINT(P_B/P_ES/2))
P_NH=2*MAX(1,NINT(P_H/P_ES/2))
LSEL,S,LENGTH,,P_B*0.999,P_B*1.001
LESIZE,ALL,,,P_NB
LSEL,S,LENGTH,,P_H*0.999,P_H*1.001
LESIZE,ALL,,,P_NH
*ENDIF
ALLSEL,ALL
TYPE,1
MAT,1
REAL,1
VSWEEP,ALL
ALLSEL,ALL
*GET,P_NM,NODE,0,NUM,MAX
P_N1=P_NM+1
P_N2=P_NM+2
*IF,P_BC,EQ,1,THEN
N,P_N1,0,0,P_L
TYPE,2
REAL,1
E,P_N1
NSEL,S,LOC,Z,P_L
NSEL,U,NODE,,P_N1,P_N2
NSEL,A,NODE,,P_N1
CERIG,P_N1,ALL,ALL
NSEL,S,LOC,Z,0
D,ALL,ALL,0
ALLSEL,ALL
*IF,P_LT,EQ,1,THEN
F,P_N1,FZ,P_F
*ELSEIF,P_LT,EQ,2
F,P_N1,FZ,-P_F
*ELSEIF,P_LT,EQ,3
F,P_N1,FY,-P_F
*ELSE
F,P_N1,MZ,P_F
*ENDIF
*ENDIF
*IF,P_BC,EQ,2,THEN
N,P_N1,0,0,0
N,P_N2,0,0,P_L
TYPE,2
REAL,1
E,P_N1
E,P_N2
NSEL,S,LOC,Z,0
NSEL,U,NODE,,P_N1,P_N2
NSEL,A,NODE,,P_N1
CERIG,P_N1,ALL,ALL
NSEL,S,LOC,Z,P_L
NSEL,U,NODE,,P_N1,P_N2
NSEL,A,NODE,,P_N2
CERIG,P_N2,ALL,ALL
ALLSEL,ALL
D,P_N1,UX,0
D,P_N1,UY,0
D,P_N1,UZ,0
D,P_N1,ROTZ,0
D,P_N2,UX,0
D,P_N2,UY,0
*ENDIF
*IF,P_BC,EQ,3,THEN
NSEL,S,LOC,Z,0
D,ALL,ALL,0
NSEL,S,LOC,Z,P_L
D,ALL,ALL,0
ALLSEL,ALL
*ENDIF
*IF,P_BC,GT,1,THEN
NSEL,S,LOC,Z,P_L/2-P_L*1E-4,P_L/2+P_L*1E-4
NSEL,R,LOC,Y,P_YT-P_ES*0.6,P_YT+P_ES*0.01
NSEL,R,LOC,X,-P_ES*0.6,P_ES*0.6
*GET,P_CNT,NODE,0,COUNT
F,ALL,FY,-P_F/P_CNT
*ENDIF
ALLSEL,ALL
/SOLU
ANTYPE,STATIC
SOLVE
FINISH
/POST1
SET,LAST
ALLSEL,ALL
*IF,P_LT,EQ,3,AND,P_BC,GT,1,THEN
NSEL,S,LOC,Y,-1E6,0
*ENDIF
NSORT,U,SUM
*GET,P_UM,SORT,,MAX
NSEL,U,NODE,,P_N1,P_N2
NSEL,R,LOC,Z,P_DX,P_L-P_DX
NSORT,S,EQV
*GET,P_VM,SORT,,MAX
NSORT,S,1
*GET,P_S1,SORT,,MAX
NSORT,S,3
*GET,P_S3,SORT,,MIN
*CFOPEN,res,csv,,APPEND
*VWRITE,P_ID,P_UM,P_VM,P_S1,P_S3
(F12.0,',',E16.8,',',E16.8,',',E16.8,',',E16.8)
*CFCLOS
ALLSEL,ALL
FINISH
*ENDDO
"""

def _cases_txt(df):
    z = lambda c: df[c].fillna(0).astype(float).values
    rows = np.column_stack([df.index.values.astype(float), (df.section == "circular").astype(float).values,
                            df.load_type.map(LT).values, df.bc.map(BC).values,
                            z("L"), z("b"), z("h"), z("d"), z("E"), z("nu"), z("Sy"), z("F")])
    return "\n".join("".join(f"{v:20.8f}" for v in r) for r in rows) + "\n"

def run_ansys_batch(df, exe=None, workdir="ansys_work", chunk=20, np_cores=2, divisions=4, timeout=3 * 3600):
    exe = exe or os.environ.get("ANSYS_EXE", DEFAULT_EXE)
    if not os.path.exists(exe): sys.exit(f"ANSYS executable not found:\n  {exe}\nFind ANSYS212.exe and pass --ansys_exe \"<full path>\"")
    workdir = os.path.abspath(workdir); os.makedirs(workdir, exist_ok=True); frames = []
    import glob
    for s in range(0, len(df), chunk):
        part = df.iloc[s:s + chunk]; txt = _cases_txt(part)
        cf, rf = os.path.join(workdir, f"cases_{s}.txt"), os.path.join(workdir, f"res_{s}.csv")
        if os.path.exists(cf) and os.path.exists(rf) and open(cf).read() == txt:       # resume: chunk already finished
            r = pd.read_csv(rf, header=None, names=["id", "u", "vm", "s1", "s3"])
            if len(r) == len(part):
                r["id"] = r["id"].round().astype(int); frames.append(r); print(f"[ANSYS] cases {s}-{s + len(part) - 1}: reused", flush=True); continue
        for f in glob.glob(os.path.join(workdir, "*.lock")) + [os.path.join(workdir, "res.csv")]:
            if os.path.exists(f): os.remove(f)
        open(os.path.join(workdir, "cases.txt"), "w").write(txt)
        open(os.path.join(workdir, "run.inp"), "w").write(APDL.replace("__N__", str(len(part))).replace("__DIV__", str(divisions)))
        print(f"[ANSYS] cases {s}-{s + len(part) - 1} of {len(df)} ...", flush=True)
        subprocess.run([exe, "-b", "-np", str(np_cores), "-dir", workdir, "-j", "job", "-i", "run.inp", "-o", "run.out"],
                       timeout=timeout, cwd=workdir)
        rp = os.path.join(workdir, "res.csv")
        if not os.path.exists(rp):
            print("  no results produced - open ansys_work\\run.out and look for the first 'ERROR'"); continue
        r = pd.read_csv(rp, header=None, names=["id", "u", "vm", "s1", "s3"]); r["id"] = r["id"].round().astype(int)
        frames.append(r)
        if len(r) == len(part): shutil.copy(rp, rf); open(cf, "w").write(txt)          # only complete chunks are cached
        else: print(f"  chunk incomplete ({len(r)}/{len(part)}) - it will be redone on the next run")
    if not frames or all((f.vm == 0).all() for f in frames):
        import glob, re
        print("\n[ANSYS] every case returned zeros -> solve failed. First ANSYS errors:")
        seen = 0
        for fn in sorted(glob.glob(os.path.join(workdir, "*.err"))):
            t = open(fn, errors="ignore").read().split("*** ")
            for blk in t:
                if blk.startswith("ERROR"):
                    print("  " + " ".join(blk.split())[:300]); seen += 1
                    if seen >= 4: break
            if seen >= 4: break
    R = pd.concat(frames) if frames else pd.DataFrame(columns=["id", "u", "vm", "s1", "s3"])
    R = R.drop_duplicates("id").set_index("id").reindex(df.index)
    bad = (R.u < 0) | (R.vm <= 0) | R.u.isna()
    out = pd.DataFrame({"max_deformation_mm": R.u, "von_mises_MPa": R.vm,
                        "max_principal_MPa": np.maximum(R.s1.abs(), R.s3.abs()), "factor_of_safety": df.Sy / R.vm}, index=df.index)
    out[bad] = np.nan
    print(f"[ANSYS] {int((~bad).sum())}/{len(df)} cases OK"); return out

def selftest(exe):
    from physics import solve
    base = dict(L=300., E=200e3, nu=0.3, Sy=250.)
    C = [("rectangular", "tensile", "fixed-free", 20000), ("circular", "compressive", "fixed-free", 20000),
         ("rectangular", "bending", "fixed-free", 500), ("circular", "bending", "simply-supported", 500),
         ("rectangular", "bending", "fixed-fixed", 800), ("circular", "torsion", "fixed-free", 2e5),
         ("rectangular", "torsion", "fixed-free", 2e5)]
    df = pd.DataFrame([dict(section=s, load_type=l, bc=b, F=F, b=30. if s == "rectangular" else np.nan,
                            h=40. if s == "rectangular" else np.nan, d=40. if s == "circular" else np.nan, **base) for s, l, b, F in C])
    A = run_ansys_batch(df, exe=exe, chunk=10, workdir="ansys_test")
    R = pd.DataFrame(solve(df.section.values, df.load_type.values, df.bc.values, df.L.values, df.b.fillna(0).values,
                           df.h.fillna(0).values, df.d.fillna(0).values, df.E.values, df.nu.values, df.Sy.values, df.F.values))
    pd.set_option("display.width", 200)
    for t in ["max_deformation_mm", "von_mises_MPa", "max_principal_MPa"]:
        print("\n", t); print(pd.DataFrame({"case": [f"{s[:4]}/{l[:4]}/{b}" for s, l, b, _ in C], "ANSYS": A[t], "formula": R[t],
                                            "diff_%": (A[t] - R[t]) / R[t] * 100}).round(4).to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true"); ap.add_argument("--ansys_exe", default=None)
    a = ap.parse_args()
    if a.test: selftest(a.ansys_exe)