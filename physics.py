"""Closed-form (beam / Saint-Venant) reference solver.
Units: mm, N, MPa (N/mm^2), N*mm.  Used for (a) the stand-in data source,
(b) physics-informed features, (c) physics-based validation of the ML model."""
import numpy as np

LOADS = ["tensile", "compressive", "bending", "torsion"]
BCS = ["fixed-free", "simply-supported", "fixed-fixed"]
SECTIONS = ["rectangular", "circular"]
VALID_BC = {"tensile": ["fixed-free"], "compressive": ["fixed-free"],
            "bending": BCS, "torsion": ["fixed-free"]}
TARGETS = ["max_deformation_mm", "von_mises_MPa", "max_principal_MPa", "factor_of_safety"]

# Saint-Venant torsion coefficients for rectangles (a>=b): tau=T/(c1 a b^2), theta=TL/(c2 a b^3 G)
_AB = np.array([1, 1.5, 2, 3, 4, 6, 10, 1e6])
_C1 = np.array([.208, .231, .246, .267, .282, .299, .312, .333])
_C2 = np.array([.141, .196, .229, .263, .281, .299, .312, .333])

def section_props(section, b, h, d):
    """b=width, h=height (bending about the axis giving I=b h^3/12), d=diameter."""
    b, h, d = map(np.asarray, (b, h, d))
    rect = np.asarray(section) == "rectangular"
    A = np.where(rect, b * h, np.pi * d**2 / 4)
    I = np.where(rect, b * h**3 / 12, np.pi * d**4 / 64)
    c = np.where(rect, h / 2, d / 2)
    a_, s_ = np.maximum(b, h), np.minimum(b, h)
    with np.errstate(all="ignore"):
        ab = np.where(s_ > 0, a_ / np.where(s_ > 0, s_, 1), 1.0)
    c1, c2 = np.interp(ab, _AB, _C1), np.interp(ab, _AB, _C2)
    J = np.where(rect, c2 * a_ * s_**3, np.pi * d**4 / 32)          # torsion constant
    tors_W = np.where(rect, c1 * a_ * s_**2, np.pi * d**3 / 16)     # tau = T / W
    r_out = np.where(rect, 0.5 * np.hypot(b, h), d / 2)             # farthest point from axis
    kappa = np.where(rect, 5 / 6, 0.9)
    return dict(A=A, I=I, c=c, J=J, W=tors_W, r_out=r_out, kappa=kappa)

def solve(section, load_type, bc, L, b, h, d, E, nu, Sy, F):
    """F = force [N] (tensile/compressive/bending) or torque [N*mm] (torsion).
    Returns dict of arrays. max_principal = largest |principal stress| (max(|s1|,|s3|)); for compression this is the compressive principal magnitude."""
    L, E, nu, Sy, F = (np.asarray(x, float) for x in (L, E, nu, Sy, F))
    section, load_type, bc = (np.asarray(x) for x in (section, load_type, bc))
    p = section_props(section, b, h, d)
    G = E / (2 * (1 + nu))
    axial = (load_type == "tensile") | (load_type == "compressive")
    sign = np.where(load_type == "compressive", -1.0, 1.0)

    # axial
    sig_ax = sign * F / p["A"]
    def_ax = F * L / (p["A"] * E)
    # bending (Euler-Bernoulli + Timoshenko shear term), point load
    kb = np.select([bc == "fixed-free", bc == "simply-supported"], [3, 48], 192)
    ks = np.select([bc == "fixed-free", bc == "simply-supported"], [1.0, 0.25], 0.25)
    Mmax = F * L * np.select([bc == "fixed-free", bc == "simply-supported"], [1, .25], .125)
    def_b = F * L**3 / (kb * E * p["I"]) + ks * F * L / (p["kappa"] * G * p["A"])
    sig_b = Mmax * p["c"] / p["I"]
    # torsion
    tau = F / p["W"]
    def_t = F * L / (G * p["J"]) * p["r_out"]

    dfm = np.select([axial, load_type == "bending"], [np.abs(def_ax), def_b], def_t)
    vm = np.select([axial, load_type == "bending"], [np.abs(sig_ax), sig_b], np.sqrt(3) * tau)
    s1 = np.select([axial, load_type == "bending"], [np.abs(sig_ax), sig_b], tau)
    return dict(max_deformation_mm=dfm, von_mises_MPa=vm, max_principal_MPa=s1,
                factor_of_safety=Sy / np.maximum(vm, 1e-9))

def euler_buckling_load(section, bc, L, b, h, d, E):
    """Fixed-free column (K=2), weak axis. Used only as a validity warning for compression."""
    p = section_props(section, b, h, d)
    Imin = np.where(np.asarray(section) == "rectangular",
                    np.minimum(b, h) ** 3 * np.maximum(b, h) / 12, p["I"])
    return np.pi**2 * E * Imin / (2 * L) ** 2
