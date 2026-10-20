"""Physics observables computed directly from (possibly coarse) energy grids.

Every function takes linear energies on the zero-padded grid and the pooling
factor f of that grid, so HR, LR and SR images are measured with the same code
and the same physical cell coordinates.

Jets (qg), treating each pixel as a massless constituent with pT = pixel value:
    sum_<channel>, sum_all, jet_pt, jet_mass, girth (pT-weighted mean dR), dR_std (pT-weighted
    spread of dR around the girth), ptD, n_hits (cells of the channel sum), n_hits_<channel>,
    tau21 (N-subjettiness)
    radial_profile: fraction of pT in annuli of dR around the jet centroid
n_hits* and ptD depend on the cell size, so on an LR grid they are not comparable with HR
(GRID_DEPENDENT).
Showers (calo):
    e_total, e_ratio = E_dep / E_inc, depth_centroid/width (layers),
    r_centroid/width (mm), x/y_centroid (mm), frac_early/mid/late, n_hits
    longitudinal / radial energy profiles
"""
import numpy as np

JET_RADIAL_EDGES = np.linspace(0.0, 0.8, 9)
GRID_DEPENDENT = ("n_hits", "ptD")  # prefixes of observables that change with the cell size


def _centres(n, f, offset=0.0):
    return (np.arange(n) + 0.5) * f - 0.5 - offset


def is_hit(E, thr):
    """Cell above the readout threshold. The decoder writes hit cells at no less than the threshold
    and the log/exp round trip can land a few ulp below it, so compare with a small tolerance."""
    return E > thr * (1 - 1e-4)


# ---------------------------------------------------------------------- jets
def _tau(w, x, n, iters=10):
    """sum_i w_i min_k dR(i, axis_k) with axes from pT-weighted k-means."""
    if len(w) == 0:
        return 0.0
    order = np.argsort(-w)
    axes = x[order[np.arange(n) % len(w)]].copy()
    for _ in range(iters):
        d = np.sqrt(((x[:, None, :] - axes[None]) ** 2).sum(-1))
        a = d.argmin(1)
        for k in range(n):
            m = a == k
            if m.any():
                axes[k] = (w[m, None] * x[m]).sum(0) / w[m].sum()
    d = np.sqrt(((x[:, None, :] - axes[None]) ** 2).sum(-1))
    return float((w * d.min(1)).sum())


def jet_observables(E, f, dcfg, nsub_max=0, chunk=500):
    N, C, H, W = E.shape
    pix, c0, thr = dcfg["pixel_size"], dcfg["center_index"], dcfg["occupancy_threshold"]
    eta = _centres(H, f[0], c0) * pix
    phi = _centres(W, f[1], c0) * pix
    ETA, PHI = np.meshgrid(eta, phi, indexing="ij")
    out = {f"sum_{ch}": E[:, i].reshape(N, -1).sum(1).astype(np.float64) for i, ch in enumerate(dcfg["channels"])}
    keys = ("sum_all", "jet_pt", "jet_mass", "girth", "dR_std", "ptD", "n_hits")
    for k in keys:
        out[k] = np.zeros(N)
    prof = np.zeros((N, len(JET_RADIAL_EDGES) - 1))
    for s in range(0, N, chunk):
        P = E[s:s + chunk].sum(1).astype(np.float64)
        tot = P.sum((1, 2))
        safe = np.maximum(tot, 1e-12)
        px, py = (P * np.cos(PHI)).sum((1, 2)), (P * np.sin(PHI)).sum((1, 2))
        pz, e = (P * np.sinh(ETA)).sum((1, 2)), (P * np.cosh(ETA)).sum((1, 2))
        ce = (P * ETA).sum((1, 2)) / safe
        cp = (P * PHI).sum((1, 2)) / safe
        dR = np.sqrt((ETA[None] - ce[:, None, None]) ** 2 + (PHI[None] - cp[:, None, None]) ** 2)
        sl = slice(s, s + len(P))
        out["sum_all"][sl] = tot
        out["jet_pt"][sl] = np.hypot(px, py)
        out["jet_mass"][sl] = np.sqrt(np.clip(e ** 2 - px ** 2 - py ** 2 - pz ** 2, 0, None))
        g = (P * dR).sum((1, 2)) / safe
        out["girth"][sl] = g
        out["dR_std"][sl] = np.sqrt(np.clip((P * dR ** 2).sum((1, 2)) / safe - g ** 2, 0, None))
        out["ptD"][sl] = np.sqrt((P ** 2).sum((1, 2))) / safe
        out["n_hits"][sl] = is_hit(P, thr).sum((1, 2))
        b = np.digitize(dR, JET_RADIAL_EDGES) - 1
        for k in range(prof.shape[1]):
            prof[sl, k] = np.where(b == k, P, 0).sum((1, 2)) / safe
    for i, ch in enumerate(dcfg["channels"]):  # a missing channel is invisible in the summed count
        out[f"n_hits_{ch}"] = np.concatenate([is_hit(E[s:s + chunk, i], thr).reshape(-1, H * W).sum(1)
                                              for s in range(0, N, chunk)]).astype(np.float64)
    if nsub_max:
        n = min(N, nsub_max)
        t1, t2 = np.zeros(n), np.zeros(n)
        R0 = 0.4
        for i in range(n):
            P = E[i].sum(0)
            m = is_hit(P, thr)
            w = P[m]
            if len(w) > 100:
                keep = np.argsort(-w)[:100]
            else:
                keep = slice(None)
            w = w[keep]
            x = np.stack([ETA[m][keep], PHI[m][keep]], 1)
            norm = max(w.sum() * R0, 1e-12)
            t1[i], t2[i] = _tau(w, x, 1) / norm, _tau(w, x, 2) / norm
        out["tau21"] = np.where(t1 > 0, t2 / np.maximum(t1, 1e-12), 0.0)
    profiles = {"radial_profile": (prof, 0.5 * (JET_RADIAL_EDGES[1:] + JET_RADIAL_EDGES[:-1]), "dR")}
    return out, profiles


# ------------------------------------------------------------------- showers
def shower_observables(E, f, dcfg, e_inc):
    X = E[:, 0].astype(np.float64)  # (N, layer, angle, radius)
    N, L, A, R = X.shape
    fL, fA, fR = f
    layer_c = _centres(L, fL)
    edges = np.asarray(dcfg["r_edges"])[::fR]
    r_c = 0.5 * (edges[1:] + edges[:-1])
    a_c = (np.arange(A) + 0.5) * (2 * np.pi / A)
    tot = X.sum((1, 2, 3))
    safe = np.maximum(tot, 1e-12)
    e_layer = X.sum((2, 3))
    e_r = X.sum((1, 2))
    depth = (e_layer * layer_c).sum(1) / safe
    rc = (e_r * r_c).sum(1) / safe
    xy = X.sum(1)  # (N, A, R)
    out = {
        "e_total": tot,
        "e_ratio": tot / e_inc,
        "depth_centroid": depth,
        "depth_width": np.sqrt((e_layer * (layer_c - depth[:, None]) ** 2).sum(1) / safe),
        "r_centroid": rc,
        "r_width": np.sqrt((e_r * (r_c - rc[:, None]) ** 2).sum(1) / safe),
        "x_centroid": (xy * np.cos(a_c)[None, :, None] * r_c[None, None]).sum((1, 2)) / safe,
        "y_centroid": (xy * np.sin(a_c)[None, :, None] * r_c[None, None]).sum((1, 2)) / safe,
        "n_hits": (X > dcfg["voxel_threshold_mev"]).sum((1, 2, 3)).astype(np.float64),
    }
    group = np.floor(layer_c / 15).astype(int)  # layers 0-14 / 15-29 / 30-44
    for g, name in enumerate(("frac_early", "frac_mid", "frac_late")):
        out[name] = e_layer[:, group == g].sum(1) / safe
    profiles = {"longitudinal_profile": (e_layer / safe[:, None], layer_c, "layer"),
                "radial_profile": (e_r / safe[:, None], r_c, "r [mm]")}
    return out, profiles


def cell_spectrum(E, thr, max_cells=300000, seed=0):
    v = np.asarray(E).reshape(-1)
    v = v[v > thr]
    if len(v) > max_cells:
        v = np.random.default_rng(seed).choice(v, max_cells, replace=False)
    return np.log10(v)
