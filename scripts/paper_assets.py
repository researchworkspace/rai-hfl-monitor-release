"""Figures and tables of the paper, generated from the outputs of the pipeline (see README).

python -I scripts/paper_assets.py --root . --out paper

Reads <root>/outputs (trajectories, replay analyses, RQ2, per-round decisions, post-hoc analyses, decidability and
the overhead measurements) and writes <out>/figures/*.pdf and <out>/tables/*.tex. `--only <job> ...` regenerates
selected items.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# ------------------------------------------------------------------ style

COL = {"P1": "#8C8C8C", "P2": "#E69F00", "P2s": "#56B4E9", "P3": "#0072B2", "P4": "#D55E00", "P4R": "#B04A8F",
       "P4-hold": "#D55E00", "P4-deploy": "#CC79A7", "P4-reviewer": "#009E73", "P4-point": "#000000"}
DS = {"acs": "Census", "isic": "Dermatology"}
DS_LONG = {"acs": "Census federation (ACS Income)", "isic": "Dermatology federation (Fed-ISIC2019)"}
PL = {"P1": "Central", "P2": "Uniform", "P2s": "Uniform-scoped", "P2c": "Uniform+clients", "P3": "Pooled",
      "P4": "Tiered", "P4-point": "Tiered, point estimates",
      "P4R": "Tiered-restricted", "P4-hold": "Tiered, fail-safe default", "P4-deploy": "Tiered, undecided deploy",
      "P4-reviewer": "Tiered + oracle approver"}
REQ = {"R1": "SP gap", "R2": "EO gap", "R3": "Worst-group accuracy", "R4": "Privacy budget", "R5": "Robustness",
       "R6": "Explanation stability", "R7": "Report integrity"}
DOM = {"acs": ["New England", "Mid-Atlantic", "E. N. Central", "W. N. Central", "South Atlantic", "E. S. Central",
               "W. S. Central", "Mountain", "Pacific"],
       "isic": ["Barcelona", "Molemax", "Modern", "Rosendahl", "MSK", "Dias"]}
DS_COL = {"acs": "#0072B2", "isic": "#D55E00"}
INJ = ["V1", "V2", "V3-f0.1", "V3-f0.2", "V4", "V5", "V6"]
INJ_LAB = {"V0": "Control", "V1": "Domain drift", "V2": "Simpson shift", "V3-f0.1": "Poisoning 10%",
           "V3-f0.2": "Poisoning 20%", "V4": "Lying clients", "V5": "Noise cut", "V6": "Acquisition shift"}
TEX_INJ = {k: v.replace("%", "\\%") for k, v in INJ_LAB.items()}
LEARN = {"fedavg": "FedAvg", "fairfed": "FairFed", "eafclc": "EA-FCL-c"}
W = 5.4  # acmsmall text width, inches


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.8,
        "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.5,
        "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def panel(ax, letter: str, title: str) -> None:
    ax.set_title(f"({letter}) {title}", loc="left", fontsize=7.8)


# ------------------------------------------------------------------ data access

class Data:
    def __init__(self, root: Path):
        self.root = root
        o = root / "outputs"
        self.o = o
        self.an = o / "analysis"
        self.ph = o / "posthoc"
        sys.path.insert(0, str(root / "src"))

    def th(self, ds: str):
        from raimon.replay import Thresholds
        plan = {"acs": "configs/acs.json", "isic": "configs/isic.json"}[ds]
        return Thresholds(**json.loads((self.root / plan).read_text())["thresholds"])

    def traj(self, ds: str, name: str):
        from raimon.replay import Trajectory
        return Trajectory.load(self.o / "trajectories" / ds / name)

    def csv(self, rel: str) -> pd.DataFrame:
        return pd.read_csv(self.o / rel, keep_default_na=True)


# ------------------------------------------------------------------ Fig. 1 motivation (real trajectories)

def fig_motivation(d: Data, out: Path) -> dict:
    from raimon.replay import Evidence, Setting, build_checks
    from raimon.soundness import gt_parts
    facts = {}
    # Fixed axes positions so each panel sits under its column of the three-cases diagram (fig_cases.drawio:
    # tier labels 0-61, columns of 250 units starting at 61, total 809 units). The three plot areas have the same
    # width and the same offset in their column; titles start at the column's left edge. Saved without cropping.
    FH = 2.45                                       # inches; plot areas 1.23 in high, legend in two rows below
    fig = plt.figure(figsize=(W, FH))
    col = [(61 + 250 * k) / 809 for k in range(3)]
    off, aw, bot, hgt = 0.088, 0.200, 0.93 / FH, 1.23 / FH
    axs = [fig.add_axes([c + off, bot, aw, hgt]) for c in col]
    for c, t in zip(col, ["(a) Census, Simpson shift", "(b) Census, domain drift", "(c) Dermatology, round 40"]):
        fig.text(c + 0.004, bot + hgt + 0.1 / FH, t, fontsize=7.8, fontweight="bold", ha="left", va="bottom")
    rounds = np.arange(60)
    for ax, (name, ri, letter, title, target) in zip(axs[:2], [
            ("acs__fedavg__V2__s1", 0, "a", "Census, Simpson shift", None),
            ("acs__fedavg__V1__s1", 1, "b", "Census, domain drift", 7)]):
        tr = d.traj("acs", name)
        th = d.th("acs")
        _, _, gv, dv = gt_parts(tr, th)
        tau = [th.sp, th.eo][ri]
        ax.axvspan(tr.t0, 59, color="#FBEDE6", lw=0, zorder=0)
        for e in range(tr.n_edges):
            hl = e == target
            ax.plot(rounds, dv[:, e, ri], color=COL["P4"] if hl else "#9DB4C8", lw=1.3 if hl else 0.6,
                    alpha=1 if hl else 0.85, zorder=3 if hl else 2)
        ax.plot(rounds, gv[:, ri], color="#1B1B1B", lw=1.6, zorder=4)
        ax.axhline(tau, color="#C0392B", ls=(0, (3, 2)), lw=0.9, zorder=5)
        # above the line, at the left in (a) and at the right in (b), where no curve passes
        ax.text(1 if letter == "a" else 58, tau + 0.006, f"$\\tau$ = {tau:.2f}", color="#C0392B", fontsize=6.5,
                va="bottom", ha="left" if letter == "a" else "right")
        ax.text(tr.t0 + 1.5, 0.335, "injection", fontsize=6, color="#B5651D")
        ax.set_xlim(0, 59)
        ax.set_xlabel("FL round")
        ax.set_ylabel(["Statistical-parity gap", "Equalised-odds gap"][ri])
        post = slice(tr.t0, None)
        facts[name] = {"global_post_median": float(np.median(gv[post, ri])),
                       "domain_post_max": float(np.nanmax(np.delete(dv[post, :, ri], target, axis=1)
                                                          if target is not None else dv[post, :, ri])),
                       "target_post_median": None if target is None else float(np.median(dv[post, target, ri]))}
    axs[0].set_ylim(0, 0.36)
    axs[1].set_ylim(0, 0.36)
    h = [Line2D([], [], color="#1B1B1B", lw=1.6, label="global (pooled)"),
         Line2D([], [], color="#9DB4C8", lw=0.8, label="each domain"),
         Line2D([], [], color=COL["P4"], lw=1.3, label="Mountain division")]
    h.append(Line2D([], [], color="#C0392B", ls=(0, (3, 2)), lw=0.9, label="threshold $\\tau$"))
    # (c) small sites: P4's 95% intervals for EO at round 40 of a D2 control run
    tr = d.traj("isic", "isic__fedavg__V0__s1")
    th = d.th("isic")
    ev = Evidence(tr, 0)
    t = 40
    checks = {c["id"]: c for c in build_checks(Setting("P4"), t, ev, th)}
    labels, est, lo, hi = [], [], [], []
    for key, lab in [("R2/global", "global")] + [(f"R2/e{e}", DOM["isic"][e]) for e in range(tr.n_edges)]:
        c = checks.get(key)
        if c is None:
            continue
        labels.append(lab)
        est.append(c["estimate"])
        lo.append(max(c["lower"], 0))
        hi.append(min(c["upper"], 1))
    ax = axs[2]
    y = np.arange(len(labels))[::-1]
    for yi, e_, l_, h_ in zip(y, est, lo, hi):
        verdict = "pass" if h_ <= th.eo else ("fail" if l_ > th.eo else "inconclusive")
        c = {"pass": "#009E73", "fail": "#C0392B", "inconclusive": "#E69F00"}[verdict]
        ax.plot([l_, h_], [yi, yi], color=c, lw=2.2, solid_capstyle="round", clip_on=False)
        ax.plot(e_, yi, "o", color="white", mec=c, mew=1.1, ms=3.6, zorder=4, clip_on=False)
    ax.axvline(th.eo, color="#C0392B", ls=(0, (3, 2)), lw=0.9)
    ax.set_yticks(y, labels)
    ax.text(th.eo + 0.02, len(labels) - 0.55, f"$\\tau$ = {th.eo:.2f}", color="#C0392B", fontsize=6.5, va="top")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xlabel("Equalised-odds gap")
    ax.grid(axis="y", visible=False)
    h += [Line2D([], [], color=c, lw=2.2, label=l) for l, c in
          [("95% interval: pass", "#009E73"), ("95% interval: inconclusive", "#E69F00")]]
    # two rows of three, filled column by column: lines of (a) and (b) on top, threshold and intervals below
    order = [0, 3, 1, 4, 2, 5]
    fig.legend(handles=[h[i] for i in order], loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=6.3,
               handlelength=1.5, columnspacing=1.6, handletextpad=0.5)
    facts["isic_eo_round40"] = {lab: [round(a, 3), round(b, 3), round(c, 3)] for lab, a, b, c in zip(labels, est, lo, hi)}
    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(out / "fig_motivation.pdf")
    plt.close(fig)
    return facts


# ------------------------------------------------------------------ Fig. decision timelines

OUTCOME_CMAP = ListedColormap(["#EEF2F6", "#D62728", "#7FBF7B", "#F2B134"])
OUTCOME_LAB = ["deployed, clean", "deployed, violating", "stopped, violating", "stopped, clean (false block)"]


def fig_timeline(d: Data, out: Path) -> dict:
    facts = {}
    cases = [("acs", "acs__fedavg__V1__s1", 7), ("isic", "isic__fedavg__V1__s1", 2)]
    pl = [(p, PL[p]) for p in ["P1", "P2", "P3"]] + [("P4", "Tiered (fail-safe default)")]
    fig, axs = plt.subplots(2, 4, figsize=(W, 2.55), sharex=True, gridspec_kw={"height_ratios": [9, 6],
                                                                                 "hspace": 0.32, "wspace": 0.08})
    for r, (ds, name, target) in enumerate(cases):
        z = np.load(d.o / "decisions" / ds / f"{name}.npz")
        meta = json.loads((d.o / "decisions" / ds / f"{name}.json").read_text())
        bad = z["bad"]
        for c, (p, lab) in enumerate(pl):
            dec = z[f"dec_{p}-full-0-0-fail_safe"]
            stopped = dec != 0                     # hold, or escalate held by the fail-safe default
            code = np.where(~stopped, np.where(bad, 1, 0), np.where(bad, 2, 3))
            ax = axs[r, c]
            ax.imshow(code.T, aspect="auto", cmap=OUTCOME_CMAP, vmin=-0.5, vmax=3.5, interpolation="nearest",
                      extent=(-0.5, code.shape[0] - 0.5, code.shape[1] - 0.5, -0.5))
            ax.axvline(meta["t0"] - 0.5, color="#333333", lw=0.7, ls=(0, (2, 1.5)))
            ax.add_patch(plt.Rectangle((-0.5, target - 0.5), code.shape[0], 1, fill=False, ec="#222222", lw=0.6))
            ax.grid(False)
            ax.set_yticks(range(code.shape[1]), DOM[ds][:code.shape[1]], fontsize=5.4)
            if c:
                ax.set_yticklabels([])
            ax.tick_params(length=1.5)
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_linewidth(0.4)
            if r == 0:
                ax.set_title(lab, fontsize=6.9, loc="center")
            if r == 1:
                ax.set_xlabel("FL round")
            facts[f"{ds}/{p}"] = {k: int((code == i).sum()) for i, k in enumerate(OUTCOME_LAB)}
        axs[r, 0].set_ylabel(DS[ds], fontsize=6.8)
    fig.legend(handles=[Patch(fc=c, ec="#999999", lw=0.3, label=l) for c, l in zip(OUTCOME_CMAP.colors, OUTCOME_LAB)],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.13), fontsize=6.5, handlelength=1.2)
    fig.savefig(out / "fig_timeline.pdf")
    plt.close(fig)
    return facts


# ------------------------------------------------------------------ Fig. RQ1 exposure (dot plot)

def fig_exposure(d: Data, out: Path) -> None:
    t = d.csv("analysis/table_a.csv")
    t = t[t.injection != "V0"]
    p2s = rv_means(d, "failsafe", "P2s", "exposure", "b")
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.3), sharey=True, gridspec_kw={"wspace": 0.06})
    off = {"P1": 0.3, "P2": 0.15, "P2s": 0.0, "P3": -0.15, "P4": -0.3}
    mk = {"P1": "s", "P2": "D", "P2s": "d", "P3": "^", "P4": "o"}
    for ax, ds in zip(axs, ["acs", "isic"]):
        s = t[t.dataset == ds].groupby("injection")[["violating_candidates", "P1", "P2", "P3", "P4"]].mean()
        s["P2s"] = [p2s[(ds, i)] for i in s.index]
        y = np.arange(len(INJ))[::-1]
        for yi, inj in zip(y, INJ):
            ax.barh(yi, s.loc[inj, "violating_candidates"], height=0.78, color="#F3F3F3", ec="none", zorder=0)
            for p in off:
                v = s.loc[inj, p]
                ax.plot(max(v, 0.03), yi + off[p], mk[p], color=COL[p], ms=4.2 if p == "P4" else 3.6,
                        mec="white", mew=0.4, zorder=3, clip_on=False)
        ax.set_xscale("log")
        ax.set_xlim(0.03, 600)
        ax.set_xticks([0.03, 0.1, 1, 10, 100], ["0", "0.1", "1", "10", "100"])
        ax.set_yticks(y, [INJ_LAB[i] for i in INJ])
        ax.grid(axis="y", visible=False)
        panel(ax, "a" if ds == "acs" else "b", DS[ds])
    fig.supxlabel("Violating deployments per run, mean over learners and runs (log scale, 0 drawn at the left edge)",
                  fontsize=7, y=-0.04)
    h = [Line2D([], [], ls="", marker=mk[p], color=COL[p], label=lab, ms=4) for p, lab in
         [(p, PL[p]) for p in ["P1", "P2", "P2s", "P3", "P4"]]]
    h.append(Patch(fc="#F3F3F3", label="violating candidates (upper bound)"))
    fig.legend(handles=h, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.22), fontsize=6.1, handletextpad=0.3,
               columnspacing=0.7)
    fig.savefig(out / "fig_exposure.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ Fig. RQ2 soundness

def fig_soundness(d: Data, out: Path) -> dict:
    groups = [("acs", "V2", "R1", "LPGF", "(a) Census, Simpson shift, SP gap:\nlocal pass, global fail"),
              ("acs", "V1", "R2", "GPLF", "(b) Census, domain drift, EO gap:\nglobal pass, local fail"),
              ("isic", "V2", "R1", "LPGF", "(c) Dermatology, Simpson shift, SP gap:\nlocal pass, global fail"),
              ("isic", "V1", "R2", "GPLF", "(d) Dermatology, domain drift, EO gap:\nglobal pass, local fail")]
    src = {"acs": "rq2/acs/rq2_summary.csv", "isic": "rq2/isic/rq2_summary.csv"}
    fig, axs = plt.subplots(2, 2, figsize=(W, 2.75), sharey=True, gridspec_kw={"wspace": 0.12, "hspace": 0.75})
    axs = axs.ravel()
    facts = {}
    for ax, (ds, inj, req, typ, title) in zip(axs, groups):
        s = d.csv(src[ds]).set_index(["injection", "req", "type"]).loc[(inj, req, typ)]
        n = s["domain_rounds"]
        facts[f"{ds}/{inj}/{req}/{typ}"] = {"n": int(n)} | {p: int(s[f"{p}_caught"]) for p in ["P1", "P2", "P3", "P4"]}
        y = np.arange(4)[::-1]
        for yi, p in zip(y, ["P1", "P2", "P3", "P4"]):
            held, esc = s[f"{p}_held"] / n, s[f"{p}_escalated"] / n
            ax.barh(yi, held, color=COL[p], height=0.62)
            ax.barh(yi, esc, left=held, color="white", ec=COL[p], hatch="//////", lw=0.6, height=0.62)
            ax.text(min(held + esc, 1) + 0.03, yi, f"{100 * (held + esc):.0f}%", va="center", fontsize=6.2)
        ax.set_xlim(0, 1.32)
        ax.set_xticks([0, 0.5, 1], ["0", "50", "100%"])
        ax.set_yticks(y, [PL[p] for p in ["P1", "P2", "P3", "P4"]])
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{title} ({int(n):,} domain-rounds)", fontsize=6.6, loc="left", fontweight="normal")
    fig.supxlabel("Share of violating domain-rounds not deployed", fontsize=7, y=-0.02)
    fig.legend(handles=[Patch(fc="#777777", label="held (failing check)"),
                        Patch(fc="white", ec="#777777", hatch="//////", label="inconclusive (held by fail-safe default)")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1), fontsize=6.4)
    fig.savefig(out / "fig_soundness.pdf")
    plt.close(fig)
    return facts


# ------------------------------------------------------------------ Fig. trade-off (PH1)

def fig_tradeoff(d: Data, out: Path) -> None:
    m = d.csv("posthoc/rules.csv")
    m["n_eval"] = m.n_eval.astype(str)
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw={"wspace": 0.28})
    order = ["P1", "P2", "P2s", "P3", "P4-point", "P4-deploy", "P4-hold", "P4-reviewer"]
    lab = PL
    mk = {"P1": "s", "P2": "D", "P2s": "d", "P3": "^", "P4-point": "X", "P4-deploy": "v", "P4-hold": "o",
          "P4-reviewer": "P"}
    for ax, ds in zip(axs, ["acs", "isic"]):
        s = m[m.dataset == ds].set_index(["n_eval", "setting"])
        for p in order:
            a = s.loc[("all", p)]
            b = s.loc[("50", p)] if ("50", p) in s.index else None
            x0, y0 = 100 * a.fb_injected, a.exposure_injected
            if b is not None and (abs(b.fb_injected - a.fb_injected) > 0.01 or abs(b.exposure_injected - y0) > 0.3):
                ax.annotate("", xy=(100 * b.fb_injected, max(b.exposure_injected, 0.05)), xytext=(x0, max(y0, 0.05)),
                            arrowprops=dict(arrowstyle="->", color=COL[p], lw=0.6, alpha=0.6, shrinkA=3, shrinkB=3))
                ax.plot(100 * b.fb_injected, max(b.exposure_injected, 0.05), mk[p], mfc="white", mec=COL[p], mew=0.9,
                        ms=4.2, zorder=3)
            ax.plot(x0, max(y0, 0.05), mk[p], color=COL[p], ms=5.2, mec="white", mew=0.5, zorder=4)
        ax.set_yscale("log")
        ax.set_ylim(0.035, 150)
        ax.set_yticks([0.05, 0.1, 1, 10, 100], ["0", "0.1", "1", "10", "100"])
        ax.set_xlim(-4, 104)
        ax.set_xlabel("False-block rate on injected runs (%)")
        ax.set_ylabel("Violating deployments per run")
        ax.fill_between([-4, 10], 0.035, 1.5, color="#E8F4EC", lw=0, zorder=0)
        ax.text(11, 1.1, "safe and\navailable", fontsize=5.8, color="#2E7D4F", va="top")
        panel(ax, "a" if ds == "acs" else "b", DS[ds])
    h = [Line2D([], [], ls="", marker=mk[p], color=COL[p], label=lab[p], ms=4.5) for p in order]
    h.append(Line2D([], [], ls="", marker="o", mfc="white", mec="#555555", label="same, at $n_{eval}$ = 50", ms=4))
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.3), fontsize=6.3, columnspacing=0.9,
               handletextpad=0.3)
    fig.savefig(out / "fig_tradeoff.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ Fig. human load and decidability

def fig_decidability(d: Data, out: Path) -> dict:
    r = d.csv("analysis/rq3.csv")
    r = r[(r.delay == 0) & (r["mode"] == "fail_safe")]
    fig, axs = plt.subplots(1, 3, figsize=(W, 1.95), gridspec_kw={"wspace": 0.42, "width_ratios": [1.05, 1, 1]})
    ax = axs[0]
    xs = {50: 0, 200: 1, 0: 2}
    for ds in ["acs", "isic"]:
        g = r[r.dataset == ds].groupby("n_eval").esc100.agg(["mean", "min", "max"])
        x = [xs[n] for n in [50, 200, 0]]
        ax.fill_between(x, [g.loc[n, "min"] for n in [50, 200, 0]], [g.loc[n, "max"] for n in [50, 200, 0]],
                        color=DS_COL[ds], alpha=0.18, lw=0)
        ax.plot(x, [g.loc[n, "mean"] for n in [50, 200, 0]], "o-", color=DS_COL[ds], ms=3.5, lw=1.2,
                label=DS[ds])
    ax.axhline(10, color="#C0392B", ls=(0, (3, 2)), lw=0.9)
    ax.text(0.02, 12, "usability bound (10)", fontsize=5.8, color="#C0392B")
    ax.set_xticks([0, 1, 2], ["50", "200", "all"])
    ax.set_xlabel("Evaluation rows per client ($n_{eval}$)")
    ax.set_ylabel("Undecided domain-rounds per 100")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.16), fontsize=6.2, handlelength=1.6, borderaxespad=0.3)
    panel(ax, "a", "Undecided candidates")
    facts = {}
    for ax, ds, letter in [(axs[1], "isic", "b"), (axs[2], "acs", "c")]:
        c = d.csv(f"decidability/{ds}/checks.csv")
        summ = json.loads((d.o / f"decidability/{ds}/summary.json").read_text())
        facts[ds] = summ
        for n, mkr, col in [("all", "o", "#1B1B1B"), ("200", "s", "#0072B2"), ("50", "^", "#E69F00")]:
            s = c[c.n_eval.astype(str) == n]
            ax.plot(s.p_inconclusive, s.obs_inconclusive, mkr, ms=3.2, mfc="none", mec=col, mew=0.8,
                    label=f"$n_{{eval}}$ = {n}", alpha=0.9)
        ax.plot([0, 1], [0, 1], color="#999999", lw=0.7, ls=(0, (2, 2)), label="equality")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Predicted P(inconclusive)")
        ax.set_ylabel("Observed P(inconclusive)")
        panel(ax, letter, DS[ds])
    # one legend shared by (b) and (c), centred under the two panels
    h, lab = axs[1].get_legend_handles_labels()
    x0, x1 = axs[1].get_position().x0, axs[2].get_position().x1
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=((x0 + x1) / 2, -0.06), ncol=4, fontsize=6.2,
               handletextpad=0.3, columnspacing=1.2)
    fig.savefig(out / "fig_decidability.pdf")
    plt.close(fig)
    return facts


# ------------------------------------------------------------------ Fig. imperfect reviewer (PH2)

def fig_reviewer(d: Data, out: Path) -> None:
    r = d.csv("posthoc/approver.csv")
    r = r[r.arm == "injected"]
    p3 = d.csv("posthoc/rules.csv")
    p3 = p3[(p3.n_eval == "all") & (p3.setting == "P3")].set_index("dataset")
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.9), gridspec_kw={"wspace": 0.3})
    ls = {0: "-", 200: (0, (4, 1.5)), 50: (0, (1, 1.2))}
    for ds in ["acs", "isic"]:
        for n in [0, 200, 50]:
            s = r[(r.dataset == ds) & (r.n_eval == n)].sort_values("q")
            kw = dict(color=DS_COL[ds], ls=ls[n], lw=1.2, marker="o", ms=2.6)
            axs[0].plot(s.q, s.exposure, **kw)
            axs[1].plot(s.q, 100 * s.fb_rate, **kw)
        axs[0].axhline(p3.loc[ds, "exposure_injected"], color=DS_COL[ds], lw=0.6, alpha=0.6)
        axs[0].text(0.995, p3.loc[ds, "exposure_injected"] + 0.8, f"Pooled, {DS[ds]}", fontsize=5.6,
                    color=DS_COL[ds], ha="left")
    for ax in axs:
        ax.set_xlim(1.01, 0.49)
        ax.set_xlabel("Approver accuracy $q$")
    axs[0].set_ylabel("Violating deployments per run")
    axs[1].set_ylabel("False-block rate (%)")
    panel(axs[0], "a", "Exposure")
    panel(axs[1], "b", "Availability cost")
    h = [Line2D([], [], color=DS_COL[ds], lw=1.2, label=DS[ds]) for ds in ["acs", "isic"]]
    h += [Line2D([], [], color="#555555", ls=ls[n], lw=1.2, label=f"$n_{{eval}}$ = {'all' if n == 0 else n}")
          for n in [0, 200, 50]]
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.2), fontsize=6.4)
    fig.savefig(out / "fig_reviewer.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ Fig. overhead (RQ4)

def fig_overhead(d: Data, out: Path) -> dict:
    a = json.loads((d.o / "overhead/mlp/summary.json").read_text())["datasets"]
    b = json.loads((d.o / "overhead/full/summary.json").read_text())["datasets"]
    rows = [("Census, MLP", a["acs"]), ("Dermatology, frozen backbone + head", a["isic"]),
            ("Dermatology, full EfficientNet-B0", b["isic-full"])]
    comp = [("train_ms_largest_client", "client training", "#C9D6E3"), ("model_upload_ms", "model upload", "#A9BCD0"),
            ("edge_compute_ms_slowest_edge", "edge monitoring", "#E69F00"), ("cloud_r5_ms", "cloud robustness", "#CC79A7"),
            ("cloud_r6_ms", "cloud explanation stability", "#D55E00"), ("cloud_rest", "cloud rate checks + OPA", "#009E73")]
    fig, ax = plt.subplots(figsize=(W, 1.45))
    facts = {}
    for i, (lab, s) in enumerate(rows):
        v = {k: s[k]["median"] for k, *_ in comp if k != "cloud_rest"}
        v["cloud_rest"] = max(s["cloud_compute_ms"]["median"] - v["cloud_r5_ms"] - v["cloud_r6_ms"], 0)
        tot = sum(v.values())
        left = 0
        y = 2 - i
        for k, name, c in comp:
            w = v[k] / tot
            ax.barh(y, w, left=left, color=c, height=0.6, ec="white", lw=0.4)
            if w > 0.07:
                ax.text(left + w / 2, y, f"{v[k] / 1000:,.0f} s" if v[k] >= 1000 else f"{v[k]:.0f} ms", ha="center",
                        va="center", fontsize=5.6, color="black" if c.startswith("#C") or c.startswith("#A") else "white")
            left += w
        share = s["monitoring_share_of_round"]["median"]
        ax.text(1.01, y, (f"round {tot / 1000:,.0f} s" if tot >= 10000 else f"round {tot:.0f} ms") + f"\nmonitoring {100 * share:.1f}%", va="center", fontsize=5.9)
        facts[lab] = {"total_ms": tot, "share": share}
    ax.set_yticks([2, 1, 0], [r[0] for r in rows])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100%"])
    ax.set_xlabel("Share of a round's critical path (medians over rounds and repetitions)")
    ax.grid(axis="y", visible=False)
    fig.legend(handles=[Patch(fc=c, label=n) for _, n, c in comp], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.47), fontsize=5.9, handlelength=1.0, columnspacing=0.8)
    fig.savefig(out / "fig_overhead.pdf")
    plt.close(fig)
    return facts


# ------------------------------------------------------------------ tables

def rv_tests(d: Data) -> pd.DataFrame:
    return pd.read_csv(d.an / "tests_per_config.csv")


def rv_means(d: Data, tiered: str, comparator: str, metric: str, side: str) -> pd.DataFrame:
    """Per (dataset, injection) mean over learners of a ('a' = Tiered variant) or b (= comparator)."""
    t = rv_tests(d)
    t = t[(t.family == "rq1") & (t.tiered == tiered) & (t.comparator == comparator) & (t.metric == metric)]
    return t.groupby(["dataset", "injection"])[f"{side}_mean"].mean()


def ptex(p: float) -> str:
    if p < 1e-4:
        return r"$<10^{-4}$"
    if abs(p - 0.0625) < 1e-9:  # smallest exact value with five clusters, shown in full
        return "0.0625"
    return f"{p:.4f}" if p < 0.001 else f"{p:.3f}"


def fmt(v: float, nd: int = 1) -> str:
    return "0" if abs(v) < 0.05 and nd == 1 else f"{v:.{nd}f}"


def table_exposure(d: Data, out: Path) -> None:
    t = d.csv("analysis/table_a.csv")
    p2s = rv_means(d, "failsafe", "P2s", "exposure", "b")
    p2s_fb = rv_means(d, "failsafe", "P2s", "false_block_rate", "b")
    hold_fb = rv_means(d, "failsafe", "P1", "false_block_rate", "a")
    ex_a = rv_means(d, "failsafe", "P1", "exposure", "a")
    cols = ["P1", "P2", "P2s", "P2c", "P3", "P4", "P4R"]
    lines = []
    for ds in ["acs", "isic"]:
        s = t[t.dataset == ds].groupby("injection")[["violating_candidates"] + [c for c in cols if c != "P2s"]].mean()
        s["P2s"] = [p2s.get((ds, i), np.nan) if i != "V0" else p2s_fb[(ds, "V0")] for i in s.index]
        lines.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{DS_LONG[ds]}}}}}\\\\")
        for inj in INJ:
            assert abs(ex_a[(ds, inj)] - s.loc[inj, "P4"]) < 1e-9, "fail-safe and oracle exposure differ"
            vals = [s.loc[inj, c] for c in cols]
            best = min(vals[i] for i, c in enumerate(cols) if c in ("P1", "P2", "P2s", "P3"))
            cells = []
            for c, v in zip(cols, vals):
                x = fmt(v)
                if c == "P4" and v < best:
                    x = f"\\textbf{{{x}}}"
                cells.append(x)
            lines.append(f"\\quad {TEX_INJ[inj]} & {s.loc[inj, 'violating_candidates']:.1f} & " + " & ".join(cells) + r"\\")
        fb = [f"{100 * s.loc['V0', c]:.1f}" for c in cols]
        fb[cols.index("P4")] = f"{100 * hold_fb[(ds, 'V0')]:.1f} ({100 * s.loc['V0', 'P4']:.1f})"
        lines.append(r"\quad Control, false-block rate (\%) & -- & " + " & ".join(fb) + r"\\")
        if ds == "acs":
            lines.append(r"\midrule")
    (out / "table_exposure.tex").write_text("\n".join(lines) + "\n")

def table_ablation(d: Data, out: Path) -> dict:
    summ = json.loads((d.an / "summary.json").read_text())["tests"]
    rows = [("A1", "Point estimates", "intervals replaced by point estimates", "all, $n_{eval}{=}50$"),
            ("A2a", "Undecided deploy", "undecided domains deploy, no approver", "all, $n_{eval}{=}50$"),
            ("A2b", "Undecided hold", "undecided domains hold, no approver", "all, $n_{eval}{=}50$"),
            ("A3", "Unadjusted local threshold", "restricted regime checks $\\tau$ instead of $\\tau-\\beta$",
             "Simpson shift"),
            ("A4", "No integrity alarm", "report-integrity alarms switched off", "Lying clients")]
    lines, facts = [], {}
    for a, name, what, inj in rows:
        cells = []
        for metric, scale in [("exposure", 1), ("false_block_rate", 100)]:
            r = summ[f"ablation/{a}/{metric}"]["both"]
            zero = r["configs"] - r["configs_mean_higher"] - r["configs_mean_lower"]
            ptxt = "--" if r["mean_diff"] == 0 else ptex(r["p_seed_cluster"])
            cells.append(f"{r['mean_diff'] * scale:+.2f} & {r['configs_mean_higher']}/{r['configs_mean_lower']}/{zero} & {ptxt}")
            facts[f"{a}/{metric}"] = r
        lines.append(f"{name} & {what} & {inj} & " + " & ".join(cells) + r"\\")
    (out / "table_ablation.tex").write_text("\n".join(lines) + "\n")
    return facts

def table_localisation(d: Data, out: Path) -> None:
    l = d.csv("posthoc/localisation.csv")
    g = l.groupby(["dataset", "injection", "placement"])[["target_violating_rounds", "caught", "localised",
                                                           "requirement_correct", "offtarget_clean",
                                                           "offtarget_alarms", "offtarget_fails"]].sum()
    sp = pd.read_csv(d.ph / "localisation_split.csv").set_index(["dataset", "injection", "placement"])
    lines = []
    for ds in ["acs", "isic"]:
        for inj, req in [("V1", "SP, EO"), ("V5", "privacy"), ("V6", "accuracy")]:
            cells = []
            for p in ["P2", "P4"]:
                r, x = g.loc[(ds, inj, p)], sp.loc[(ds, inj, p)]
                assert int(x.caught) == int(r.caught)
                cells += [f"{int(r.caught)}/{int(r.target_violating_rounds)}"]
                if p == "P2":
                    cells += [f"{100 * r.localised / r.caught:.0f}"]
                else:
                    assert int(x.target_failing + x.target_inconclusive_only) == int(r.localised)
                    cells += [f"{100 * x.target_failing / r.caught:.0f}", f"{100 * x.target_inconclusive_only / r.caught:.0f}"]
                cells += [f"{100 * r.requirement_correct / r.caught:.0f}", f"{100 * r.offtarget_fails / r.offtarget_clean:.1f}"]
                if p == "P4":
                    cells += [f"{100 * (r.offtarget_alarms - r.offtarget_fails) / r.offtarget_clean:.1f}"]
            head = DS[ds] if inj == "V1" else ""
            lines.append(f"{head} & {INJ_LAB[inj]} ({req}) & " + " & ".join(cells) + r"\\")
        if ds == "acs":
            lines.append(r"\midrule")
    (out / "table_localisation.tex").write_text("\n".join(lines) + "\n")

def table_overhead(d: Data, out: Path) -> None:
    a = json.loads((d.o / "overhead/mlp/summary.json").read_text())["datasets"]
    b = json.loads((d.o / "overhead/full/summary.json").read_text())["datasets"]
    S = [a["acs"], a["isic"], b["isic-full"]]

    def ms(k, s):
        v = s[k]["median"]
        return f"{v / 1000:,.1f} s" if v >= 1000 else (f"{v:.1f} ms" if v < 100 else f"{v:.0f} ms")

    rows = [("Client training, largest client (1 CPU)", lambda s: ms("train_ms_largest_client", s)),
            ("Model upload (20 Mbit/s, 25 ms)", lambda s: ms("model_upload_ms", s)),
            ("Model size", lambda s: f"{s['model_bytes']['median'] / 1e6:.2f} MB" if s['model_bytes']['median'] > 1e6
             else f"{s['model_bytes']['median'] / 1e3:.0f} KB"),
            ("Edge monitoring, median edge", lambda s: ms("edge_compute_ms_median", s)),
            ("\\quad of which OPA policy evaluation", lambda s: ms("edge_opa_ms", s)),
            ("Edge monitoring, slowest edge", lambda s: ms("edge_compute_ms_slowest_edge", s)),
            ("Monitoring message per edge", lambda s: f"{s['message_bytes_per_edge_median']['median'] / 1e3:.1f} KB"),
            ("Cloud monitoring", lambda s: ms("cloud_compute_ms", s)),
            ("\\quad robustness", lambda s: ms("cloud_r5_ms", s)),
            ("\\quad explanation stability", lambda s: ms("cloud_r6_ms", s)),
            ("\\quad OPA for all domains", lambda s: ms("cloud_opa_ms", s)),
            ("Peak memory, edge / cloud", lambda s: f"{s['peak_rss_mb_edge_max']['median'] / 1e3:.2f} / "
                                                    f"{s['peak_rss_mb_cloud']['median'] / 1e3:.2f} GB"),
            ("Edge share of an edge's round", lambda s: f"{100 * s['edge_share_of_edge_round']['median']:.1f}\\%"),
            ("\\textbf{Monitoring share of round}", lambda s: f"\\textbf{{{100 * s['monitoring_share_of_round']['median']:.1f}\\%}}")]
    def derived(s):
        med = {k: s[k]["median"] for k in ("train_ms_largest_client", "model_upload_ms", "edge_compute_ms_slowest_edge",
                                           "cloud_compute_ms", "cloud_r5_ms", "cloud_r6_ms")}
        msg = 1e3 * 8 * s["message_bytes_per_edge_median"]["median"] / 20e6
        mon = med["edge_compute_ms_slowest_edge"] + msg + med["cloud_compute_ms"]
        rnd = med["train_ms_largest_client"] + med["model_upload_ms"] + mon
        no6 = (mon - med["cloud_r6_ms"]) / (rnd - med["cloud_r6_ms"])
        no56 = (mon - med["cloud_r6_ms"] - med["cloud_r5_ms"]) / (rnd - med["cloud_r6_ms"] - med["cloud_r5_ms"])
        breakeven = mon / 0.05 - mon - med["model_upload_ms"]
        return mon, no6, no56, breakeven

    def t(v):
        return f"{v / 1000:,.1f} s" if v >= 1000 else f"{v:.0f} ms"

    rows += [("\\midrule Monitoring on the critical path$^\\dagger$", lambda s: t(derived(s)[0])),
             ("Share without explanation stability$^\\dagger$", lambda s: f"{100 * derived(s)[1]:.1f}\\%"),
             ("Share without robustness and explanation$^\\dagger$", lambda s: f"{100 * derived(s)[2]:.1f}\\%"),
             ("Training per round for a 5\\% share$^\\dagger$", lambda s: t(derived(s)[3]))]
    lines = [f"{lab} & " + " & ".join(f(s) for s in S) + r"\\" for lab, f in rows]
    (out / "table_overhead.tex").write_text("\n".join(lines) + "\n")


def table_rq1(d: Data, out: Path) -> dict:
    t = rv_tests(d)
    summ = json.loads((d.an / "summary.json").read_text())["tests"]
    facts = {"prespecified": summ["prespecified_rq1_family"]}
    lines = []
    for cmp_ in ["P1", "P2", "P2s", "P3"]:
        ex = t[(t.family == "rq1") & (t.tiered == "failsafe") & (t.comparator == cmp_) & (t.metric == "exposure")]
        orc = t[(t.family == "rq1") & (t.tiered == "oracle") & (t.comparator == cmp_) & (t.metric == "exposure")]
        assert np.allclose(ex.mean_diff.to_numpy(), orc.mean_diff.to_numpy()), "fail-safe and oracle exposure differ"
        fb = t[(t.family == "rq1") & (t.tiered == "failsafe") & (t.comparator == cmp_) & (t.metric == "false_block_rate")
               & (t.injection != "V0")]
        j = ex.merge(fb, on=["dataset", "learner", "injection"], suffixes=("_e", "_f"))
        for k, ds in enumerate(["acs", "isic"]):
            pr = summ[f"rq1/failsafe/{cmp_}/exposure"][ds]
            jj = j[j.dataset == ds]
            dom = int(((jj.mean_diff_e <= 0) & (jj.mean_diff_f <= 0) & ((jj.mean_diff_e < 0) | (jj.mean_diff_f < 0))).sum())
            facts[f"{cmp_}/{ds}"] = pr | {"dominates": dom}
            head = f"\\Tiered vs {PL[cmp_]}" if k == 0 else ""
            lines.append(f"{head} & {DS[ds]} & {pr['configs_mean_lower']}/{pr['configs_mean_higher']} & "
                         f"{pr['configs_all_runs_lower']} & {pr['mean_diff']:+.2f} & {pr['seed_clusters_a_lower']}/"
                         f"{pr['seed_clusters_a_higher']} & {ptex(pr['p_seed_cluster'])} & {ptex(pr['p_cluster'])} & {dom}/21" + r"\\")
        if cmp_ != "P3":
            lines.append(r"\addlinespace")
    (out / "table_rq1.tex").write_text("\n".join(lines) + "\n")
    return facts

def table_tradeoff(d: Data, out: Path) -> dict:
    """Exposure and false-block rates of the four simple placements and Tiered's fail-safe default, all rows."""
    m = d.csv("posthoc/rules.csv")
    lines, facts = [], {}
    for key_ in ["P1", "P2", "P2s", "P3", "P4-hold"]:
        cells = []
        for ds in ["acs", "isic"]:
            r = m[(m.dataset == ds) & (m.n_eval.astype(str) == "all") & (m.setting == key_)].iloc[0]
            facts[f"{ds}/{key_}"] = {"exposure": r.exposure_injected, "fb_inj": r.fb_injected, "fb_V0": r.fb_V0}
            cells += [f"{r.exposure_injected:.2f}", f"{100 * r.fb_injected:.1f}", f"{100 * r.fb_V0:.1f}"]
        lab = r"\Tiered (fail-safe default)" if key_ == "P4-hold" else PL[key_].replace("Uniform-scoped", r"\UniformS")
        lines.append(f"{lab} & " + " & ".join(cells) + r"\\")
    (out / "table_tradeoff.tex").write_text("\n".join(lines) + "\n")
    return facts


def table_point_vs_intervals(d: Data, out: Path) -> dict:
    """Point estimates against intervals (fail-safe default), with the report-integrity alarm on and off."""
    m = d.csv("posthoc/point_vs_intervals.csv")
    order = [("point, alarm", "Point estimates", "on"), ("intervals fail-safe, alarm", "Intervals", "on"),
             ("point, no alarm", "Point estimates", "off"), ("intervals fail-safe, no alarm", "Intervals", "off")]
    lines, facts = [], {}
    for n in ["all", "50"]:
        head = "All evaluation rows" if n == "all" else r"$\neval=50$"
        lines.append(r"\multicolumn{8}{l}{\textit{" + head + r"}}\\")
        for key_, lab, alarm in order:
            cells = []
            for ds in ["acs", "isic"]:
                r = m[(m.dataset == ds) & (m.n_eval.astype(str) == n) & (m.setting == key_)].iloc[0]
                facts[f"{ds}/{n}/{key_}"] = {"exposure": r.exposure_injected, "fb_inj": r.fb_injected, "fb_V0": r.fb_V0}
                cells += [f"{r.exposure_injected:.2f}", f"{100 * r.fb_injected:.1f}", f"{100 * r.fb_V0:.1f}"]
            lines.append(f"\\quad {lab} & {alarm} & " + " & ".join(cells) + r"\\")
        if n == "all":
            lines.append(r"\midrule")
    (out / "table_ph5.tex").write_text("\n".join(lines) + "\n")
    return facts


def table_ph1(d: Data, out: Path) -> dict:
    m = d.csv("posthoc/rules.csv")
    names = ["P1", "P2", "P2s", "P3", "P4-point", "P4-hold", "P4-deploy", "P4-reviewer"]
    lines, facts = [], {}
    for ds in ["acs", "isic"]:
        lines.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{DS_LONG[ds]}}}}}\\\\")
        s = m[m.dataset == ds].set_index(["n_eval", "setting"])
        for key_ in names:
            cells = []
            for n in ["all", "50"]:
                r = s.loc[(n, key_)]
                facts[f"{ds}/{n}/{key_}"] = {"exposure": r.exposure_injected, "fb_inj": r.fb_injected, "fb_V0": r.fb_V0}
                cells += [f"{r.exposure_injected:.2f}", f"{100 * r.fb_injected:.1f}", f"{100 * r.fb_V0:.1f}",
                          f"{r.esc100:.1f}" if key_ == "P4-reviewer" else "--"]
            lab = PL[key_].replace("Tiered", r"\Tiered").replace("Uniform-scoped", r"\UniformS")
            lines.append(f"\\quad {lab} & " + " & ".join(cells) + r"\\")
        if ds == "acs":
            lines.append(r"\midrule")
    (out / "table_ph1.tex").write_text("\n".join(lines) + "\n")
    return facts

def table_rq2(d: Data, out: Path) -> None:
    src = {"acs": "rq2/acs/rq2_summary.csv", "isic": "rq2/isic/rq2_summary.csv"}
    req = {"R1": "SP", "R2": "EO", "R3": "worst-group accuracy"}
    typ = {"LPGF": "local pass, global fail", "GPLF": "global pass, local fail"}
    lines = []
    for ds in ["acs", "isic"]:
        s = d.csv(src[ds])
        s = s[s.domain_rounds > 0]
        first = True
        for _, r in s.iterrows():
            if first:
                lines.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{DS_LONG[ds]}}}}}\\\\")
            first = False
            lines.append(f"\\quad {TEX_INJ[r.injection]} & {req[r.req]} & {typ[r.type]} & {int(r.domain_rounds):,} & "
                         f"{int(r.P1_caught):,} & {int(r.P2_caught):,} & {int(r.P3_caught):,} & {int(r.P4_caught):,} "
                         f"& {int(r.P4_held):,} / {int(r.P4_escalated):,}" + r"\\")
        if ds == "acs":
            lines.append(r"\midrule")
    (out / "table_rq2.tex").write_text("\n".join(lines) + "\n")


def table_rq3(d: Data, out: Path) -> None:
    r = d.csv("analysis/rq3.csv")
    q = d.csv("analysis/rq5.csv").set_index(["dataset", "learner"])
    ta = d.csv("analysis/table_a.csv").set_index(["dataset", "learner", "injection"])
    t = rv_tests(d)
    hold = t[(t.family == "rq1") & (t.tiered == "failsafe") & (t.comparator == "P1") & (t.metric == "false_block_rate")
             & (t.injection == "V0")].set_index(["dataset", "learner"]).a_mean
    lines = []
    for ds in ["acs", "isic"]:
        for le in ["fedavg", "fairfed", "eafclc"]:
            s = r[(r.dataset == ds) & (r.learner == le)]
            fs = s[(s.delay == 0) & (s["mode"] == "fail_safe")].set_index("n_eval")
            fo = s[(s.delay == 1) & (s["mode"] == "fail_operational") & (s.n_eval == 0)].iloc[0]
            head = DS[ds] if le == "fedavg" else ""
            lines.append(f"{head} & {LEARN[le]} & {100 * hold[(ds, le)]:.1f} & {100 * ta.loc[(ds, le, 'V0'), 'P4']:.1f} & "
                         f"{fs.loc[0, 'esc100']:.1f} & {fs.loc[200, 'esc100']:.1f} & "
                         f"{fs.loc[50, 'esc100']:.1f} & {fo.exposure:.1f} & {q.loc[(ds, le), 'p4_mean']:.2f} & "
                         f"{q.loc[(ds, le), 'p3_mean']:.2f} & {q.loc[(ds, le), 'mean_diff_P4_minus_P3']:.2f}" + r"\\")
        if ds == "acs":
            lines.append(r"\midrule")
    (out / "table_rq3.tex").write_text("\n".join(lines) + "\n")

def table_reviewer(d: Data, out: Path) -> None:
    r = d.csv("posthoc/approver.csv")
    r = r[(r.arm == "injected")]
    lines = []
    for qv in [1.0, 0.95, 0.9, 0.8, 0.7, 0.5]:
        cells = []
        for ds in ["acs", "isic"]:
            for n in [0, 200]:
                x = r[(r.dataset == ds) & (r.n_eval == n) & (np.isclose(r.q, qv))].iloc[0]
                cells += [f"{x.exposure:.2f}", f"{100 * x.fb_rate:.1f}"]
        lines.append(f"{qv:g} & " + " & ".join(cells) + r"\\")
    (out / "table_reviewer.tex").write_text("\n".join(lines) + "\n")


def table_decidability(d: Data, out: Path) -> None:
    lines = []
    for ds in ["acs", "isic"]:
        s = json.loads((d.o / f"decidability/{ds}/summary.json").read_text())
        for key, lab in [("checks_inconclusive", "check inconclusive"), ("domains_escalate", "domain undecided")]:
            x = s[key]
            lines.append(f"{DS[ds]} & {lab} & {x['n']} & {x['mae']:.3f} & {x['binary_agreement']:.2f} & "
                         f"{x['spearman']:.2f}" + r"\\")
    (out / "table_decidability.tex").write_text("\n".join(lines) + "\n")


def table_details(d: Data, out: Path) -> None:
    beta_src = {"acs": "rq2/acs/summary.json", "isic": "rq2/isic/summary.json"}
    lines = []
    for ds in ["acs", "isic"]:
        b = json.loads((d.o / beta_src[ds]).read_text())["beta"]
        lines.append(f"{DS[ds]} & {b['beta_sp_pre']['V0']:.3f} & {b['beta_eo_pre']['V0']:.3f} & "
                     f"{b['beta_sp_post']['V1']:.2f} & {b['beta_eo_post']['V1']:.2f} & {b['beta_sp_post']['V2']:.2f} & "
                     f"{b['beta_eo_post']['V2']:.2f}" + r"\\")
    (out / "table_beta.tex").write_text("\n".join(lines) + "\n")
    miss_src = {"acs": "rq2/acs/p4_misses.json", "isic": "rq2/isic/p4_misses.json"}
    lines = []
    for ds in ["acs", "isic"]:
        m = pd.DataFrame(json.loads((d.o / miss_src[ds]).read_text()))
        for (inj, req), g in m.groupby(["injection", "req"]):
            truth = g.gt_domain if req in ("R1", "R2", "R3") else g.gt_global
            chk = [next(c for c in cs if c["id"].endswith("/" + dom) or (req in ("R5", "R6") and c["id"].endswith("global")))
                   for cs, dom in zip(g.checks, g.domain)]
            lo = np.median([c["lower"] for c in chk])
            hi = np.median([c["upper"] for c in chk])
            tau = chk[0]["threshold"]
            doms = ", ".join(DOM[ds][int(x[1:])] for x in sorted(set(g.domain)))
            lines.append(f"{DS[ds]} & {INJ_LAB[inj]} & {REQ[req]} & {len(g)} & {doms} & "
                         f"{tau:.2f} & {np.median(truth):.3f} & [{lo:.3f}, {hi:.3f}]" + r"\\")
    (out / "table_misses.tex").write_text("\n".join(lines) + "\n")
    r = d.csv("posthoc/integrity.csv").groupby(["dataset", "injection"])[
        ["liars", "alarms", "alarms_on_liars", "liars_alarmed"]].sum()
    lines = []
    for (ds, inj), x in r.iterrows():
        prec = f"{x.alarms_on_liars / x.alarms:.2f}" if x.alarms else "--"
        rec = f"{x.liars_alarmed / x.liars:.2f}" if x.liars else "--"
        lines.append(f"{DS[ds]} & {INJ_LAB[inj]} & {int(x.liars)} & {int(x.alarms)} & {int(x.alarms_on_liars)} & "
                     f"{int(x.liars_alarmed)} & {prec} & {rec}" + r"\\")
    (out / "table_integrity.tex").write_text("\n".join(lines) + "\n")
    f = d.csv("analysis/detection_delay.csv")
    f = f[f.placement.isin(["P1", "P2", "P3", "P4"])]
    lines = []
    for ds in ["acs", "isic"]:
        for inj in INJ:
            g = f[(f.dataset == ds) & (f.inj == inj)]
            cells = []
            for p in ["P1", "P2", "P3", "P4"]:
                v = g[g.placement == p].detection_delay.dropna()
                cells.append(f"{np.median(v):.0f} ({len(v)})" if len(v) else "-- (0)")
            head = DS[ds] if inj == "V1" else ""
            lines.append(f"{head} & {TEX_INJ[inj]} & " + " & ".join(cells) + r"\\")
        if ds == "acs":
            lines.append(r"\midrule")
    (out / "table_delay.tex").write_text("\n".join(lines) + "\n")


def table_retained(d: Data, out: Path) -> dict:
    r = pd.read_csv(d.ph / "retained.csv")
    r = r[r.injection != "V0"]
    order = ["Central", "Uniform", "Pooled", "Tiered, fail-safe default", "Tiered + oracle approver"]
    r = r.assign(none_share=r.never_deployed_domain_rounds / (r.never_deployed_domain_rounds + r.served_domain_rounds))
    g = r.groupby(["dataset", "setting"])[["candidate_violating_deployments", "retained_violating", "mean_model_age",
                                           "none_share"]].mean()
    lines, facts = [], {}
    for st in order:
        cells = []
        for ds in ["acs", "isic"]:
            x = g.loc[(ds, st)]
            facts[f"{ds}/{st}"] = x.to_dict()
            cells += [f"{x.candidate_violating_deployments:.2f}", f"{x.retained_violating:.2f}", f"{x.mean_model_age:.1f}",
                      f"{100 * x.none_share:.0f}"]
        lab = st.replace("Tiered", r"\Tiered").replace("Central", r"\Central").replace("Uniform", r"\Uniform").replace(
            "Pooled", r"\Pooled")
        lines.append(f"{lab} & " + " & ".join(cells) + r"\\")
    facts["runs"] = r.groupby("dataset").trajectory.nunique().to_dict()
    (out / "table_retained.tex").write_text("\n".join(lines) + "\n")
    return facts


def table_coverage(d: Data, out: Path) -> dict:
    c = pd.read_csv(d.ph / "coverage.csv.gz")
    g = c.groupby(["dataset", "requirement"]).agg(n=("covered", "size"), cov=("covered", "mean"),
                                                   exp=("expected", "mean"), w=("width", "median")).reset_index()
    sim = pd.read_csv(d.ph / "simulated_coverage.csv")
    sim = sim[sim.scope != "global"].groupby(["dataset", "requirement"]).coverage.agg(["mean", "min"])
    lines, facts = [], {}
    for ds in ["acs", "isic"]:
        for i, req in enumerate(["SP gap", "EO gap", "Worst-group accuracy"]):
            x = g[(g.dataset == ds) & (g.requirement == req)].iloc[0]
            facts[f"{ds}/{req}"] = x.drop(["dataset", "requirement"]).to_dict()
            head = DS[ds] if i == 0 else ""
            sm = sim.loc[(ds, req)]
            facts[f"{ds}/{req}/simulated"] = sm.to_dict()
            lines.append(f"{head} & {req} & {int(x.n):,} & {100 * x['cov']:.1f} & {100 * x.exp:.1f} & {x.w:.3f} & "
                         f"{100 * sm['mean']:.1f} & {100 * sm['min']:.1f}" + r"\\")
        if ds == "acs":
            lines.append(r"\midrule")
    (out / "table_coverage.tex").write_text("\n".join(lines) + "\n")
    return facts


def table_samplesize(d: Data, out: Path) -> dict:
    s = pd.read_csv(d.ph / "sample_size.csv")
    lines, facts = [], {}
    for x in s[s.dataset == "isic"].itertuples():
        name = DOM["isic"][x.domain]
        pos = json.loads(x.positives_by_group)
        if np.isnan(x.factor):
            need = r"\multicolumn{3}{l}{not reached at 256}"
        else:
            need = f"{x.factor:g} & {int(x.records_needed):,} & {100 * x.p_hold_at_factor:.0f}"
        lines.append(f"{name} & {x.eval_records:,} & {pos[0]}/{pos[1]} & {100 * x.k1:.0f} & {need}" + r"\\")
        facts[name] = {"k1": x.k1, "factor": x.factor, "records": x.records_needed, "hold": x.p_hold_at_factor,
                       "pilot_sp": x.pilot_sp, "pilot_eo": x.pilot_eo, "pilot_wga": x.pilot_wga}
    facts["acs_max_k1"] = float(s[s.dataset == "acs"].k1.max())
    (out / "table_samplesize.tex").write_text("\n".join(lines) + "\n")
    return facts

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    style()
    d = Data(Path(a.root).resolve())
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    f, t = out / "figures", out / "tables"
    jobs = {"motivation": lambda: fig_motivation(d, f), "timeline": lambda: fig_timeline(d, f),
            "exposure": lambda: fig_exposure(d, f), "soundness": lambda: fig_soundness(d, f),
            "tradeoff": lambda: fig_tradeoff(d, f), "decidability": lambda: fig_decidability(d, f),
            "reviewer": lambda: fig_reviewer(d, f), "overhead": lambda: fig_overhead(d, f),
            "t_exposure": lambda: table_exposure(d, t),
            "t_ablation": lambda: table_ablation(d, t), "t_localisation": lambda: table_localisation(d, t),
            "t_overhead": lambda: table_overhead(d, t), "t_rq1": lambda: table_rq1(d, t),
            "t_ph1": lambda: table_ph1(d, t), "t_tradeoff": lambda: table_tradeoff(d, t), "t_ph5": lambda: table_point_vs_intervals(d, t), "t_rq2": lambda: table_rq2(d, t), "t_rq3": lambda: table_rq3(d, t),
            "t_reviewer": lambda: table_reviewer(d, t), "t_decidability": lambda: table_decidability(d, t),
            "t_details": lambda: table_details(d, t), "t_retained": lambda: table_retained(d, t),
            "t_coverage": lambda: table_coverage(d, t), "t_samplesize": lambda: table_samplesize(d, t)}
    facts = {}
    for k, fn in jobs.items():
        if a.only and k not in a.only:
            continue
        r = fn()
        if r:
            facts[k] = r
        print("done", k, flush=True)
    prev = out / "assets_facts.json"
    old = json.loads(prev.read_text()) if prev.exists() else {}
    prev.write_text(json.dumps(old | facts, indent=1, default=float))


if __name__ == "__main__":
    main()
