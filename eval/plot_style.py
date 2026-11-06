"""Shared figure style for the report (static PDF + PNG for LaTeX).

Colour rules (validated with the dataviz skill's validate_palette.js on a white page):
- Categorical slots are assigned in a fixed order and follow the *entity* (a model keeps its colour
  in every figure).
- The 8-slot order passes all adjacent-pair CVD checks: worst adjacent ΔE 9.1 (protan), normal-vision 19.6.
- Only slots 1-3 pass all-pairs (worst ΔE 9.2 CVD, 24.0 normal). Figures where any two series can meet
  (crossing lines, scatter) therefore emphasise at most 3 models in colour; others are grey with marker shapes.
- Slots 3-5 are < 3:1 contrast on white, so every figure carries a legend and/or direct labels.
- Sequential = one hue (blue ramp). Diverging = blue <-> red with a grey midpoint.
- Recessive solid hairline grid, thin marks, text in ink colours (never the series colour).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CONTEXT_GREY = "#a8a69f"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")

COL_W = 3.35    # inches, one column of a two-column paper
FULL_W = 6.9    # inches, full text width


def apply():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "legend.frameon": False,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.axisbelow": True, "lines.linewidth": 1.4, "lines.markersize": 4.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.prop_cycle": matplotlib.cycler(color=SLOTS),
    })


def color(slot):
    """1-based categorical slot -> hex; None/0 -> context grey."""
    if not slot:
        return CONTEXT_GREY
    return SLOTS[(int(slot) - 1) % len(SLOTS)]


def marker(slot):
    return MARKERS[(int(slot) - 1) % len(MARKERS)] if slot else "o"


def save(fig, out_base, formats=("pdf", "png")):
    """Write <out_base>.pdf and .png; returns the list of paths."""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(out_base)) or ".", exist_ok=True)
    paths = []
    for ext in formats:
        p = f"{out_base}.{ext}"
        # deterministic files: no creation timestamp, so an unchanged figure is byte-identical in git
        fig.savefig(p, metadata={"CreationDate": None} if ext == "pdf" else None)
        paths.append(p)
    plt.close(fig)
    return paths
