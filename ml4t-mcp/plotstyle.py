"""Shared chart style for all skills in /opt/skills (matplotlib). Rules: STYLE.md.

No fixed color per quantity. Colors are assigned by rule:
  - series take palette slots in order of first use (ps.color("anything") -> next free slot,
    the same name keeps its color for the rest of the run = consistent within one report),
  - data is always colored; gray is reserved for axes, grid and text,
  - >= 2 series -> colored series names under the title (finish(keys=...)) or a legend on top of
    the panel (automatic in finish() for artists with label=...); all lines solid, equal width,
  - diverging / sequential scales for signed values / magnitudes.

Usage:
    import sys; sys.path.insert(0, "/opt/skills/_shared")
    import plotstyle as ps
    ps.apply()                                  # dark (default) or ps.apply("light")
    fig, ax = ps.figure()                       # portrait, iPhone-readable
    ax.plot(x, y, color=ps.color("IV"), label="IV")
    ax.plot(x, y2, color=ps.color("RV"), label="RV")   # legend is added by finish()
    ps.ylabel(ax, "Vol, % p.a.")                # normal y-label with unit (default)
    ps.finish(fig, ax, "IV stays above RV most days", subtitle="20-day mean",
              keys=[("IV", ps.color("IV")), ("RV", ps.color("RV"))],   # colored names = legend
              source="vrp-daily · 2022-06 – 2026-10", out="/path/chart.png")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mc  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# Validated categorical palette (CVD-checked), fixed order, never cycled beyond 8.
PALETTE = {
    "dark": ["#3987e5", "#d95926", "#199e70", "#f2c744", "#d55181", "#9085e9", "#008300", "#e66767"],
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#c2407a", "#6250d6", "#008300", "#e34948"],
}
DIVERGING = {"dark": ("#3987e5", "#383835", "#e66767"), "light": ("#2a78d6", "#e1e0d9", "#e34948")}
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
INK = {
    "dark": dict(surface="#141413", primary="#f0efec", secondary="#c3c2b7", muted="#898781",
                 grid="#2a2a28", axis="#45443f", ),
    "light": dict(surface="#fcfcfb", primary="#141413", secondary="#52514e", muted="#898781",
                  grid="#e8e7e1", axis="#c3c2b7", ),
}

LW = 1.8
PORTRAIT = (6.0, 7.5)   # 1200x1500 px at 200 dpi -> iPhone width
WIDE = (8.0, 5.0)
DPI = 200
_mode = "dark"
_assigned: dict[str, int] = {}

_FONT_DIR = Path(__file__).resolve().parent / "fonts"
for _f in _FONT_DIR.glob("*.ttf"):
    font_manager.fontManager.addfont(str(_f))
FONT = "Inter" if any(_FONT_DIR.glob("Inter-*.ttf")) else "DejaVu Sans"


def apply(mode: str = "dark") -> None:
    """Global style. Dark is the default, light on request."""
    global _mode
    _mode = mode
    reset_colors()
    k = INK[mode]
    plt.rcParams.update({
        "figure.facecolor": k["surface"], "axes.facecolor": k["surface"], "savefig.facecolor": k["surface"],
        "font.family": "sans-serif", "font.sans-serif": [FONT, "DejaVu Sans"],
        "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 12,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
        "text.color": k["primary"], "axes.labelcolor": k["muted"],
        "xtick.color": k["muted"], "ytick.color": k["muted"],
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
        "axes.edgecolor": k["axis"], "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": k["grid"], "grid.linewidth": 0.7,
        "axes.axisbelow": True, "axes.titlelocation": "left", "axes.titleweight": "semibold",
        "axes.titlecolor": k["secondary"], "axes.titlepad": 10,
        "lines.linewidth": LW, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "axes.prop_cycle": plt.cycler(color=PALETTE[mode]),
        "axes.formatter.useoffset": False,
    })


def reset_colors() -> None:
    """Forget name->color assignments (call between unrelated reports)."""
    _assigned.clear()


def color(key: str | int = 0, slot: int | None = None) -> str:
    """Series color. int -> palette slot. str -> next free slot on first use, then stable for the run.
    slot=... pins a name to a chosen slot (e.g. to give overlapping lines different lightness)."""
    if isinstance(key, int):
        return PALETTE[_mode][key % len(PALETTE[_mode])]
    if slot is not None and key not in _assigned:
        _assigned[key] = slot
    if key not in _assigned:
        free = [i for i in range(len(PALETTE[_mode])) if i not in _assigned.values()]
        _assigned[key] = free[0] if free else len(PALETTE[_mode])
        if _assigned[key] >= len(PALETTE[_mode]):
            raise ValueError("more than 8 colored series - use small multiples or group into 'Other'")
    return PALETTE[_mode][_assigned[key]]


def lightness(c: str | int) -> float:
    """Relative luminance contrast of a color against the current background (WCAG ratio).
    Overlapping lines should differ clearly (pick the largest gap, rule of thumb >= 1.0)."""
    def lum(h):
        r = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in mc.to_rgb(h)]
        return 0.2126 * r[0] + 0.7152 * r[1] + 0.0722 * r[2]
    h = c if isinstance(c, str) and c.startswith("#") else color(c)
    a, b = lum(h), lum(INK[_mode]["surface"])
    return round((max(a, b) + 0.05) / (min(a, b) + 0.05), 2)


def diverging() -> tuple[str, str, str]:
    """(positive, midpoint, negative): blue / gray / red."""
    return DIVERGING[_mode]


def sequential(n: int = 7) -> list[str]:
    """Single-hue ramp, low -> high = least -> most contrast against the background."""
    ramp = SEQUENTIAL if _mode == "light" else SEQUENTIAL[::-1]
    if n >= len(ramp):
        return ramp
    step = (len(ramp) - 1) / max(n - 1, 1)
    return [ramp[round(i * step)] for i in range(n)]


def ink(name: str = "primary") -> str:
    return INK[_mode][name]


def figure(kind: str = "portrait", nrows: int = 1, sharex: bool = True):
    size = PORTRAIT if kind == "portrait" else WIDE
    return plt.subplots(nrows, 1, figsize=size, sharex=sharex)


def ylabel(ax, text: str, c: str | None = None) -> None:
    """Default: normal (rotated) y-axis label incl. unit. c = series color, used for a second y-axis
    so it is clear which line belongs to which axis (label and tick labels in that color)."""
    ax.set_ylabel(text, color=c or ink("secondary"), fontsize=12, labelpad=8)
    if c:
        ax.tick_params(axis="y", colors=c)


def twin(ax):
    """Second y-axis on the right (allowed). Label both axes with ylabel(..., c=series color)."""
    ax2 = ax.twinx()
    ax2.grid(False)
    for side in ("top", "left", "right"):
        ax2.spines[side].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    return ax2


def panel_title(ax, text: str) -> None:
    """Alternative to ylabel() when space is tight: small left-aligned panel heading incl. unit."""
    ax.set_title(text, fontsize=12.5)


def legend(ax) -> None:
    """Horizontal legend on top of the panel, below the panel title. Called by finish() for every
    axes with >= 2 labeled artists; call it yourself only to force a single-entry legend."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    lg = ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=min(len(labels), 4),
                   frameon=False, handlelength=1.6, handletextpad=0.5, columnspacing=1.4,
                   borderaxespad=0.3, markerscale=2.5, fontsize=12, labelcolor=ink("secondary"))
    for h in getattr(lg, "legend_handles", None) or getattr(lg, "legendHandles", []):  # mpl >= 3.7 / 3.6
        if hasattr(h, "set_alpha"):
            h.set_alpha(1)
    t = ax.get_title(loc="left")
    if t:
        ax.set_title(t, loc="left", fontsize=12.5, pad=32)


def label_end(ax, x, y, text: str, c: str | None = None) -> None:
    """Optional (legend is the default). Direct label at the last point of a line. Dot in the series color (default: color(text)),
    text in ink. Placed in finish() so that close labels are pushed apart."""
    xs, ys = list(x), list(y)
    ax.scatter([xs[-1]], [ys[-1]], s=34, color=c or color(text), zorder=6,
               edgecolors=ink("surface"), linewidths=1.5, clip_on=False)
    if not hasattr(ax, "_ps_labels"):
        ax._ps_labels = []
    ax._ps_labels.append((float(ax.convert_xunits(xs[-1])), float(ys[-1]), text))


def _place_labels(fig, ax, min_gap_pt: float = 16.0) -> None:
    labels = getattr(ax, "_ps_labels", [])
    if not labels:
        return
    fig.canvas.draw()
    to_px = ax.transData.transform
    k = fig.dpi / 72.0
    pts = sorted(((to_px((x, y))[1], x, y, t) for x, y, t in labels), key=lambda r: r[0])
    placed = []
    for ypx, x, y, t in pts:
        if placed and ypx - placed[-1][0] < min_gap_pt * k:
            ypx = placed[-1][0] + min_gap_pt * k
        placed.append((ypx, x, y, t))
    for ypx, x, y, t in placed:
        ax.annotate(t, (x, y), xytext=(9, (ypx - to_px((x, y))[1]) / k), textcoords="offset points",
                    va="center", ha="left", color=ink("secondary"), fontsize=12, fontweight="medium",
                    annotation_clip=False, zorder=7)


def _date_axis(ax) -> None:
    import matplotlib.dates as mdates
    if isinstance(ax.xaxis.get_major_locator(), mdates.AutoDateLocator):
        loc = mdates.AutoDateLocator(minticks=3, maxticks=6)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc, show_offset=False))


def _key_line(fig, y: float, keys) -> None:
    """Legend replacement under the title: series names written in their series color, side by side."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    x = 0.05
    for i, (name, c) in enumerate(keys):
        t = fig.text(x, y, name, fontsize=13, fontweight="semibold", color=c, va="top")
        w = t.get_window_extent(renderer=r).width / fig.bbox.width
        x += w + 0.035


def finish(fig, ax, title: str, subtitle: str = "", source: str = "", out: str | None = None,
           keys=None) -> None:
    """Title = the takeaway, subtitle = unit/definition, footer = source + period.
    keys = [(name, color), ...] -> colored names under the title replace the legend (preferred
    for 2-4 lines in all panels); without keys the legend goes on top of each panel."""
    axes = list(ax) if hasattr(ax, "__len__") else [ax]
    n_text = (1 if subtitle else 0) + subtitle.count("\n") + (1 if keys else 0)
    top = 0.87 - 0.035 * n_text
    fig.subplots_adjust(left=0.14, right=0.86, top=top, bottom=0.11, hspace=0.42)
    fig.text(0.05, 0.965, title, fontsize=17, fontweight="bold", color=ink("primary"), va="top")
    y = 0.925
    if keys:
        _key_line(fig, y, keys)
        y -= 0.04
    if subtitle:
        fig.text(0.05, y, subtitle, fontsize=12.5, color=ink("secondary"), va="top")
    if source:
        fig.text(0.05, 0.03, source, fontsize=10, color=ink("muted"), va="bottom")
    from matplotlib import patheffects as pe
    for a in axes:
        if a.collections:  # lines over points/areas -> thin halo in background color keeps them sharp
            for ln in a.get_lines():
                if not ln.get_label().startswith("_"):
                    ln.set_path_effects([pe.Stroke(linewidth=ln.get_linewidth() + 2.4,
                                                   foreground=ink("surface")), pe.Normal()])
        _date_axis(a)
        if not keys and len(a.get_legend_handles_labels()[1]) >= 2 and a.get_legend() is None:
            legend(a)
    fig.align_ylabels(axes)
    for a in axes:
        _place_labels(fig, a)
    if out:
        fig.savefig(out, dpi=DPI)
        plt.close(fig)
