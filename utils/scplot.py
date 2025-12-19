"""
scplot.py

Provides enhanced functionality for creating publication-quality
plots using Matplotlib.

Example
-------
.. code-block:: python

    from scplot import subplots

    fig, axs = subplots(nrows=2, ncols=2, aspect_ratio=3/4, width_pt='thesis')
    for ax in axs:
        ax.plot(x, y)
    fig.savefig("example_plot.png", dpi=300)
"""

import math
from collections.abc import Sequence
from typing import Any, Dict, Literal, Optional, Union

import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import rc_context, style
from matplotlib.figure import Figure

PUBLICATION_STYLE = {
    # axes properties
    "axes.labelsize": 10.0,
    "axes.titlesize": 11,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.linestyle": ":",
    # "grid.alpha" : 0.5,
    "axes.prop_cycle": (
        cycler(
            "color",
            [
                "#377eb8",
                "#ff7f00",
                "#4daf4a",
                "#f781bf",
                "#a65628",
                "#984ea3",
                "#999999",
                "#e41a1c",
                "#dede00",
            ],
        )
        + cycler(
            "ls",
            [
                "-",
                "--",
                ":",
                "-.",
                "-",
                "--",
                ":",
                "-.",
                "-",
            ],
        )
    ),
    "lines.marker": "None",
    # patch properties
    "patch.edgecolor": "None",
    "patch.facecolor": "#808080",
    # x-axis
    "xtick.direction": "out",
    "xtick.major.size": 3,
    "xtick.major.width": 0.5,
    "xtick.minor.size": 1.5,
    "xtick.minor.width": 0.5,
    "xtick.minor.visible": True,
    "xtick.top": True,
    "xtick.labelsize": 9.0,
    # y-axis
    "ytick.direction": "out",
    "ytick.major.size": 3,
    "ytick.major.width": 0.5,
    "ytick.minor.size": 1.5,
    "ytick.minor.width": 0.5,
    "ytick.minor.visible": True,
    "ytick.right": True,
    "ytick.labelsize": 9.0,
    # line properties
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "lines.markersize": 5,
    # scatter properties
    "scatter.marker": "o",
    # legend
    "legend.frameon": True,
    "legend.fontsize": 9,
    "legend.framealpha": 1.0,  # or < 1.0 if you want transparency
    "legend.loc": "best",  # or upper right / etc. if you prefer consistency
    # fonts
    "font.size": 10,
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "text.usetex": True,
    "text.latex.preamble": (
        r"\usepackage{amsmath} " r"\usepackage{amssymb} " r"\usepackage{lmodern} " r"\boldmath"
    ),
    # savefig
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0,
    "savefig.transparent": False,
    "savefig.dpi": 300,
    "figure.autolayout": False,  # default; rely on bbox='tight'
    # or, if you like:
    # figure.autolayout : True
}


def _update_style_with_figsize(fig_in, default_in, base_rc):
    """
    Dynamically adjust font sizes based on the figure width.

    :param fig_width_in: The width of the figure in inches.
    :param base_fontsize: The base font size for a standard figure width.
    """
    #  Scale based on actual figure width
    sc_fac = 0.5 * (fig_in[1] / default_in[1] + fig_in[0] / default_in[0])

    # Scale all text elements while preserving their relative proportions
    font_keys = [
        "font.size",
        "axes.titlesize",
        "axes.labelsize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
        "lines.linewidth",
    ]

    rc = base_rc.copy()
    for key in font_keys:
        value = rc.get(key, None)
        if isinstance(value, (list, tuple)):
            # If it's a sequence (list or tuple), scale each element
            rc[key] = [v * sc_fac for v in value]
        elif isinstance(value, (int, float)):
            # Check if it's a number (int/float)
            rc[key] = value * sc_fac

    return rc


def _get_figdim(nrows, ncols, aspect_ratio=None, width_pt=None):
    if width_pt == "thesis":
        fig_width_pt = 426.79135
    elif width_pt == "beamer":
        fig_width_pt = 307.28987
    elif isinstance(width_pt, (int, float)):
        fig_width_pt = width_pt
    else:
        # 'one column' by default
        fig_width_pt = 469.75502

    num_plots = nrows * ncols
    if num_plots == 1:
        fig_width_pt = fig_width_pt * 0.6

    # Set max 5 columns, adjust columns based on total plots
    ncols = min(ncols, 5)

    # Calculate number of rows and columns based on num_plots
    nrows = math.ceil(num_plots / ncols)  # Calculate required rows

    width_pt_per_plt = fig_width_pt / ncols
    # Convert from pt to inches as latex recognizes 72 dpi by default
    inches_per_pt = 1 / 72.27
    # Figure width in inches
    width_in_per_plts = width_pt_per_plt * inches_per_pt
    fig_width_in = fig_width_pt * inches_per_pt

    if aspect_ratio is None:
        if ncols < 4:
            aspect_ratio = 3 / 4
        else:
            aspect_ratio = 1

    # Figure height in inches
    height_in_per_plts = width_in_per_plts * aspect_ratio
    fig_height_in = height_in_per_plts * nrows

    return (fig_width_in, fig_height_in), nrows, ncols


def subplots(  # pylint: disable=too-many-arguments, too-many-locals
    nrows: int = 1,
    ncols: int = 1,
    *,
    aspect_ratio=None,
    width_pt=None,
    sharex: Union[bool, Literal["none", "all", "row", "col"]] = False,
    sharey: Union[bool, Literal["none", "all", "row", "col"]] = False,
    squeeze: bool = True,
    width_ratios: Optional[Sequence[float]] = None,
    height_ratios: Optional[Sequence[float]] = None,
    subplot_kw: Optional[Dict[str, Any]] = None,
    gridspec_kw: Optional[Dict[str, Any]] = None,
    **fig_kw,
) -> tuple[Figure, Any]:
    """
    Publication-style subplots with automatic figsize and font scaling,
    without affecting global Matplotlib state.
    """

    num_plots = nrows * ncols

    # Use custom width_map if provided, otherwise fallback to the default
    figsize, nrows, ncols = _get_figdim(nrows, ncols, aspect_ratio, width_pt)

    # Check if figsize is provided in fig_kw, and if so, pop it from fig_kw
    # Default to the calculated figsize if not in fig_kw
    figsize_from_kw = fig_kw.pop("figsize", figsize)

    with style.context(PUBLICATION_STYLE):
        base_rc = plt.rcParams.copy()

    scaled_rc = _update_style_with_figsize(figsize_from_kw, figsize, base_rc)

    with rc_context(scaled_rc):
        fig = plt.figure(figsize=figsize_from_kw, **fig_kw)

        axs = fig.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=sharex,
            sharey=sharey,
            squeeze=squeeze,
            subplot_kw=subplot_kw,
            gridspec_kw=gridspec_kw,
            height_ratios=height_ratios,
            width_ratios=width_ratios,
        )

        # Forward the kwargs to plt.subplots()
        fig.subplots_adjust(hspace=0.4 / nrows, wspace=0.6 / ncols)

        # Flatten axs for easier iteration
        if num_plots > 1:
            axs = axs.flatten()
            # Hide unused subplots
            for j in range(num_plots, len(axs)):
                axs[j].axis("off")

    return fig, axs


# Forward all method calls to plt
def __getattr__(name):
    # This will forward the method calls to matplotlib.pyplot
    return getattr(plt, name)
