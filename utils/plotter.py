import matplotlib.pyplot as plt
import copy
import numpy as np
from utils.common_functions import get_color_iterator



def do_ratio_plot(
    x,
    y1,
    y2,
    err1=None,
    err2=None,
    ax=None,
    ax_ratio=None,
    color1="blue",
    color2="red",
    linestyle1='-',
    linestyle2='-',
    label1="Data 1",
    label2="Data 2",
    title="Ratio Plot",
    xlabel="X-axis",
    ylabel="Y-axis",
    ratio_label=None,
    ratio_color="black",
    ratio_ylabel="Ratio (Data 1 / Data 2)",
    legend_loc="best",
):
    """
    Create a ratio plot with two datasets.

    Parameters:
    - ax: matplotlib Axes object to plot on.
    - x: array-like, x values.
    - y1: array-like, y values for the first dataset.
    - y2: array-like, y values for the second dataset.
    - label1: str, label for the first dataset.
    - label2: str, label for the second dataset.
    - title: str, title of the plot.
    - xlabel: str, label for the x-axis.
    - ylabel: str, label for the y-axis.
    - ratio_ylabel: str, label for the ratio y-axis.
    - legend_loc: str, location of the legend.

    Returns:
    - ax: matplotlib Axes object with the plot.
    """
    if ax is None or ax_ratio is None:
        fig, (ax, ax_ratio) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(8, 6), dpi=300)

    # Plot the two datasets
    ax.step(x, y1, where='mid', label=label1, alpha=0.7, color=color1, linestyle=linestyle1)
    ax.step(x, y2, where='mid', label=label2, alpha=0.7, color=color2, linestyle=linestyle2)
    if err1 is not None:
        ax.errorbar(x, y1, yerr=err1, fmt='o', color=color1)
    if err2 is not None:
        ax.errorbar(x, y2, yerr=err2, fmt='o', color=color2)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc)

    # Create ratio plot
    ratio = []
    ratio_err = []
    for i in range(len(y1)):
        if y2[i] != 0:
            ratio.append(y1[i] / y2[i])
            if err1 is not None and err2 is not None:
                ratio_err.append(ratio[-1] * ((err1[i]/y1[i])**2 + (err2[i]/y2[i])**2)**0.5)
            else:
                ratio_err.append(0)
        else:
            ratio.append(1)
            ratio_err.append(0)
    # Create an axis on the bottom for the ratio plot
    ax_ratio.step(x, ratio, where='mid', color=ratio_color)
    if any(ratio_err):
        ax_ratio.errorbar(x, ratio, yerr=ratio_err, fmt='o', color=ratio_color, label=ratio_label)

    ax_ratio.set_xlabel(xlabel)
    ax_ratio.set_ylabel(ratio_ylabel)
    ax_ratio.axhline(1, color='gray', linestyle=':')
    if ratio_label is not None:
        ax_ratio.legend(loc='upper right')
    return (ax, ax_ratio)

def do_control_plot_from_hists(
    hist_data,
    hist_MC_dict,
    hist_MC_err2_dict,
    bin_edges,
    x_label="X-axis",
    title="Control Plot",
    normalize=True,
    log_scale=True,
):
    """
    Create a control plot comparing data and MC from precomputed histograms.
    Parameters:
    - data_hist: array-like, histogram of data.
    - mc_hists_dict: dict of array-like, histograms of MC samples.
    - mc_hists_err2_dict: dict of array-like, squared errors of MC histograms.
    - x_label: str, label for the x-axis.
    - title: str, title of the plot.
    Returns:
    - fig: matplotlib Figure object.
    - ax: matplotlib Axes object with the plot.
    - ax_ratio: matplotlib Axes object with the ratio plot.
    """
    fig, (ax, ax_ratio) = plt.subplots(2, 1, dpi=300, figsize=(8, 6), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)

    num_bins = len(bin_edges) - 1
    # Calculate sum of MC yields
    sum_MC_yields = 0
    for hist in hist_MC_dict.values():
        sum_MC_yields += np.sum(hist)

    # Plot MC stacked
    cumulative_MC = np.zeros(num_bins)
    cumulative_MC_err2 = np.zeros(num_bins)
    color_iterator = get_color_iterator(len(hist_MC_dict))
    for sample_name in hist_MC_dict.keys():
        hist = hist_MC_dict.get(sample_name, np.zeros(num_bins))
        normhist = hist / sum_MC_yields if normalize else hist
        color = next(color_iterator)
        ax.bar(bin_edges[:-1], normhist, bottom=cumulative_MC, width=np.diff(bin_edges), align='edge', color=color, label=sample_name, alpha=0.7, edgecolor='black')
        cumulative_MC += normhist
        cumulative_MC_err2 += hist_MC_err2_dict.get(sample_name, np.zeros(num_bins)) / (sum_MC_yields**2 if normalize else 1)

    # plot uncertainty band for MC
    cumulative_MC_err = np.sqrt(cumulative_MC_err2)
    ax.fill_between((bin_edges[:-1] + bin_edges[1:]) / 2,
                    cumulative_MC - cumulative_MC_err,
                    cumulative_MC + cumulative_MC_err,
                    step='mid',
                    color='gray',
                    alpha=0.5,
                    label='MC Uncertainty')

    # Plot data
    data_yields = np.sum(hist_data)
    if normalize:
        hist_data = hist_data / data_yields
        data_err = np.sqrt(hist_data) / data_yields
    else:
        hist_data = hist_data
        data_err = np.sqrt(hist_data)
    ax.errorbar((bin_edges[:-1] + bin_edges[1:]) / 2, hist_data, yerr=data_err, fmt='o', color='black', label='Data')
    # ax.set_yscale('log')
    if normalize:
        ax.set_ylabel('Normalized Events')
    else:
        ax.set_ylabel('Events')
    ax.set_title(title)
    ax.legend(loc='best')

    # Create ratio plot
    ratio = hist_data / cumulative_MC
    ratio_err = ratio * np.sqrt( (data_err / hist_data)**2 + (cumulative_MC_err / cumulative_MC)**2 )
    ax_ratio.step((bin_edges[:-1] + bin_edges[1:]) / 2, ratio, where='mid', color='black')
    ax_ratio.errorbar((bin_edges[:-1] + bin_edges[1:]) / 2, ratio, yerr=ratio_err, fmt='.', color='black')
    ax_ratio.set_xlabel(x_label)
    ax_ratio.set_ylabel('Data / MC')
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.axhline(1, color='gray', linestyle=':')
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylim(bottom=1)
    else:
        ax.set_ylim(bottom=0)

    return fig, ax, ax_ratio




def do_control_plot(
    dl_dict,
    region_name,
    func_get_variable,
    bin_edges,
    x_label="X-axis",
    title="Control Plot",
    luminosity=None,
    normalize=True,
    log_scale=True,
    blind=False,
):
    """
    Create a control plot comparing data and MC.
    Parameters:
    - dl_dict: dict of DataLoader objects.
    - func_get_variable: function to extract the variable of interest from a DataLoader.
    - bin_edges: array-like, edges of the histogram bins.
    - x_label: str, label for the x-axis.
    - title: str, title of the plot.
    Returns:
    - fig: matplotlib Figure object.
    - ax: matplotlib Axes object with the plot.
    - ax_ratio: matplotlib Axes object with the ratio plot.
    """
    print("Control Plot:", title)

    signal_keys = [key for key, val in dl_dict.items() if val.is_signal]
    background_keys = [key for key, val in dl_dict.items() if (not val.is_signal and not val.is_data)]
    data_keys = [key for key, val in dl_dict.items() if val.is_data]
    num_MC_samples = len(signal_keys) + len(background_keys) 

    hists_MC = {}
    hist_MC_err2 = {}
    sum_MC_yields = 0
    hist_data = np.zeros(len(bin_edges)-1)
    for dl_name, dl in dl_dict.items():
        events = dl.data[region_name]
        # mask = (events['flags_valid'] > 0) & (events['theta_cm'] > 0.6) & (events['mtautau'] > 80)
        # import awkward as ak
        # mask = ak.fill_none(mask, False)
        # events = events[mask]
        weights = events['weight'].to_numpy() if 'weight' in events.fields else np.ones(len(events))
        variable_values = np.array([])
        if len(events) > 0:
            variable_values = func_get_variable(events)
        if type(variable_values) is tuple:
            variable_values, weights = variable_values
        variable_values = np.asarray(variable_values).flatten()
        weights = np.asarray(weights).flatten()
        hist, _ = np.histogram(variable_values, bins=bin_edges, weights=weights)
        hist_err2, _ = np.histogram(variable_values, bins=bin_edges, weights=weights**2)
        if not dl.is_data:
            sum_MC_yields += np.sum(hist)
            hists_MC[dl_name] = hist
            hist_MC_err2[dl_name] = hist_err2
        elif dl.is_data and (not blind):
            hist_data += hist

    return do_control_plot_from_hists(
        hist_data=hist_data,
        hist_MC_dict=hists_MC,
        hist_MC_err2_dict=hist_MC_err2,
        bin_edges=bin_edges,
        x_label=x_label,
        title=title,
        normalize=normalize,
        log_scale=log_scale,
    )




def _weighted_quantile(values, quantiles, sample_weight=None):
    """
    Compute weighted quantiles of 1D `values`.
    quantiles in [0, 1]. Returns array with same shape as quantiles.
    """
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)

    if sample_weight is None:
        return np.quantile(values, quantiles)

    w = np.asarray(sample_weight, dtype=float)
    if values.size == 0:
        return np.full_like(quantiles, np.nan, dtype=float)

    # sort by values
    sorter = np.argsort(values)
    v = values[sorter]
    w = w[sorter]

    w = np.clip(w, 0.0, np.inf)
    ws = np.sum(w)
    if ws <= 0:
        return np.full_like(quantiles, np.nan, dtype=float)

    cdf = np.cumsum(w) / ws
    return np.interp(quantiles, cdf, v)


def plot_y_vs_x(
    x,
    y,
    weight=None,
    *,
    bins=30,
    x_range=None,
    # stat="median",                 # "mean" or "median" (median uses weighted quantile 0.5 if weight provided)
    stat="mean",                 # "mean" or "median" (median uses weighted quantile 0.5 if weight provided)
    band=("68", "95"),           # any subset of {"68", "95"}; or () / None for no band
    band_method="quantile",      # "quantile" (central intervals) or "stderr" (mean ± z*stderr)
    min_count=20,                # minimum entries per bin to draw stat/bands
    ax=None,
    fig=None,
    label=None,
    color=None,
    linestyle="-",
    linewidth=2.0,
    draw_points=False,
    points_kwargs=None,
    band_alpha=0.20,
    band_edge=False,
    band_label=False,
    step="mid",                  # "mid" uses bin centers; "stairs" uses bin edges (matplotlib stairs)
):
    """
    Plot y as a function of x with optional weights and uncertainty bands.

    Parameters
    ----------
    x, y : array-like (same length)
    weight : array-like or None
        Event weights (same length as x/y).
    bins : int or array-like
        Number of x-bins or explicit bin edges.
    x_range : (xmin, xmax) or None
        Range for binning when bins is an int.
    stat : {"mean", "median"}
        Central curve per bin.
    band : tuple/list like ("68","95") or ()
        Draw central intervals per x-bin.
    band_method : {"quantile", "stderr"}
        - "quantile": bands are central credible intervals of y within each x-bin
          (16–84 for 68%, 2.5–97.5 for 95%), computed with weights if provided.
        - "stderr": bands are mean ± z * stderr(mean), with effective N from weights.
    min_count : int
        Minimum number of events in a bin to compute stats/bands.
    ax, fig : matplotlib Axes/Figure or None
        If ax is provided, draw on it (good for overlays). If not, create new.
    step : {"mid","stairs"}
        Draw central curve at bin centers or as a step function.

    Returns
    -------
    fig, ax
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y must have the same length, got {x.shape[0]} vs {y.shape[0]}")

    if weight is not None:
        w = np.asarray(weight)
        if w.shape[0] != x.shape[0]:
            raise ValueError(f"weight must have the same length as x/y, got {w.shape[0]} vs {x.shape[0]}")
    else:
        w = None

    # drop non-finite rows
    mask = np.isfinite(x) & np.isfinite(y)
    if w is not None:
        mask &= np.isfinite(w)
    x = x[mask]
    y = y[mask]
    w = w[mask] if w is not None else None

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    else:
        fig = ax.figure if fig is None else fig

    # optional scatter of points
    if draw_points:
        pk = dict(s=8, alpha=0.25)
        if points_kwargs:
            pk.update(points_kwargs)
        ax.scatter(x, y, c=color, label=None, **pk)

    # define bins
    if np.isscalar(bins):
        if x_range is None:
            xmin, xmax = np.nanmin(x), np.nanmax(x)
        else:
            xmin, xmax = x_range
        edges = np.linspace(xmin, xmax, int(bins) + 1)
    else:
        edges = np.asarray(bins, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("bins must be an int or a 1D array of bin edges")

    centers = 0.5 * (edges[:-1] + edges[1:])
    nb = edges.size - 1

    y_stat = np.full(nb, np.nan, dtype=float)
    y_lo68 = np.full(nb, np.nan, dtype=float)
    y_hi68 = np.full(nb, np.nan, dtype=float)
    y_lo95 = np.full(nb, np.nan, dtype=float)
    y_hi95 = np.full(nb, np.nan, dtype=float)

    # assign bins
    bin_idx = np.digitize(x, edges) - 1
    in_range = (bin_idx >= 0) & (bin_idx < nb)
    x, y = x[in_range], y[in_range]
    w = w[in_range] if w is not None else None
    bin_idx = bin_idx[in_range]

    # per-bin computations
    for b in range(nb):
        sel = bin_idx == b
        if not np.any(sel):
            continue
        yy = y[sel]
        if yy.size < min_count:
            continue

        if w is None:
            ww = None
            if stat == "mean":
                y_stat[b] = float(np.mean(yy))
            elif stat == "median":
                y_stat[b] = float(np.median(yy))
            else:
                raise ValueError("stat must be 'mean' or 'median'")
        else:
            ww = w[sel].astype(float)
            ww = np.clip(ww, 0.0, np.inf)
            if np.sum(ww) <= 0:
                continue
            if stat == "mean":
                y_stat[b] = float(np.average(yy, weights=ww))
            elif stat == "median":
                y_stat[b] = float(_weighted_quantile(yy, 0.5, sample_weight=ww))
            else:
                raise ValueError("stat must be 'mean' or 'median'")

        if not band:
            continue

        if band_method == "quantile":
            if "68" in band:
                q16, q84 = _weighted_quantile(yy, [0.16, 0.84], sample_weight=ww)
                y_lo68[b], y_hi68[b] = float(q16), float(q84)
            if "95" in band:
                q025, q975 = _weighted_quantile(yy, [0.025, 0.975], sample_weight=ww)
                y_lo95[b], y_hi95[b] = float(q025), float(q975)

        elif band_method == "stderr":
            # Only meaningful with mean; if median requested, still use mean for stderr band.
            if w is None:
                mu = float(np.mean(yy))
                # standard error of mean
                se = float(np.std(yy, ddof=1) / np.sqrt(yy.size)) if yy.size > 1 else np.nan
            else:
                mu = float(np.average(yy, weights=ww))
                # weighted variance and effective N
                wsum = float(np.sum(ww))
                w2sum = float(np.sum(ww * ww))
                neff = (wsum * wsum / w2sum) if w2sum > 0 else np.nan
                var = float(np.average((yy - mu) ** 2, weights=ww))
                se = float(np.sqrt(var / neff)) if np.isfinite(neff) and neff > 1 else np.nan

            if "68" in band:
                z = 1.0
                y_lo68[b], y_hi68[b] = mu - z * se, mu + z * se
            if "95" in band:
                z = 1.959963984540054  # ~N(0,1) 97.5% quantile
                y_lo95[b], y_hi95[b] = mu - z * se, mu + z * se
        else:
            raise ValueError("band_method must be 'quantile' or 'stderr'")

    # draw bands first (so curve is on top)
    def _fill(lo, hi, tag):
        ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(centers)
        if not np.any(ok):
            return
        fill_label = None
        if band_label and label is not None:
            fill_label = f"{label} ({tag} band)"
        ax.fill_between(
            centers[ok], lo[ok], hi[ok],
            alpha=band_alpha,
            color=color,
            label=fill_label,
            linewidth=1.0 if band_edge else 0.0,
            edgecolor=color if band_edge else None,
        )

    if band:
        if "95" in band:
            _fill(y_lo95, y_hi95, "95%")
        if "68" in band:
            _fill(y_lo68, y_hi68, "68%")

    # draw central curve
    okc = np.isfinite(y_stat) & np.isfinite(centers)
    if np.any(okc):
        if step == "stairs":
            # matplotlib stairs expects edges and values length nb
            ax.stairs(y_stat, edges, label=label, color=color, linestyle=linestyle, linewidth=linewidth)
        else:
            ax.plot(centers[okc], y_stat[okc], label=label, color=color, linestyle=linestyle, linewidth=linewidth)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)

    return fig, ax



