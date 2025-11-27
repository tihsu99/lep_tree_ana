import matplotlib.pyplot as plt
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
        fig, (ax, ax_ratio) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(8, 8), dpi=300)

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


def do_control_plot(
    dl_dict,
    func_get_variable,
    bin_edges,
    x_label="X-axis",
    title="Control Plot",
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
    fig, (ax, ax_ratio) = plt.subplots(2, 1, dpi=300, figsize=(8, 8), gridspec_kw={'height_ratios': [4, 1]})

    signal_keys = [key for key, val in dl_dict.items() if val.is_signal]
    background_keys = [key for key, val in dl_dict.items() if (not val.is_signal and not val.is_data)]
    data_keys = [key for key, val in dl_dict.items() if val.is_data]
    num_MC_samples = len(signal_keys) + len(background_keys) 

    hists_MC = {}
    hist_MC_err2 = {}
    sum_MC_yields = 0
    hist_data = np.zeros(len(bin_edges)-1)
    for dl_name, dl in dl_dict.items():
        variable_values = func_get_variable(dl)
        hist, _ = np.histogram(variable_values, bins=bin_edges)
        hist_err2 = hist
        if not dl.is_data:
            scale = dl.norm_factor / dl.initial_total_num_events
            hist =   hist * scale
            hist_err2 = hist_err2 * scale**2
            sum_MC_yields += np.sum(hist)
            hists_MC[dl_name] = hist
            hist_MC_err2[dl_name] = hist_err2
        else:
            hist_data += hist

    # Plot MC stacked
    norm_cumulative_MC = np.zeros(len(bin_edges)-1)
    norm_cumulative_MC_err2 = np.zeros(len(bin_edges)-1)
    color_iterator = get_color_iterator(num_MC_samples)
    for dl_name in background_keys + signal_keys:
        hist = hists_MC.get(dl_name, np.zeros(len(bin_edges)-1))
        normhist = hist / sum_MC_yields # normalize total MC to 1
        color = next(color_iterator)
        ax.bar(bin_edges[:-1], normhist, bottom=norm_cumulative_MC, width=np.diff(bin_edges), align='edge', color=color, label=dl_name, alpha=0.7, edgecolor='black')
        norm_cumulative_MC += normhist
        norm_cumulative_MC_err2 += hist_MC_err2.get(dl_name, np.zeros(len(bin_edges)-1)) / (sum_MC_yields**2)

    # Plot data
    normhist_data = hist_data / np.sum(hist_data)
    ax.errorbar((bin_edges[:-1] + bin_edges[1:]) / 2, normhist_data, yerr=np.sqrt(hist_data) / np.sum(hist_data), fmt='o', color='black', label='Data')
    ax.set_yscale('log')
    ax.set_ylabel('Normalized Events')
    ax.set_title(title)
    ax.legend(loc='best')

    # Create ratio plot
    normhist_data_err = np.sqrt(hist_data) / np.sum(hist_data)
    norm_cumulative_MC_err = np.sqrt(norm_cumulative_MC_err2)
    ratio = normhist_data / norm_cumulative_MC
    ratio_err = ratio * np.sqrt( (normhist_data_err / normhist_data)**2 + (norm_cumulative_MC_err / norm_cumulative_MC)**2 )
    ax_ratio.step((bin_edges[:-1] + bin_edges[1:]) / 2, ratio, where='mid', color='black')
    ax_ratio.errorbar((bin_edges[:-1] + bin_edges[1:]) / 2, ratio, yerr=ratio_err, fmt='o', color='black')
    ax_ratio.set_xlabel(x_label)
    ax_ratio.set_ylabel('Data / MC')
    ax_ratio.set_ylim(0, 2)
    ax_ratio.axhline(1, color='gray', linestyle=':')

    return fig, ax, ax_ratio





