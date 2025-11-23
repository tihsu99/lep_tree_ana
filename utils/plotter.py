import matplotlib.pyplot as plt



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
        fig, (ax, ax_ratio) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Plot the two datasets
    ax.step(x, y1, where='mid', label=label1, alpha=0.7, color=color1, linestyle=linestyle1)
    ax.step(x, y2, where='mid', label=label2, alpha=0.7, color=color2, linestyle=linestyle2)
    if err1 is not None:
        ax.errorbar(x, y1, yerr=err1, fmt='o', color=color1)
    if err2 is not None:
        ax.errorbar(x, y2, yerr=err2, fmt='o', color=color2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
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

    ax_ratio.set_ylabel(ratio_ylabel)
    ax_ratio.axhline(1, color='gray', linestyle=':')
    if ratio_label is not None:
        ax_ratio.legend(loc='upper right')
    return (ax, ax_ratio)


