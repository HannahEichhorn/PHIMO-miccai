import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib.patches as mpatches


def quick_imshow(img, title="", cmap="gray", vmin=None, vmax=None,
                 colorbar=False, text=None, save_path=None,
                 zoom_coords=None, show_rectangle=False):
    """Quickly show an image  with optional title, colorbar, text,
    saving and zoom-in"""

    shift = 3

    fig, ax = plt.subplots(figsize=(3, 3*112/92))
    ax.imshow(np.roll(img.T, shift=shift, axis=0), cmap=cmap,
              vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")

    # Draw a rectangle around the zoom-in region if required
    if show_rectangle and zoom_coords is not None:
        rect = plt.Rectangle(
            xy=(zoom_coords[0]-0.5, zoom_coords[2]-0.5+shift),
            width=zoom_coords[1]-zoom_coords[0],
            height=zoom_coords[3]-zoom_coords[2],
            fill=False, color='white', linewidth=1)
        ax.add_patch(rect)

    # Remove white space
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    if text is not None:
        plt.text(0, 0.4, text, fontsize=18.5, color="white", ha="left",
                 va="top")

    if save_path is not None:
        plt.savefig(save_path + ".svg", format="svg", dpi=300)
    plt.show()

    # Show the zoom-in plot if zoom_coords is provided
    if zoom_coords is not None:
        fig_ratio = (zoom_coords[3] - zoom_coords[2]) / (
                    zoom_coords[1] - zoom_coords[0])
        fig, ax = plt.subplots(figsize=(3, 3*fig_ratio))
        ax.imshow(img.T[zoom_coords[2]:zoom_coords[3], zoom_coords[0]:zoom_coords[1]],
                  cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis("off")
        # Remove white space
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0,
                            wspace=0)
        plt.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        if save_path is not None:
            plt.savefig(save_path + "_zoom.svg", format="svg", dpi=300)
        plt.show()

    # Show additional plot with colorbar if required
    if colorbar:
        plt.imshow(img.T, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=20)
        plt.axis("off")
        plt.show()


def make_violin_plot(metrics, statistical_tests, img_types_plot,
                     metric_types_plot, motion_types, show_title=False,
                     y_lim=None, height_brackets=1, bracket_at_top=True,
                     plot_legend=True, cols=None, alphas=None, save_path=None):
    """ Create violin plots for the given metric types and masks. """

    ylabel_dict = {
        "t2s-MAE": "MAE [ms]",
        "t2s-SSIM": "SSIM",
        "t2s-FSIM": "FSIM",
        "t2s-MAE-diff": "$\Delta$ MAE [ms]",
        "t2s-SSIM-diff": "$\Delta$ SSIM ",
        "t2s-FSIM-diff": "$\Delta$ FSIM",
        "img-SSIM": "SSIM",
        "img-FSIM": "FSIM",
        "img-PSNR": "PSNR",
    }
    x_label_dict = {
        "motion_reg": "No \nMoCo",
        "phimo_reg": "PHIMO",
        "hrqrmoco_reg": "HR/QR-\nMoCo",
        "bootstrap_reg": "OR-BA",
        "phimo_excl_reg": "PHIMO\n(Excl.)",
        "gray_matter": "gray matter",
        "white_matter": "white matter",
    }

    for metric_type in metric_types_plot:
        for motion_type in motion_types:
            fig, ax = plt.subplots(figsize=(6 / 5 * 4.2, 4))
            positions_ = np.arange(0, len(img_types_plot)).astype(float)
            offsets = [0, 5] if len(img_types_plot) == 4 else [0, 6]
            masks = ["gray_matter", "white_matter"]
            if cols is None:
                cols = ["tab:blue",  "tab:green", "darkorange", "tab:gray"]
                if len(img_types_plot) == 5:
                    cols = ["tab:blue", "tab:green", "darkorange", "goldenrod",
                            "tab:gray"]
            if alphas is None:
                alphas = [1] * len(cols)
            for mask,  offset in zip(
                    masks,
                    offsets):
                positions = positions_ + offset
                vp = ax.violinplot(
                    [metrics[metric_type][mask][motion_type][img] for img in
                     img_types_plot],
                    positions=positions, showmeans=True, showextrema=False,
                    widths=0.4
                )
                for v, col, al in zip(vp['bodies'], cols, alphas):
                    v.set_facecolor(col)
                    v.set_edgecolor(col)
                    v.set_alpha(al)
                vp['cmeans'].set_edgecolor("#E8E8E8")
                if mask == "gray_matter":
                    labels = []
                    for label, col in zip([x_label_dict[x] for x in img_types_plot],
                                          cols):
                        labels.append((mpatches.Patch(color=col, ),
                                       label))
                if statistical_tests is not None:
                    show_brackets(
                        p_cor=np.array(
                            statistical_tests[metric_type][mask][motion_type][
                                "p_values"]),
                        ind=np.array(
                            statistical_tests[metric_type][mask][motion_type][
                                "combinations"]),
                        violins=positions,
                        height=height_brackets,
                        col='gray',
                        top=bracket_at_top,
                        p_value_threshold=0.001,
                        ax=ax,
                        ylim=y_lim
                    )
            plt.ylabel(ylabel_dict[metric_type], fontsize=18)
            if show_title:
                plt.title("{} motion".format(motion_type))
            plt.xticks(
                np.array(offsets) + 1.5,
                [x_label_dict[x] for x in masks],
                fontsize=18)
            plt.yticks(fontsize=14)
            if y_lim is not None:
                plt.ylim(y_lim[0], y_lim[1])
            for y_tick in ax.get_yticks():
                ax.axhline(y_tick, color='lightgray', linestyle='--',
                           linewidth=0.5, zorder=0)
            plt.tight_layout()
            if save_path is not None:
                plt.savefig(save_path + metric_type + "_" + motion_type +
                            ".svg", format="svg", dpi=300, bbox_inches="tight")
            plt.show()

        # plot legend in separate figure
        if plot_legend:
            plt.figure(figsize=(6 / 5 * 4.2, 4))
            plt.legend(*zip(*labels), fontsize=13)
            plt.show()


def show_brackets(p_cor, ind, violins, height, col='dimgrey', top=True,
                  p_value_threshold=0.05, ax=None, ylim=None):
    """Show brackets with 'N' for p_values larger than p_value_threshold.

    Parameters
    ----------
    p_cor : np.array
        Corrected p-values.
    ind : np.array
        Indices of the violins to annotate.
    violins : np.array
        Violin plot positions.
    height : float
        Height of the brackets.
    col : str
        Color of the brackets.
    top : bool
        Whether to show the brackets above the violins.
    p_value_threshold : float
        Threshold for p-values to show the brackets.
    ax : matplotlib.axes.Axes
        Axes to plot the brackets on.
    ylim : list
        Y-axis limits of the plot.
    """

    # Calculate the height of the plot for scaling
    if ylim is not None:
        plot_height = ylim[1] - ylim[0]
    else:
        plot_height = ax.get_ylim()[1] - ax.get_ylim()[0]

    ind = ind[p_cor >= p_value_threshold]
    heights = [height for s in ind]
    previous_brackets = []
    for i in range(len(ind)):
        for tmp in previous_brackets:
            if ind[i][0] in tmp or ind[i][1] in tmp:
                if top:
                    heights[i] += 0.12*plot_height
                else:
                    heights[i] -= 0.12*plot_height
        previous_brackets.append(ind[i])

    # Calculate the height of the bracket as a ratio of the plot height
    bracket_heights = 0.02 * plot_height

    for i in range(len(ind)):
        barplot_annotate_brackets(
            violins[ind[i][0]], violins[ind[i][1]],  heights[i],
            bracket_heights, col, 13, top, label="N"
        )


def barplot_annotate_brackets(num1, num2, y, bracket_height,
                              col, fs, top, label=""):
    """Annotate bar plot with brackets and text."""
    lx, ly = num1, y
    rx, ry = num2, y

    if top:
        bary = [y, y + bracket_height, y + bracket_height, y]
        text_position = y + bracket_height
        va = 'bottom'
    else:
        bary = [y + bracket_height, y, y, y + bracket_height]
        text_position = y - bracket_height
        va = 'top'

    plt.plot([lx, lx, rx, rx], bary, c=col)
    plt.text((lx + rx) / 2, text_position, label, ha='center', va=va,
             fontsize=fs, color=col)


def statistical_testing(keys, metric):
    """
    Perform statistical testing using Wilcoxon signed rank tests and multiple
    comparison correction (FDR) on metric values for pairs of keys.

    Parameters
    ----------
    keys : list
        Sorted keys of the metrics dictionary.
    metric : dict
        Dictionary containing metric values for each key.

    Returns
    -------
    tuple
        A tuple containing combinations and p-values for statistical testing.
    """

    combinations = list(itertools.combinations(np.arange(0, len(keys)),
                                               2))
    p_values = []
    for comb in combinations:
        p_values.append(wilcoxon(metric[keys[comb[0]]],
                                 metric[keys[comb[1]]],
                                 alternative='two-sided')[1])
    rej, p_values_cor, _, __ = multipletests(p_values, alpha=0.05,
                                             method='fdr_bh', is_sorted=False,
                                             returnsorted=False)

    insign = np.where(p_values_cor >= 0.05)
    for ins in insign[0]:
        print(keys[combinations[ins][0]],
              keys[combinations[ins][1]],
              " No significant difference.")

    return combinations, p_values
