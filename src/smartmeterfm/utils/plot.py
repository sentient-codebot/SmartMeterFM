import logging
import os

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from matplotlib.ticker import FuncFormatter
from numpy import ndarray


@plt.style.context("smartmeterfm.utils.article")
def plot_sampled_data_v2(
    samples: dict[str, Float[ndarray, "batch sequence"]],
    save_filepath: str | None = None,
    num_samples_to_plot: int | None = 1000,
    seed=0000,
):
    """this plots only one season"""
    # alpha adjustment factor
    dot_alpha_coef = 10  # 20 / 1000 = 0.020
    line_alpha_coef = 72  # 5 / 1000 = 0.005

    num_subfigure = len(samples)
    num_samples_to_plot = min(
        num_samples_to_plot, min([data.shape[0] for data in samples.values()])
    )
    if save_filepath is None:
        save_filepath = "results/untitled_plot_sampled_data.png"

    # Step 2: Format
    if isinstance(list(samples.values())[0], torch.Tensor):
        samples = {season: data.cpu().numpy() for season, data in samples.items()}

    # Step 3: Visualize samples
    rng = np.random.default_rng()  # for selecting indices to plot
    fig = plt.figure(figsize=(10, 2.5 * num_subfigure))
    gspec = gridspec.GridSpec(nrows=num_subfigure, ncols=1, figure=fig)

    def _get_color(sample_day, _min_day_consumption, _max_day_consumption):
        line_cmap = mpl.cm.get_cmap("RdYlBu_r")
        _line_color_norm = mpl.colors.Normalize(
            vmin=_min_day_consumption, vmax=_max_day_consumption
        )
        return line_cmap(_line_color_norm(sample_day.sum()))

    for idx, (name, sample) in enumerate(samples.items()):
        # Get data
        rng = np.random.default_rng(seed)

        indices = rng.permutation(sample.shape[0])[:num_samples_to_plot]
        sample = sample[indices]

        #  -- sort by total consumption --
        samples_daily_sum = sample.sum(axis=1)
        sample_idx = np.argsort(samples_daily_sum)

        sample = sample[sample_idx]
        # -- end of sorting --

        # Statistics
        _sample_mean = np.mean(sample, axis=0)
        _sample_std = np.std(sample, axis=0)

        max_day_consumption = sample.sum(axis=1).max()
        min_day_consumption = sample.sum(axis=1).min()

        # Normalize
        data_vec_dim = sample.shape[1]  # data.shape = [num_samples_to_plot, 96]

        ### subplot 0
        ax_sample = fig.add_subplot(gspec[idx, 0])
        ax_sample.grid(True)
        for index in range(num_samples_to_plot):
            ax_sample.scatter(
                np.arange(data_vec_dim),
                sample[index, :],
                color=_get_color(
                    sample[index, :], min_day_consumption, max_day_consumption
                ),
                s=0.5,
                alpha=min(dot_alpha_coef / num_samples_to_plot, 1),
                rasterized=True,
            )
            ax_sample.plot(
                np.arange(data_vec_dim),
                sample[index, :],
                linewidth=2.5,
                color=_get_color(
                    sample[index, :], min_day_consumption, max_day_consumption
                ),
                alpha=min(line_alpha_coef / num_samples_to_plot, 1),
                rasterized=True,
            )

        ax_sample.set_xlabel("Time Step [-]")
        ax_sample.set_ylabel("Power [W]")
        ax_sample.set_xlim([0, data_vec_dim - 1])
        # ax_sample.set_ylim([common_minimum, common_maximum])
        ax_sample.set_title(f"{name}")

    os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
    plt.savefig(save_filepath, dpi=300, bbox_inches="tight")

    return fig


def process_mask(mask):
    """
    Process a binary mask to identify continuous regions where mask == 1.

    Parameters:
    -----------
    mask : array-like of 0s and 1s
        Binary mask array.

    Returns:
    --------
    regions : list of tuples
        List of (start_idx, end_idx) tuples for each continuous region where mask == 1.
    """
    mask = np.array(mask, dtype=int)
    regions = []
    in_region = False
    start_idx = None

    for i, val in enumerate(mask):
        if val == 1 and not in_region:
            # Start of a masked region
            start_idx = i
            in_region = True
        elif val == 0 and in_region:
            # End of a masked region
            regions.append((start_idx, i - 1))
            in_region = False

    # If the mask ends with 1, close the last region
    if in_region:
        regions.append((start_idx, len(mask) - 1))

    return regions


def plot_time_series_comparison(
    generated: torch.Tensor,
    real: torch.Tensor,
    output_path: str,
    title: str = "Time Series Comparison",
    mask: torch.Tensor | None = None,
    ymax: int | None = None,
    ymin: int | None = None,
):
    """
    Create a plot comparing generated and real time series data in a style similar to plot_sampled_data_v2.

    Args:
        generated: Generated data tensor
        real: Real data tensor
        output_path: Path to save the plot
        title: Title for the plot
        mask: a 1d mask, length == data sequence length. 1 == grey, 0 == white.
    """
    # Check mask
    if mask is not None:
        _data_seq_len = (
            generated.shape[1] * generated.shape[2]
            if generated.dim() == 3
            else generated.shape[1]
        )
        if len(mask) != _data_seq_len:
            raise ValueError("length of mask should be equal to data sequence length.")
        regions = process_mask(mask)

    # Apply custom style
    with plt.style.context("smartmeterfm.utils.article_compatible"):
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], figure=fig)

        # Parameters for visualization
        dot_alpha_coef = 10  # Alpha coefficient for scatter dots
        line_alpha_coef = 72  # Alpha coefficient for lines

        # Function to get color based on daily consumption
        def _get_color(sample_day, _min_day_consumption, _max_day_consumption):
            line_cmap = mpl.cm.get_cmap("RdYlBu_r")
            _line_color_norm = mpl.colors.Normalize(
                vmin=_min_day_consumption, vmax=_max_day_consumption
            )
            return line_cmap(_line_color_norm(sample_day.sum()))

        # Normalize and process data
        def prepare_data(data, ax, title_text):
            if data.dim() == 3:
                # If data is [batch, sequence, channel], flatten it to [batch, sequence*channel]
                data = rearrange(data, "b s c -> b (s c)")

            # Limit number of samples to plot
            max_samples_to_plot = min(100, data.shape[0])
            data = data[:max_samples_to_plot]

            # Calculate daily consumption
            samples_daily_sum = data.sum(dim=1)
            # Sort by total consumption
            sample_idx = torch.argsort(samples_daily_sum)
            data = data[sample_idx]

            # Get statistics
            _sample_mean = torch.mean(data, dim=0)
            _sample_std = torch.std(data, dim=0)
            max_day_consumption = data.sum(dim=1).max().item()
            min_day_consumption = data.sum(dim=1).min().item()

            # Plot samples
            data_vec_dim = data.shape[1]  # Time points in a day

            # Grid
            ax.grid(True)

            # Plot each sample
            for index in range(max_samples_to_plot):
                sample = data[index].cpu().numpy()
                color = _get_color(
                    data[index], min_day_consumption, max_day_consumption
                )

                # Scatter plot
                ax.scatter(
                    np.arange(data_vec_dim),
                    sample,
                    color=color,
                    s=0.5,
                    alpha=min(dot_alpha_coef / max_samples_to_plot, 1),
                    rasterized=True,
                )

                # Line plot
                ax.plot(
                    np.arange(data_vec_dim),
                    sample,
                    linewidth=2.5,
                    color=color,
                    alpha=min(line_alpha_coef / max_samples_to_plot, 1),
                    rasterized=True,
                )

            # Plot grey areas specified by mask
            if mask is not None:
                for _idx_region, (start, end) in enumerate(regions):
                    if _idx_region == len(regions) - 1:
                        ax.axvspan(
                            start, end, alpha=0.2, color="gray", label="Measured"
                        )
                        # only label the last region
                    else:
                        ax.axvspan(start, end, alpha=0.2, color="gray")
                    if start != 0:
                        ax.axvline(x=start, color="r", linestyle="--")
                    if end != data_vec_dim - 1:
                        ax.axvline(x=end, color="r", linestyle="--")

            # Set labels and title
            ax.set_xlabel("Time Step [-]")
            ax.set_ylabel("Normalized Power [-]")
            ax.set_xlim([0, data_vec_dim - 1])
            ax.set_ylim([ymin or -2, ymax or 2])
            ax.legend()
            ax.set_title(title_text)

            return ax

        # First subplot: Generated data
        ax_gen = fig.add_subplot(gs[0])
        prepare_data(
            generated, ax_gen, f"Generated Samples ({generated.shape[0]} samples)"
        )

        # Second subplot: Real data
        ax_real = fig.add_subplot(gs[1])
        prepare_data(real, ax_real, f"Real Samples ({real.shape[0]} samples)")

        # Add overall title
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"Saved time series comparison plot to {output_path}")


def plot_time_series_comparison_advanced(
    generated: torch.Tensor,
    real: torch.Tensor,
    output_path: str | None = None,
    title: str = "Time Series Comparison",
    mask: torch.Tensor | None = None,
    ymax: int | None = None,
    ymin: int | None = None,
    grid_generated_labels: list[str] | None = None,
    overlap_generated_label: str | None = None,
    layout: str = "overlap",  # New parameter: "overlap" or "grid"
    overlap_print_num_samples: bool = True,
) -> plt.figure:
    """
    Create a plot comparing generated and real time series data in a style similar to plot_sampled_data_v2.

    Args:
        generated: Generated data tensor
        real: Real data tensor
        output_path: Path to save the plot
        title: Title for the plot
        mask: a 1d mask, length == data sequence length. 1 == grey, 0 == white.
        ymax: Maximum value for y-axis
        ymin: Minimum value for y-axis
        grid_generated_labels: the title text above each grid cell in generated data (grid layout)
        overlap_generated_label: the title text for the generated Axes (overlap layout)
        overlap_print_num_samples: whether to print how many samples are displayed (overlap layout)
        layout: Plot layout:
                - "overlap": all samples overlaid in two subplots (default)
                - "grid": each sample in its own subplot in a grid (max 4x4)
                  Falls back to "overlap" if too many samples
    """
    # Check mask
    if mask is not None:
        _data_seq_len = (
            generated.shape[1] * generated.shape[2]
            if generated.dim() == 3
            else generated.shape[1]
        )
        if len(mask) != _data_seq_len:
            raise ValueError("length of mask should be equal to data sequence length.")
        regions = process_mask(mask)

    # Apply custom style
    with plt.style.context("smartmeterfm.utils.article_compatible"):
        # Determine figure size
        _W, _H = plt.rcParams["figure.figsize"]
        rc_diag = (_W**2 + _H**2) ** 0.5
        # Common parameters
        dot_alpha_coef = 10  # Alpha coefficient for scatter dots
        line_alpha_coef = 72  # Alpha coefficient for lines

        # Function to get color based on daily consumption
        def _get_color(sample_day, _min_day_consumption, _max_day_consumption):
            line_cmap = mpl.cm.get_cmap("RdYlBu_r")
            _line_color_norm = mpl.colors.Normalize(
                vmin=_min_day_consumption, vmax=_max_day_consumption
            )
            return line_cmap(_line_color_norm(sample_day.sum()))

        # Process and flatten data
        def _preprocess_data(data):
            if data.dim() == 3:
                # If data is [batch, sequence, channel], flatten it to [batch, sequence*channel]
                return rearrange(data, "b s c -> b (s c)")
            return data

        # Function to plot mask regions on an axis
        def _plot_mask_regions(ax, data_vec_dim):
            if mask is not None:
                for _idx_region, (start, end) in enumerate(regions):
                    if _idx_region == len(regions) - 1:
                        ax.axvspan(
                            start, end, alpha=0.2, color="gray", label="Measured"
                        )
                        # only label the last region
                    else:
                        ax.axvspan(start, end, alpha=0.2, color="gray")
                    if start != 0:
                        ax.axvline(x=start, color="r", linestyle="--")
                    if end != data_vec_dim - 1:
                        ax.axvline(x=end, color="r", linestyle="--")

        # Set common axis properties
        def _set_axis_properties(ax, title_text, data_vec_dim):
            ax.set_xlabel("Time Step [-]")
            ax.set_ylabel("Normalized Power [-]")
            ax.set_xlim([0, data_vec_dim - 1])
            ax.set_ylim([ymin or -2, ymax or 2])
            ax.grid(True)
            ax.set_title(title_text)

        # Determine layout to use
        if layout == "grid":
            # For grid layout, determine if we have too many samples
            max_grid_size = 4
            max_plots = max_grid_size * max_grid_size - 1  # -1 for real sample

            if generated.shape[0] > max_plots:
                logging.warning(
                    f"Too many samples ({generated.shape[0]}) for grid layout. "
                    f"Switching to overlap layout."
                )
                layout = "overlap"

        # Create plots based on layout
        if layout == "overlap":
            if overlap_generated_label is None:
                overlap_generated_label = "Generated Samples"
            # Create figure with two subplots as in the original function
            _w, _h = 3, 2
            _w, _h = (
                (rc_diag / (_w**2 + _h**2) ** 0.5) * _w,
                (rc_diag / (_w**2 + _h**2) ** 0.5) * _h,
            )
            fig = plt.figure(figsize=(_w, _h))
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], figure=fig)

            # Function to plot overlapping data
            def plot_overlapping_data(data, ax, title_text):
                data = _preprocess_data(data)

                # Limit number of samples to plot
                max_samples_to_plot = min(100, data.shape[0])
                data = data[:max_samples_to_plot]

                # Calculate daily consumption for coloring
                samples_daily_sum = data.sum(dim=1)
                # Sort by total consumption
                sample_idx = torch.argsort(samples_daily_sum)
                data = data[sample_idx]

                # Get statistics for coloring
                max_day_consumption = data.sum(dim=1).max().item()
                min_day_consumption = data.sum(dim=1).min().item()

                # Get data dimensions
                data_vec_dim = data.shape[1]

                # Plot each sample
                for index in range(max_samples_to_plot):
                    sample = data[index].cpu().numpy()
                    color = _get_color(
                        data[index], min_day_consumption, max_day_consumption
                    )

                    # Scatter plot
                    ax.scatter(
                        np.arange(data_vec_dim),
                        sample,
                        color=color,
                        s=0.5,
                        alpha=min(dot_alpha_coef / max_samples_to_plot, 1),
                        rasterized=True,
                    )

                    # Line plot
                    ax.plot(
                        np.arange(data_vec_dim),
                        sample,
                        linewidth=2.5,
                        color=color,
                        alpha=min(line_alpha_coef / max_samples_to_plot, 1),
                        rasterized=True,
                    )

                # Plot mask regions
                _plot_mask_regions(ax, data_vec_dim)

                # Set axis properties
                _set_axis_properties(ax, title_text, data_vec_dim)
                ax.legend()

            # Plot generated data
            ax_gen = fig.add_subplot(gs[0])
            if overlap_print_num_samples:
                overlap_generated_label += f" ({generated.shape[0]} "
                if generated.shape[0] > 1:
                    overlap_generated_label += "Samples)"
                else:
                    overlap_generated_label += "Sample)"
            plot_overlapping_data(
                generated,
                ax_gen,
                overlap_generated_label,
            )

            # Plot real data
            ax_real = fig.add_subplot(gs[1])
            _real_sample_title = "Real Samples"
            if overlap_print_num_samples:
                _real_sample_title += f" ({real.shape[0]} "
                if real.shape[0] > 1:
                    _real_sample_title += "Samples)"
                else:
                    _real_sample_title += "Sample)"
            plot_overlapping_data(
                real,
                ax_real,
                _real_sample_title,
            )

        elif layout == "grid":
            # For grid layout, show each sample in its own subplot

            # Check labels for each subplot
            if grid_generated_labels is not None and len(grid_generated_labels) != len(
                generated
            ):
                print(len(grid_generated_labels), len(generated), "uh huh")
                grid_generated_labels = None
            # Restrict real to just one sample if it has more
            if real.shape[0] > 1:
                real = real[:1]  # Take only the first sample
                logging.warning(
                    "Multiple real samples provided. Using only the first sample for grid layout."
                )

            # Preprocess data
            gen_data = _preprocess_data(generated)
            real_data = _preprocess_data(real)

            # Get data dimensions
            data_vec_dim = gen_data.shape[1]

            # Calculate statistics for consistent coloring across subplots
            all_data = torch.cat([gen_data, real_data], dim=0)
            _max_day_consumption = all_data.sum(dim=1).max().item()
            _min_day_consumption = all_data.sum(dim=1).min().item()

            # Determine the number of generated samples to show
            max_grid_size = 4
            max_plots = max_grid_size * max_grid_size - 1  # -1 for real sample
            num_gen_samples = min(max_plots, gen_data.shape[0])

            # Calculate grid dimensions
            total_plots = num_gen_samples + 1  # +1 for real sample
            grid_size = min(max_grid_size, int(np.ceil(np.sqrt(total_plots))))

            # Create figure
            _w, _h = 5, 4
            _w, _h = (
                (rc_diag / (_w**2 + _h**2) ** 0.5) * _w,
                (rc_diag / (_w**2 + _h**2) ** 0.5) * _h,
            )
            fig = plt.figure(figsize=(_w * grid_size, _h * grid_size))

            # Colors for grid layout
            REAL_COLOR = "blue"
            GENERATED_COLOR = "orange"

            # Function to plot a single sample
            def plot_single_sample(data, ax, title_text, is_real=False, idx=0):
                # Extract and plot the sample
                sample = data[idx].cpu().numpy()

                # Use simpler color scheme for grid layout
                color = REAL_COLOR if is_real else GENERATED_COLOR

                ax.plot(
                    np.arange(data_vec_dim),
                    sample,
                    linewidth=2.5,
                    color=color,
                    rasterized=True,
                )

                # Plot mask regions
                _plot_mask_regions(ax, data_vec_dim)

                # Set axis properties
                _set_axis_properties(ax, title_text, data_vec_dim)

            # Plot real sample in the first position
            ax_real = plt.subplot2grid((grid_size, grid_size), (0, 0))
            plot_single_sample(real_data, ax_real, "Real Sample", is_real=True)

            # Plot generated samples
            for i in range(num_gen_samples):
                # Calculate position in grid
                row = (i + 1) // grid_size
                col = (i + 1) % grid_size

                # Create subplot
                ax = plt.subplot2grid((grid_size, grid_size), (row, col))
                _title_text = (
                    f"Generated {i + 1}"
                    if grid_generated_labels is None
                    else grid_generated_labels[i]
                )
                plot_single_sample(gen_data, ax, _title_text, is_real=False, idx=i)

        else:
            raise ValueError(
                f"Unknown layout: {layout}. Choose from 'overlap' or 'grid'."
            )

        # Add overall title
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Save figure
        plt.tight_layout()
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")

            logging.info(f"Saved time series comparison plot to {output_path}")

        return fig


def plot_violinplot_mmds(
    metrics_data: dict,
    categories: list[str],
    dates: list[str],
    raw_dates: list[str] = None,
    save_filepath: str | None = None,
    figsize: tuple[float, float] = (15, 10),
):
    """Plot violin plots of MMD distributions over time for each category.

    Args:
        metrics_data: Nested dict with structure [category][raw_date] -> {'null_mmds': array, 'true_mmd': float}
        categories: List of category names
        dates: List of formatted date strings for display
        raw_dates: List of raw date strings used as keys in metrics_data (if None, uses dates)
        save_filepath: Optional path to save the figure
        figsize: Figure size tuple
    """
    if raw_dates is None:
        raw_dates = dates
    n_categories = len(categories)
    fig, axes = plt.subplots(n_categories, 1, figsize=figsize, sharex=True)

    if n_categories == 1:
        axes = [axes]

    # Convert dates to numeric positions for plotting
    date_positions = np.arange(len(dates))

    for i, category in enumerate(categories):
        ax = axes[i]

        # Prepare data for violin plots
        violin_data = []
        true_mmds = []
        p_values = []

        for raw_date in raw_dates:
            if raw_date in metrics_data[category]:
                null_mmds = metrics_data[category][raw_date]["null_mmds"]
                true_mmd = metrics_data[category][raw_date]["true_mmd"]
                p_value = metrics_data[category][raw_date]["p_value"]
                violin_data.append(null_mmds)
                true_mmds.append(true_mmd)
                p_values.append(p_value)
            else:
                violin_data.append([])
                true_mmds.append(np.nan)
                p_values.append(np.nan)

        # Create violin plots
        parts = ax.violinplot(
            violin_data,
            positions=date_positions,
            widths=0.6,
            showmeans=False,
            showmedians=False,
            quantiles=[[0.95]] * len(violin_data),
        )

        # Style violin plots
        for pc in parts["bodies"]:
            pc.set_facecolor("lightblue")
            pc.set_alpha(0.7)

        # Add true MMD line
        ax.plot(
            date_positions,
            true_mmds,
            "ro",
            linewidth=2,
            markersize=4,
            label="True MMD",
        )
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.3f}"))

        # Add p-value annotations
        for j, (pos, p_val) in enumerate(zip(date_positions, p_values, strict=True)):
            if not np.isnan(p_val):
                # Format p-value for display
                if p_val < 0.001:
                    p_text = "p<0.001"
                elif p_val < 0.01:
                    p_text = f"p={p_val:.3f}"
                else:
                    p_text = f"p={p_val:.2f}"

                # Position p-value text below the violin plot
                y_pos = min(violin_data[j]) if len(violin_data[j]) > 0 else 0
                ax.annotate(
                    p_text,
                    (pos, y_pos),
                    textcoords="offset points",
                    xytext=(0, -7.5),
                    ha="center",
                    fontsize=8,
                    color="darkblue",
                    weight="bold",
                )

        # Formatting
        ax.set_title(f"Category: {category}", fontweight="bold")
        ax.set_ylabel("MMD Value")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

        # Set y-axis to start from 0 for better comparison
        ax.set_ylim(bottom=0)

    # Set x-axis labels only on bottom plot
    # axes[-1].set_xlabel("Date")
    axes[-1].set_xticks(date_positions)
    axes[-1].set_xticklabels(dates, rotation=0)

    # plt.tight_layout()

    if save_filepath:
        os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
        plt.savefig(save_filepath, dpi=300, bbox_inches="tight")
        logging.info(f"Saved violin plot to {save_filepath}")

    return fig
