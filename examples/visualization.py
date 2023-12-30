"""visualization.py."""

from math import ceil

import jax.numpy as jnp
import matplotlib.pyplot as plt
from luxglm.dataclasses import LuxResult
from matplotlib.figure import Figure


def plot_experimental_parameters(lux_result: LuxResult) -> Figure:
    """Visualize posterior distributions of experimental parameters.

    Args:
        lux_result: LuxResult object.

    Returns:
        Matplotlib figure object.
    """
    fig = plt.figure()

    fig.set_size_inches(6, 4)

    experimental_parameters_df = lux_result.experimental_parameters()

    for idx, parameter in enumerate(
        experimental_parameters_df.index.unique("parameter"), start=1
    ):
        ax = fig.add_subplot(2, 2, idx)

        for sample in experimental_parameters_df.index.unique("sample"):
            ax.hist(
                experimental_parameters_df.loc[(sample, parameter), :],
                20,
                density=True,
                alpha=0.4,
            )

        ax.set_xlabel(parameter)
        if (idx - 1) % 2 == 0:
            ax.set_ylabel("Posterior density")

    fig.tight_layout()

    return fig


def plot_methylation_levels(lux_result: LuxResult) -> Figure:
    """Visualize posterior distributions of methylation levels.

    Args:
        lux_result: LuxResult object.

    Returns:
        Matplotlib figure object.
    """
    fig = plt.figure()

    methylation_df = lux_result.methylation()

    cytosines = set(
        zip(
            methylation_df.index.get_level_values("chromosome"),
            methylation_df.index.get_level_values("position"),
        )
    )

    num_cytosines = len(cytosines)

    num_cols = 3
    num_rows = ceil(num_cytosines / num_cols)

    fig.set_size_inches(num_cols * 3, 2 * num_rows)

    for cytosine_idx, (chromosome, position) in enumerate(cytosines, start=1):
        ax = fig.add_subplot(num_rows, num_cols, cytosine_idx)

        ax.set_title(f"{chromosome}:{position}")

        methylation_subset_df = methylation_df.query(
            "chromosome == @chromosome and position == @position"
        )

        samples = methylation_subset_df.index.unique("sample")

        data = {
            "C": methylation_subset_df.query("modification == 'C'").mean(1).to_numpy(),
            "5mC": methylation_subset_df.query("modification == '5mC'")
            .mean(1)
            .to_numpy(),
            "5hmC": methylation_subset_df.query("modification == '5hmC'")
            .mean(1)
            .to_numpy(),
        }

        bottom = jnp.zeros(len(samples))
        for key, datum in data.items():
            ax.bar(
                jnp.arange(len(samples)),
                datum,
                label=key,
                bottom=bottom,
            )
            bottom += datum

        if cytosine_idx == 1:
            ax.legend()

        ax.set_xticks(jnp.arange(len(samples)))

        ax.set_xticklabels(samples, rotation=90, fontsize=5)

    fig.tight_layout()

    return fig


def plot_coefficients(lux_result: LuxResult) -> Figure:
    """Visualize posterior distributions of coefficients.

    Args:
        lux_result: LuxResult object.

    Returns:
        Matplotlib figure object.
    """
    fig = plt.figure()

    coefficients_df = lux_result.coefficients()

    cytosines = set(
        zip(
            coefficients_df.index.get_level_values("chromosome"),
            coefficients_df.index.get_level_values("position"),
        )
    )

    num_cytosines = len(cytosines)

    fig.set_size_inches(9, 2 * num_cytosines)

    for cytosine_idx, (chromosome, position) in enumerate(cytosines):
        ax = fig.add_subplot(num_cytosines, 3, cytosine_idx * 3 + 1)
        ax2 = fig.add_subplot(num_cytosines, 3, cytosine_idx * 3 + 2)
        ax3 = fig.add_subplot(num_cytosines, 3, cytosine_idx * 3 + 3)

        ax.set_title(f"C\n{chromosome}:{position}")
        ax2.set_title(f"5mC\n{chromosome}:{position}")
        ax3.set_title(f"5hmC\n{chromosome}:{position}")

        for covariate in coefficients_df.index.unique("covariate"):
            ax.hist(
                coefficients_df.loc[(covariate, chromosome, position, "C"), :],
                50,
                alpha=0.4,
                label=f"{covariate} 5C",
            )
            ax2.hist(
                coefficients_df.loc[(covariate, chromosome, position, "5mC"), :],
                50,
                alpha=0.4,
                label=f"{covariate} 5mC",
            )
            ax3.hist(
                coefficients_df.loc[(covariate, chromosome, position, "5hmC"), :],
                50,
                alpha=0.4,
                label=f"{covariate} 5hmC",
            )

        ax3.legend()

    fig.tight_layout()

    return fig
