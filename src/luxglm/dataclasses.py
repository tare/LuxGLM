"""dataclasses.py."""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pandas as pd
from jax import Array


@dataclass
class LuxModelData:
    """Model data.

    Args:
        bs_c: Number of C read-outs from BS-seq for non-control cytosines.
        bs_total: Number of total read-outs from BS-seq for non-control cytosines.
        oxbs_c: Number of C read-outs from oxBS-seq for non-control cytosines.
        oxbs_total: Number of total read-outs from oxBS-seq for non-control cytosines.
        bs_c_control: Number of C read-outs from BS-seq for control cytosines.
        bs_total_control: Number of total read-outs from BS-seq for control cytosines.
        oxbs_c_control: Number of C read-outs from oxBS-seq for control cytosines.
        oxbs_total_control: Number of total read-outs from oxBS-seq for control cytosines.
        alpha_control: Pseudocounts for priors of control cytosine.
        design_matrix: Design matrix.
        covariates: List of covariates of interest.
    """

    bs_c: npt.NDArray[np.int_]
    bs_total: npt.NDArray[np.int_]
    oxbs_c: npt.NDArray[np.int_]
    oxbs_total: npt.NDArray[np.int_]
    bs_c_control: npt.NDArray[np.int_]
    bs_total_control: npt.NDArray[np.int_]
    oxbs_c_control: npt.NDArray[np.int_]
    oxbs_total_control: npt.NDArray[np.int_]
    alpha_control: npt.NDArray[np.float64]
    design_matrix: npt.NDArray[np.float64]
    covariates: npt.NDArray[np.str_]


@dataclass
class LuxInputData:
    """Model data.

    Args:
        metadata_df: Metadata dataframe.
        count_df: Non-control counts dataframe.
        control_count_df: Control counts dataframe.
        control_definition_df: Control definition dataframe.
    """

    metadata_df: pd.DataFrame
    count_df: pd.DataFrame
    control_count_df: pd.DataFrame
    control_definition_df: pd.DataFrame

    def get_data(self, covariates: list[str]) -> LuxModelData:
        """Get model data.

        Args:
            covariates: List of covariates of interest.

        Returns:
            LuxModelData object.
        """

        def helper(name: str) -> LuxModelData:
            bs_c = self.count_df[name].bs_c
            bs_total = self.count_df[name].bs_total
            oxbs_c = self.count_df[name].oxbs_c
            oxbs_total = self.count_df[name].oxbs_total

            bs_c_control = self.control_count_df[name].bs_c
            bs_total_control = self.control_count_df[name].bs_total
            oxbs_c_control = self.control_count_df[name].oxbs_c
            oxbs_total_control = self.control_count_df[name].oxbs_total

            mapping = self.control_definition_df[name].apply(list, axis=1).to_dict()
            alpha_control = np.asarray(
                (
                    self.control_count_df[name]
                    .index.get_level_values("control_type")
                    .map(mapping)
                    .to_list()
                )
            )
            design_matrix = (
                self.metadata_df.query("name == @name").iloc[0][covariates].to_numpy()
            )

            return LuxModelData(
                bs_c,
                bs_total,
                oxbs_c,
                oxbs_total,
                bs_c_control,
                bs_total_control,
                oxbs_c_control,
                oxbs_total_control,
                alpha_control,
                design_matrix,
                np.asarray(covariates),
            )

        data = [helper(name) for name in self.metadata_df.name]

        return LuxModelData(
            np.stack([datum.bs_c for datum in data]),
            np.stack([datum.bs_total for datum in data]),
            np.stack([datum.oxbs_c for datum in data]),
            np.stack([datum.oxbs_total for datum in data]),
            np.stack([datum.bs_c_control for datum in data]),
            np.stack([datum.bs_total_control for datum in data]),
            np.stack([datum.oxbs_c_control for datum in data]),
            np.stack([datum.oxbs_total_control for datum in data]),
            np.stack([datum.alpha_control for datum in data]),
            np.stack([datum.design_matrix for datum in data]),
            data[0].covariates,
        )


@dataclass
class LuxResult:
    """Model results.

    Args:
        samples: List of sample names.
        covariates: List of covariates of interest.
        inference_metrics: Metrics related to inference.
        positions: Position information of non-control cytosines.
        control_positions: Position information of control cytosines.
        posterior_samples: Posterior samples.
    """

    samples: list[str]
    covariates: list[str]
    inference_metrics: dict[str, Any]
    positions: list[tuple[str, int]]
    control_positions: list[tuple[str, int]]
    posterior_samples: dict[str, Array]

    def methylation(self) -> pd.DataFrame:
        """Posterior samples of methylation levels of non-control cytosines.

        Returns:
            Posterior samples of methylation levels of non-control cytosines in a dataframe.
        """
        (
            num_posterior_samples,
            num_samples,
            num_cytosines,
            num_modifications,
        ) = self.posterior_samples["theta"].shape

        return pd.DataFrame(
            np.reshape(
                jnp.moveaxis(self.posterior_samples["theta"], 0, -1),
                (
                    num_samples * num_cytosines * num_modifications,
                    num_posterior_samples,
                ),
            ),
            index=pd.MultiIndex.from_tuples(
                (
                    (x[0], x[1][0], x[1][1], x[2])
                    for x in pd.MultiIndex.from_product(
                        (self.samples, self.positions, ("C", "5mC", "5hmC"))
                    )
                ),
                names=["sample", "chromosome", "position", "modification"],
            ),
        )

    def methylation_controls(self) -> pd.DataFrame:
        """Posterior samples of methylation levels of control cytosines.

        Returns:
            Posterior samples of methylation levels of control cytosines in a dataframe.
        """
        (
            num_posterior_samples,
            num_samples,
            num_cytosines,
            num_modifications,
        ) = self.posterior_samples["theta_control"].shape

        return pd.DataFrame(
            np.reshape(
                jnp.moveaxis(self.posterior_samples["theta_control"], 0, -1),
                (
                    num_samples * num_cytosines * num_modifications,
                    num_posterior_samples,
                ),
            ),
            index=pd.MultiIndex.from_tuples(
                (
                    (x[0], x[1][0], x[1][1], x[1][2], x[2])
                    for x in pd.MultiIndex.from_product(
                        (self.samples, self.control_positions, ("C", "5mC", "5hmC"))
                    )
                ),
                names=[
                    "sample",
                    "chromosome",
                    "position",
                    "control_type",
                    "modification",
                ],
            ),
        )

    def experimental_parameters(self) -> pd.DataFrame:
        """Posterior samples of experimental parameters.

        Returns:
            Posterior samples of experimental parameters in a dataframe.
        """
        num_posterior_samples, num_samples, _ = self.posterior_samples["bs_eff"].shape
        experimental_parameters = ("bs_eff", "inaccurate_bs_eff", "ox_eff", "seq_err")
        num_experimental_parameters = len(experimental_parameters)

        return pd.DataFrame(
            jnp.reshape(
                jnp.stack(
                    (
                        jnp.moveaxis(self.posterior_samples["bs_eff"], 0, 2),
                        jnp.moveaxis(self.posterior_samples["inaccurate_bs_eff"], 0, 2),
                        jnp.moveaxis(self.posterior_samples["ox_eff"], 0, 2),
                        jnp.moveaxis(self.posterior_samples["seq_err"], 0, 2),
                    ),
                    axis=1,
                ),
                (num_samples * num_experimental_parameters, num_posterior_samples),
            ),
            index=pd.MultiIndex.from_product(
                (self.samples, experimental_parameters),
                names=["sample", "parameter"],
            ),
        )

    def coefficients(self) -> pd.DataFrame:
        """Posterior samples of coefficients.

        Returns:
            Posterior samples of coefficients in a dataframe.
        """
        (
            num_posterior_samples,
            num_covariates,
            num_cytosines,
            num_modifications,
        ) = self.posterior_samples["b"].shape

        return pd.DataFrame(
            jnp.reshape(
                jnp.moveaxis(
                    jnp.insert(self.posterior_samples["b"], 0, 0.0, axis=3), 0, 3
                ),
                (
                    num_covariates * num_cytosines * (num_modifications + 1),
                    num_posterior_samples,
                ),
            ),
            index=pd.MultiIndex.from_tuples(
                (
                    (x[0], x[1][0], x[1][1], x[2])
                    for x in pd.MultiIndex.from_product(
                        (self.covariates, self.positions, ("C", "5mC", "5hmC"))
                    )
                ),
                names=["covariate", "chromosome", "position", "modification"],
            ),
        )
