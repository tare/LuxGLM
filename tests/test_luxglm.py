"""test_luxglm.py."""
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import pytest
from jax import random
from luxglm.dataclasses import LuxInputData
from luxglm.inference import run_nuts, run_svi
from luxglm.utils import get_mcmc_summary
from numpyro.infer import MCMC, NUTS

numpyro.enable_x64()

LOW_EXPERIMENTAL_PARAMETER_VALUE = 0.1
HIGH_EXPERIMENTAL_PARAMETER_VALUE = 0.9


@pytest.fixture()
def luxinputdata() -> LuxInputData:
    """LuxInputData."""
    count_files = ["count_file_1.tsv", "count_file_2.tsv"]
    control_count_files = ["control_count_file_1.tsv", "control_count_file_2.tsv"]
    control_definition_files = ["control_definitions.tsv", "control_definitions.tsv"]

    metadata_df = pd.DataFrame(
        {
            "name": ["sample_1", "sample_2"],
            "basal": [1, 1],
            "dko": [0, 1],
            "count_file": count_files,
            "control_count_file": control_count_files,
            "control_definition_file": control_definition_files,
        }
    )
    count_dfs = [
        pd.DataFrame(
            {
                "chromosome": ["chr1", "chr2", "chr3"],
                "position": [1, 1, 1],
                "bs_c": [0, 50, 50],
                "bs_total": [50, 50, 50],
                "oxbs_c": [0, 50, 50],
                "oxbs_total": [50, 50, 50],
            },
        ).set_index(["chromosome", "position"]),
        pd.DataFrame(
            {
                "chromosome": ["chr1", "chr2", "chr3", "chr3"],
                "position": [1, 1, 1, 2],
                "bs_c": [50, 0, 50, 0],
                "bs_total": [50, 50, 50, 50],
                "oxbs_c": [50, 0, 0, 0],
                "oxbs_total": [50, 50, 50, 50],
            },
        ).set_index(["chromosome", "position"]),
    ]

    for name, count_df in zip(metadata_df.name, count_dfs):
        count_df.columns = pd.MultiIndex.from_product(
            [[name], count_df.columns], names=["name", "count_type"]
        )

    count_df = pd.concat(count_dfs, copy=False, axis=1, sort=True).fillna(0)

    control_count_dfs = [
        pd.DataFrame(
            {
                "chromosome": ["lambda", "lambda", "lambda"],
                "position": [1, 2, 3],
                "control_type": ["C", "5mC", "5hmC"],
                "bs_c": [0, 50, 50],
                "bs_total": [50, 50, 50],
                "oxbs_c": [0, 50, 0],
                "oxbs_total": [50, 50, 50],
            },
        ).set_index(["chromosome", "position", "control_type"]),
        pd.DataFrame(
            {
                "chromosome": ["lambda", "lambda", "lambda"],
                "position": [1, 2, 3],
                "control_type": ["C", "5mC", "5hmC"],
                "bs_c": [0, 50, 50],
                "bs_total": [50, 50, 50],
                "oxbs_c": [0, 50, 0],
                "oxbs_total": [50, 50, 50],
            },
        ).set_index(["chromosome", "position", "control_type"]),
    ]

    for name, control_count_df in zip(metadata_df.name, control_count_dfs):
        control_count_df.columns = pd.MultiIndex.from_product(
            [[name], control_count_df.columns], names=["name", "count_type"]
        )

    control_count_df = pd.concat(
        control_count_dfs, copy=False, axis=1, sort=True
    ).fillna(0)

    control_definition_dfs = [
        pd.DataFrame(
            {
                "control_type": ["C", "5mC", "5hmC"],
                "C_pseudocount": [98, 1, 1],
                "5mC_pseudocount": [1, 98, 1],
                "5hmC_pseudocount": [1, 1, 98],
            },
        ).set_index("control_type"),
        pd.DataFrame(
            {
                "control_type": ["C", "5mC", "5hmC"],
                "C_pseudocount": [98, 1, 1],
                "5mC_pseudocount": [1, 98, 1],
                "5hmC_pseudocount": [1, 1, 98],
            },
        ).set_index("control_type"),
    ]

    for name, control_definition_df in zip(metadata_df.name, control_definition_dfs):
        control_definition_df.columns = pd.MultiIndex.from_product(
            [[name], control_definition_df.columns], names=["name", "count_type"]
        )

    control_definition_df = pd.concat(
        control_definition_dfs, copy=False, axis=1, sort=True
    )

    return LuxInputData(metadata_df, count_df, control_count_df, control_definition_df)


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (["basal", "dko"], np.asarray([[1, 0], [1, 1]])),
        (["basal"], np.asarray([[1], [1]])),
    ],
)
def test_get_data_covariates(
    luxinputdata: LuxInputData, test_input: list[str], expected: np.ndarray
) -> None:
    """Test get_data()."""
    lux_input_data = luxinputdata.get_data(test_input)
    assert np.all(lux_input_data.design_matrix == expected)
    assert np.all(lux_input_data.covariates == np.asarray(test_input))


@pytest.mark.parametrize("two_steps_inference", [False, True])
@pytest.mark.parametrize(
    ("covariates", "expected"),
    [
        (
            ["basal", "dko"],
            {
                ("sample_1", "chr1", 1): {
                    "C": HIGH_EXPERIMENTAL_PARAMETER_VALUE,
                    "5mC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5hmC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                },
                ("sample_2", "chr1", 1): {
                    "C": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5mC": HIGH_EXPERIMENTAL_PARAMETER_VALUE,
                    "5hmC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                },
                ("sample_1", "chr2", 1): {
                    "C": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5mC": HIGH_EXPERIMENTAL_PARAMETER_VALUE,
                    "5hmC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                },
                ("sample_2", "chr2", 1): {
                    "C": HIGH_EXPERIMENTAL_PARAMETER_VALUE,
                    "5mC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5hmC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                },
                ("sample_1", "chr3", 1): {
                    "C": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5mC": HIGH_EXPERIMENTAL_PARAMETER_VALUE,
                    "5hmC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                },
                ("sample_2", "chr3", 1): {
                    "C": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5mC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5hmC": HIGH_EXPERIMENTAL_PARAMETER_VALUE,
                },
            },
        ),
        (
            ["basal"],
            {
                ("sample_1", "chr1", 1): {
                    "C": HIGH_EXPERIMENTAL_PARAMETER_VALUE,
                    "5mC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5hmC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                },
                ("sample_2", "chr1", 1): {
                    "C": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5mC": HIGH_EXPERIMENTAL_PARAMETER_VALUE,
                    "5hmC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                },
                ("sample_1", "chr2", 1): {
                    "C": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5mC": HIGH_EXPERIMENTAL_PARAMETER_VALUE,
                    "5hmC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                },
                ("sample_2", "chr2", 1): {
                    "C": HIGH_EXPERIMENTAL_PARAMETER_VALUE,
                    "5mC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5hmC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                },
                ("sample_1", "chr3", 1): {
                    "C": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5mC": HIGH_EXPERIMENTAL_PARAMETER_VALUE,
                    "5hmC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                },
                ("sample_2", "chr3", 1): {
                    "C": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5mC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5hmC": HIGH_EXPERIMENTAL_PARAMETER_VALUE,
                },
            },
        ),
    ],
)
def test_run_nuts(
    luxinputdata: LuxInputData,
    two_steps_inference: bool,
    covariates: list[str],
    expected: dict[tuple[str, str, int], dict[str, float]],
) -> None:
    """Test run_nuts()."""
    key = random.PRNGKey(0)
    lux_result = run_nuts(
        key,
        luxinputdata,
        covariates,
        num_warmup=100,
        num_samples=100,
        num_chains=4,
        two_steps_inference=two_steps_inference,
    )

    # experimental parameters
    experimental_parameters_df = lux_result.experimental_parameters()
    assert np.all(
        experimental_parameters_df.query("parameter == 'bs_eff'").mean(1)
        > HIGH_EXPERIMENTAL_PARAMETER_VALUE
    )
    assert np.all(
        experimental_parameters_df.query("parameter == 'inaccurate_bs_eff'").mean(1)
        < LOW_EXPERIMENTAL_PARAMETER_VALUE
    )
    assert np.all(
        experimental_parameters_df.query("parameter == 'ox_eff'").mean(1)
        > HIGH_EXPERIMENTAL_PARAMETER_VALUE
    )
    assert np.all(
        experimental_parameters_df.query("parameter == 'seq_err'").mean(1)
        < LOW_EXPERIMENTAL_PARAMETER_VALUE
    )

    # methylation
    methylation_df = lux_result.methylation()
    for (sample, chromosome, position), modifications in expected.items():  # noqa: PERF102, B007
        methylation_subset_df = methylation_df.query(
            "sample == @sample and chromosome == @chromosome and position == @position"
        )
        for modification, value in modifications.items():  # noqa: PERF102, B007
            assert np.all(
                methylation_subset_df.query("modification == @modification").mean(1)
                > value
                if value == HIGH_EXPERIMENTAL_PARAMETER_VALUE
                else methylation_subset_df.query("modification == @modification").mean(
                    1
                )
                < value
            )


@pytest.mark.parametrize(
    ("covariates", "expected"),
    [
        (
            ["basal", "dko"],
            {
                ("sample_1", "chr1", 1): {
                    "C": HIGH_EXPERIMENTAL_PARAMETER_VALUE,
                    "5mC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5hmC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                },
                ("sample_2", "chr1", 1): {
                    "C": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5mC": HIGH_EXPERIMENTAL_PARAMETER_VALUE,
                    "5hmC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                },
                ("sample_1", "chr2", 1): {
                    "C": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5mC": HIGH_EXPERIMENTAL_PARAMETER_VALUE,
                    "5hmC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                },
                ("sample_2", "chr2", 1): {
                    "C": HIGH_EXPERIMENTAL_PARAMETER_VALUE,
                    "5mC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5hmC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                },
                ("sample_1", "chr3", 1): {
                    "C": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5mC": HIGH_EXPERIMENTAL_PARAMETER_VALUE,
                    "5hmC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                },
                ("sample_2", "chr3", 1): {
                    "C": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5mC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5hmC": HIGH_EXPERIMENTAL_PARAMETER_VALUE,
                },
            },
        ),
        (
            ["basal"],
            {
                ("sample_1", "chr1", 1): {
                    "C": HIGH_EXPERIMENTAL_PARAMETER_VALUE,
                    "5mC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5hmC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                },
                ("sample_2", "chr1", 1): {
                    "C": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5mC": HIGH_EXPERIMENTAL_PARAMETER_VALUE,
                    "5hmC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                },
                ("sample_1", "chr2", 1): {
                    "C": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5mC": HIGH_EXPERIMENTAL_PARAMETER_VALUE,
                    "5hmC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                },
                ("sample_2", "chr2", 1): {
                    "C": HIGH_EXPERIMENTAL_PARAMETER_VALUE,
                    "5mC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5hmC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                },
                ("sample_1", "chr3", 1): {
                    "C": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5mC": HIGH_EXPERIMENTAL_PARAMETER_VALUE,
                    "5hmC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                },
                ("sample_2", "chr3", 1): {
                    "C": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5mC": LOW_EXPERIMENTAL_PARAMETER_VALUE,
                    "5hmC": HIGH_EXPERIMENTAL_PARAMETER_VALUE,
                },
            },
        ),
    ],
)
def test_run_svi(
    luxinputdata: LuxInputData,
    covariates: list[str],
    expected: dict[tuple[str, str, int], dict[str, float]],
) -> None:
    """Test run_svi()."""
    key = random.PRNGKey(1)
    lux_result = run_svi(
        key,
        luxinputdata,
        covariates,
        num_steps=5_000,
        num_samples=100,
    )

    # experimental parameters
    experimental_parameters_df = lux_result.experimental_parameters()
    assert np.all(
        experimental_parameters_df.query("parameter == 'bs_eff'").mean(1)
        > HIGH_EXPERIMENTAL_PARAMETER_VALUE
    )
    assert np.all(
        experimental_parameters_df.query("parameter == 'inaccurate_bs_eff'").mean(1)
        < LOW_EXPERIMENTAL_PARAMETER_VALUE
    )
    assert np.all(
        experimental_parameters_df.query("parameter == 'ox_eff'").mean(1)
        > HIGH_EXPERIMENTAL_PARAMETER_VALUE
    )
    assert np.all(
        experimental_parameters_df.query("parameter == 'seq_err'").mean(1)
        < LOW_EXPERIMENTAL_PARAMETER_VALUE
    )

    # coefficients
    lux_result.coefficients()

    # control methylation
    lux_result.methylation_controls()

    # methylation
    methylation_df = lux_result.methylation()
    for (sample, chromosome, position), modifications in expected.items():  # noqa: PERF102, B007
        methylation_subset_df = methylation_df.query(
            "sample == @sample and chromosome == @chromosome and position == @position"
        )
        for modification, value in modifications.items():  # noqa: PERF102, B007
            assert np.all(
                methylation_subset_df.query("modification == @modification").mean(1)
                > value
                if value == HIGH_EXPERIMENTAL_PARAMETER_VALUE
                else methylation_subset_df.query("modification == @modification").mean(
                    1
                )
                < value
            )


def test_get_mcmc_summary() -> None:
    """Test get_mcmc_summary()."""
    num_obs = 5

    def model() -> None:
        mu = numpyro.sample("mu", dist.Normal(0, 1))
        with numpyro.plate("N", num_obs):
            numpyro.sample("y", dist.Normal(mu, 1))

    nuts_kernel = NUTS(model)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=500,
        num_samples=500,
        num_chains=4,
    )
    key = random.PRNGKey(0)
    key, key_ = random.split(key, 2)
    mcmc.run(key_)

    summary = get_mcmc_summary(mcmc)

    assert tuple(summary.columns) == (
        "variable",
        "index",
        "mean",
        "std",
        "median",
        "5.0%",
        "95.0%",
        "n_eff",
        "r_hat",
    )
    assert summary.shape[0] == 1 + num_obs
    assert summary.query("variable == 'mu'").shape[0] == 1
    assert summary.query("variable == 'y'").shape[0] == num_obs
    assert summary.query("r_hat > 1.05").shape[0] == 0
