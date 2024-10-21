"""utils.py."""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pandas as pd
from numpyro.diagnostics import summary
from numpyro.infer import MCMC

from luxglm.dataclasses import LuxInputData

# ruff: noqa: FURB140


def read_count_files(
    names: list[str], filenames: list[str]
) -> pd.DataFrame:  # pragma: nocover
    """TBA.

    Args:
        names: Names of the samples.
        filenames: Names of the count files.

    Returns:
        Count dataframe.
    """

    def helper(name: str, filename: str) -> pd.DataFrame:
        count_df = pd.read_csv(filename, sep="\t").set_index(["chromosome", "position"])
        count_df.columns = pd.MultiIndex.from_product(
            ((name,), count_df.columns), names=["name", "count_type"]
        )
        return count_df

    return pd.concat(
        (
            helper(name, filename)
            for name, filename in zip(names, filenames, strict=True)
        ),
        axis=1,
    ).fillna(0)


def read_control_count_files(
    names: list[str], filenames: list[str]
) -> pd.DataFrame:  # pragma: nocover
    """TBA.

    Args:
        names: Names of the samples.
        filenames: Names of the count files.

    Returns:
        Count dataframe.
    """

    def helper(name: str, filename: str) -> pd.DataFrame:
        count_df = pd.read_csv(filename, sep="\t").set_index(
            ["chromosome", "position", "control_type"]
        )
        count_df.columns = pd.MultiIndex.from_product(
            ((name,), count_df.columns), names=["name", "count_type"]
        )
        return count_df

    return pd.concat(
        (
            helper(name, filename)
            for name, filename in zip(names, filenames, strict=True)
        ),
        axis=1,
    ).fillna(0)


def read_control_definitions(
    names: list[str], filenames: list[str]
) -> pd.DataFrame:  # pragma: nocover
    """TBA.

    Args:
        names: Names of the samples.
        filenames: Names of the count files.

    Returns:
        Control definition dataframe.
    """

    def helper(name: str, filename: str) -> pd.DataFrame:
        definition_df = pd.read_csv(filename, sep="\t").set_index(["control_type"])
        definition_df.columns = pd.MultiIndex.from_product(
            ((name,), definition_df.columns), names=["name", "modification"]
        )
        return definition_df

    return pd.concat(
        (
            helper(name, filename)
            for name, filename in zip(names, filenames, strict=True)
        ),
        axis=1,
    )


def get_input_data(metadata: str) -> pd.DataFrame:  # pragma: nocover
    """Get Lux input data.

    Args:
      metadata: Metadata filename.

    Returns:
        Input data stored in LuxInputData dataclass.
    """
    metadata_df = pd.read_csv(metadata, sep="\t")

    count_df = read_count_files(metadata_df.name, metadata_df.count_file)
    control_count_df = read_control_count_files(
        metadata_df.name, metadata_df.control_count_file
    )
    control_definition_df = read_control_definitions(
        metadata_df.name, metadata_df.control_definition_file
    )

    return LuxInputData(metadata_df, count_df, control_count_df, control_definition_df)


def get_mcmc_summary(mcmc: MCMC) -> pd.DataFrame:
    """Get MCMC summary DataFrame.

    Args:
        mcmc: MCMC object.

    Returns:
        MCMC summary stored in a dataframe.
    """

    def process_variable(
        variable: str, data: Mapping[str, npt.NDArray[np.float64] | np.float64]
    ) -> dict[str, Any]:
        res: dict[str, str | list[tuple[int, ...]] | list[np.float64] | list[None]] = {
            "variable": variable
        }
        for statistic, values in data.items():
            if "index" not in res:
                if isinstance(values, np.ndarray):
                    res["index"] = [
                        tuple(map(int, x))
                        for x in zip(
                            *jnp.unravel_index(jnp.arange(values.size), values.shape),
                            strict=True,
                        )
                    ]
                else:
                    res["index"] = [None]
            if isinstance(values, np.ndarray):
                res[statistic] = values.flatten().tolist()
            else:
                res[statistic] = [values]
        return res

    return pd.concat(
        [
            pd.DataFrame.from_dict(process_variable(variable, data))
            for variable, data in summary(mcmc.get_samples(group_by_chain=True)).items()
        ],
        axis=0,
        ignore_index=True,
    )
