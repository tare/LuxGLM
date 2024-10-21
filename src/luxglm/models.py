"""models.py."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import Array
from jax.nn import sigmoid, softmax
from jax.typing import ArrayLike

# ruff: noqa: SIM117, PLR0913, PLR0914, PLR0915, PLR0917


def get_p_bs_c(
    theta: ArrayLike,
    bs_eff: ArrayLike,
    inaccurate_bs_eff: ArrayLike,
    seq_err: ArrayLike,
) -> Array:
    """Calculate p_bs(C|theta, experimental parameters).

    Args:
        theta: Methylation level parameter.
        bs_eff: Bisulfite conversion rate parameter.
        inaccurate_bs_eff: Inaccurate bisulphite conversion rate parameter.
        seq_err: Sequencing error rate parameter.

    Returns:
        Probability of C.
    """
    return (
        theta[..., 0] * ((1.0 - seq_err) * (1.0 - bs_eff) + seq_err * bs_eff)
        + theta[..., 1]
        * ((1.0 - inaccurate_bs_eff) * (1.0 - seq_err) + seq_err * inaccurate_bs_eff)
        + theta[..., 2]
        * ((1.0 - inaccurate_bs_eff) * (1.0 - seq_err) + seq_err * inaccurate_bs_eff)
    )


def get_p_oxbs_c(
    theta: ArrayLike,
    bs_eff: ArrayLike,
    inaccurate_bs_eff: ArrayLike,
    ox_eff: ArrayLike,
    seq_err: ArrayLike,
) -> Array:
    """Calculate p_oxbs(C|theta, experimental parameters).

    Args:
        theta: Methylation level parameter.
        bs_eff: Bisulfite conversion rate parameter.
        inaccurate_bs_eff: Inaccurate bisulphite conversion rate parameter.
        ox_eff: Oxidation efficiency rate parameter.
        seq_err: Sequencing error rate parameter.

    Returns:
        Probability of C.
    """
    return (
        theta[..., 0] * ((1.0 - seq_err) * (1.0 - bs_eff) + seq_err * bs_eff)
        + theta[..., 1]
        * ((1.0 - inaccurate_bs_eff) * (1.0 - seq_err) + seq_err * inaccurate_bs_eff)
        + theta[..., 2]
        * (
            ox_eff * ((1.0 - seq_err) * (1.0 - bs_eff) + seq_err * bs_eff)
            + (1.0 - ox_eff)
            * (
                (1.0 - inaccurate_bs_eff) * (1.0 - seq_err)
                + inaccurate_bs_eff * seq_err
            )
        )
    )


def get_experimental_parameter(
    name: str,
    num_samples: int,
    mu_mu: float,
    sigma_mu: float,
    mu_sigma: float,
    sigma_sigma: float,
) -> Array:
    """Generative model of experimental parameter.

    mu ~ Normal(mu_mu, sigma_mu),
    sigma ~ LogNormal(mu_sigma, sigma_sigma),
    raw ~ Normal(0, 1),
    param = sigmoid(mu+sigma*raw)


    Args:
        name: Name of the experimental parameter.
        num_samples: Number of samples.
        mu_mu: Mean of the mean.
        sigma_mu: Scale of the mean.
        mu_sigma: Mean of the scale.
        sigma_sigma: Scale of the scale.

    Returns:
        Experimental parameter.
    """
    mu = numpyro.sample(f"mu_{name}", dist.Normal(mu_mu, sigma_mu))
    sigma = numpyro.sample(f"sigma_{name}", dist.LogNormal(mu_sigma, sigma_sigma))

    with numpyro.plate("samples", num_samples, dim=-2):
        raw = numpyro.sample(f"raw_{name}", dist.Normal(0, 1))
        param = numpyro.deterministic(name, sigmoid(mu + sigma * raw))

    return param  # noqa: RET504


def bs_likelihood(
    bs_total: ArrayLike,
    bs_c: ArrayLike,
    bs_eff: ArrayLike,
    inaccurate_bs_eff: ArrayLike,
    seq_err: ArrayLike,
    theta: ArrayLike,
) -> None:
    """Bisulphite sequencing likelihood.

    Args:
        bs_total: Number of C or T read-outs.
        bs_c: Number of C read-outs.
        bs_eff: Bisulfite conversion rate parameter.
        inaccurate_bs_eff: Inaccurate bisulphite conversion rate parameter.
        seq_err: Sequencing error rate parameter.
        theta: Methylation level parameter.
    """
    numpyro.sample(
        "bs_num_c",
        dist.Binomial(
            bs_total,
            get_p_bs_c(theta, bs_eff, inaccurate_bs_eff, seq_err),
        ),
        obs=bs_c,
    )


def oxbs_likelihood(
    oxbs_total: ArrayLike,
    oxbs_c: ArrayLike,
    bs_eff: ArrayLike,
    inaccurate_bs_eff: ArrayLike,
    ox_eff: ArrayLike,
    seq_err: ArrayLike,
    theta: ArrayLike,
) -> None:
    """Oxidative bisulphite sequencing likelihood.

    Args:
        oxbs_total: Number of C or T read-outs.
        oxbs_c: Number of C read-outs.
        bs_eff: Bisulfite conversion rate parameter.
        inaccurate_bs_eff: Inaccurate bisulphite conversion rate parameter.
        ox_eff: Oxidation efficiency rate parameter.
        seq_err: Sequencing error rate parameter.
        theta: Methylation level parameter.
    """
    numpyro.sample(
        "ox_num_c",
        dist.Binomial(
            oxbs_total,
            get_p_oxbs_c(theta, bs_eff, inaccurate_bs_eff, ox_eff, seq_err),
        ),
        obs=oxbs_c,
    )


def luxglm_bs_oxbs_model(
    design_matrix: ArrayLike,
    bs_c: ArrayLike,
    bs_total: ArrayLike,
    oxbs_c: ArrayLike,
    oxbs_total: ArrayLike,
    bs_c_control: ArrayLike,
    bs_total_control: ArrayLike,
    oxbs_c_control: ArrayLike,
    oxbs_total_control: ArrayLike,
    alpha_control: ArrayLike,
) -> None:
    """LuxGLM BS/oxBS model.

    Args:
        design_matrix: Design matrix.
        bs_c: Number of C read-outs from BS-seq for non-control cytosines.
        bs_total: Number of total read-outs from BS-seq for non-control cytosines.
        oxbs_c: Number of C read-outs from oxBS-seq for non-control cytosines.
        oxbs_total: Number of total read-outs from oxBS-seq for non-control cytosines.
        bs_c_control: Number of C read-outs from BS-seq for control cytosines.
        bs_total_control: Number of total read-outs from BS-seq for control cytosines.
        oxbs_c_control: Number of C read-outs from oxBS-seq for control cytosines.
        oxbs_total_control: Number of total read-outs from oxBS-seq for control cytosines.
        alpha_control: Pseudocounts for priors of control cytosine.
    """
    num_samples = bs_c.shape[-2]
    num_predictors = design_matrix.shape[-1]
    num_modifications = 3
    num_cytosines = bs_c.shape[-1]

    bs_eff = get_experimental_parameter("bs_eff", num_samples, 2, 1.29, 0.4, 0.5)
    inaccurate_bs_eff = get_experimental_parameter(
        "inaccurate_bs_eff", num_samples, -3, 1.29, 0.4, 0.5
    )
    ox_eff = get_experimental_parameter("ox_eff", num_samples, 2, 1.29, 0.4, 0.5)
    seq_err = get_experimental_parameter("seq_err", num_samples, -3, 1.29, 0.4, 0.5)

    theta_control = numpyro.sample("theta_control", dist.Dirichlet(alpha_control))

    with numpyro.handlers.scope(prefix="control"):
        bs_likelihood(
            bs_total_control,
            bs_c_control,
            bs_eff,
            inaccurate_bs_eff,
            seq_err,
            theta_control,
        )
        oxbs_likelihood(
            oxbs_total_control,
            oxbs_c_control,
            bs_eff,
            inaccurate_bs_eff,
            ox_eff,
            seq_err,
            theta_control,
        )

    sigma_epsilon = numpyro.sample("sigma_y", dist.HalfNormal(1))

    b = numpyro.sample(
        "b",
        dist.Normal(0, 1).expand(
            (num_predictors, num_cytosines, num_modifications - 1)
        ),
    )

    epsilon_raw = numpyro.sample(
        "epsilon_raw",
        dist.Normal(0, 1).expand((num_samples, num_cytosines, num_modifications - 1)),
    )

    y = jnp.einsum("ik,klm->ilm", design_matrix, b) + sigma_epsilon * epsilon_raw
    theta = numpyro.deterministic(
        "theta",
        softmax(
            jnp.insert(y, 0, jnp.zeros(y.shape[:-1]), axis=-1),
            axis=-1,
        ),
    )

    bs_likelihood(
        bs_total,
        bs_c,
        bs_eff,
        inaccurate_bs_eff,
        seq_err,
        theta,
    )
    oxbs_likelihood(
        oxbs_total,
        oxbs_c,
        bs_eff,
        inaccurate_bs_eff,
        ox_eff,
        seq_err,
        theta,
    )


def luxglm_bs_oxbs_two_step_control_model(
    bs_c_control: ArrayLike,
    bs_total_control: ArrayLike,
    oxbs_c_control: ArrayLike,
    oxbs_total_control: ArrayLike,
    alpha_control: ArrayLike,
) -> None:
    """LuxGLM BS/oxBS model.

    This model is used to estimate the posterior distributions of experimental parameters.

    Args:
        bs_c_control: Number of C read-outs from BS-seq for control cytosines.
        bs_total_control: Number of total read-outs from BS-seq for control cytosines.
        oxbs_c_control: Number of C read-outs from oxBS-seq for control cytosines.
        oxbs_total_control: Number of total read-outs from oxBS-seq for control cytosines.
        alpha_control: Pseudocounts for priors of control cytosine.
    """
    num_samples = bs_c_control.shape[-2]

    bs_eff = get_experimental_parameter("bs_eff", num_samples, 2, 1.29, 0.4, 0.5)
    inaccurate_bs_eff = get_experimental_parameter(
        "inaccurate_bs_eff", num_samples, -3, 1.29, 0.4, 0.5
    )
    ox_eff = get_experimental_parameter("ox_eff", num_samples, 2, 1.29, 0.4, 0.5)
    seq_err = get_experimental_parameter("seq_err", num_samples, -3, 1.29, 0.4, 0.5)

    theta_control = numpyro.sample("theta_control", dist.Dirichlet(alpha_control))

    with numpyro.handlers.scope(prefix="control"):
        bs_likelihood(
            bs_total_control,
            bs_c_control,
            bs_eff,
            inaccurate_bs_eff,
            seq_err,
            theta_control,
        )
        oxbs_likelihood(
            oxbs_total_control,
            oxbs_c_control,
            bs_eff,
            inaccurate_bs_eff,
            ox_eff,
            seq_err,
            theta_control,
        )


def luxglm_bs_oxbs_two_step_noncontrol_model(
    design_matrix: ArrayLike,
    bs_c: ArrayLike,
    bs_total: ArrayLike,
    oxbs_c: ArrayLike,
    oxbs_total: ArrayLike,
) -> None:
    """LuxGLM BS/oxBS model.

    This model requires that the posterior distributions of experimental parameters has been estimated beforehand.

    Args:
        design_matrix: Design matrix.
        bs_c: Number of C read-outs from BS-seq for non-control cytosines.
        bs_total: Number of total read-outs from BS-seq for non-control cytosines.
        oxbs_c: Number of C read-outs from oxBS-seq for non-control cytosines.
        oxbs_total: Number of total read-outs from oxBS-seq for non-control cytosines.
    """
    num_samples = bs_c.shape[-2]
    num_predictors = design_matrix.shape[-1]
    num_modifications = 3
    num_cytosines = bs_c.shape[-1]

    with numpyro.plate("samples", num_samples, dim=-2):
        bs_eff = numpyro.sample(
            "bs_eff",
            dist.Normal(0, 1),
        )
        inaccurate_bs_eff = numpyro.sample(
            "inaccurate_bs_eff",
            dist.Normal(0, 1),
        )
        ox_eff = numpyro.sample(
            "ox_eff",
            dist.Normal(0, 1),
        )
        seq_err = numpyro.sample(
            "seq_err",
            dist.Normal(0, 1),
        )

    sigma_epsilon = numpyro.sample("sigma_y", dist.HalfNormal(1))

    b = numpyro.sample(
        "b",
        dist.Normal(0, 1).expand(
            (num_predictors, num_cytosines, num_modifications - 1)
        ),
    )

    epsilon_raw = numpyro.sample(
        "epsilon_raw",
        dist.Normal(0, 1).expand((num_samples, num_cytosines, num_modifications - 1)),
    )

    y = jnp.einsum("ik,klm->ilm", design_matrix, b) + sigma_epsilon * epsilon_raw
    theta = numpyro.deterministic(
        "theta",
        softmax(
            jnp.insert(y, 0, jnp.zeros(y.shape[:-1]), axis=-1),
            axis=-1,
        ),
    )

    bs_likelihood(
        bs_total,
        bs_c,
        bs_eff,
        inaccurate_bs_eff,
        seq_err,
        theta,
    )
    oxbs_likelihood(
        oxbs_total,
        oxbs_c,
        bs_eff,
        inaccurate_bs_eff,
        ox_eff,
        seq_err,
        theta,
    )
