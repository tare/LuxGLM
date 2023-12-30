"""models.py."""
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import Array
from jax.nn import sigmoid, softmax

# ruff: noqa: SIM117, PLR0913, PLR0914, PLR0915, PLR0917

NORMAL_APPROXIMATION_N = 500


def get_p_bs_c(
    theta: Array, bs_eff: Array, inaccurate_bs_eff: Array, seq_err: Array
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
    theta: Array,
    bs_eff: Array,
    inaccurate_bs_eff: Array,
    ox_eff: Array,
    seq_err: Array,
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


def get_experimental_parameters(num_samples: int) -> tuple[Array, Array, Array, Array]:
    """Generative model of experimental parameters.

    Args:
        num_samples: Number of samples.

    Returns:
        Following rate parameters: bs_eff, inaccurate_bs_eff, ox_eff, and seq_err.
    """
    mu_mu_bs_eff, sigma_mu_bs_eff = 2, 1.29
    mu_sigma_bs_eff, sigma_sigma_bs_eff = 0.4, 0.5

    mu_mu_inaccurate_bs_eff, sigma_mu_inaccurate_bs_eff = -3, 1.29
    mu_sigma_inaccurate_bs_eff, sigma_sigma_inaccurate_bs_eff = 0.4, 0.5

    mu_mu_ox_eff, sigma_mu_ox_eff = 2, 1.29
    mu_sigma_ox_eff, sigma_sigma_ox_eff = 0.4, 0.5

    mu_mu_seq_err, sigma_mu_seq_err = -3, 1.29
    mu_sigma_seq_err, sigma_sigma_seq_err = 0.4, 0.5

    mu_bs_eff = numpyro.sample("mu_bs_eff", dist.Normal(mu_mu_bs_eff, sigma_mu_bs_eff))
    sigma_bs_eff = numpyro.sample(
        "sigma_bs_eff", dist.LogNormal(mu_sigma_bs_eff, sigma_sigma_bs_eff)
    )

    mu_inaccurate_bs_eff = numpyro.sample(
        "mu_inaccurate_bs_eff",
        dist.Normal(mu_mu_inaccurate_bs_eff, sigma_mu_inaccurate_bs_eff),
    )
    sigma_inaccurate_bs_eff = numpyro.sample(
        "sigma_inaccurate_bs_eff",
        dist.LogNormal(mu_sigma_inaccurate_bs_eff, sigma_sigma_inaccurate_bs_eff),
    )

    mu_ox_eff = numpyro.sample("mu_ox_eff", dist.Normal(mu_mu_ox_eff, sigma_mu_ox_eff))
    sigma_ox_eff = numpyro.sample(
        "sigma_ox_eff", dist.LogNormal(mu_sigma_ox_eff, sigma_sigma_ox_eff)
    )

    mu_seq_err = numpyro.sample(
        "mu_seq_err", dist.Normal(mu_mu_seq_err, sigma_mu_seq_err)
    )
    sigma_seq_err = numpyro.sample(
        "sigma_seq_err", dist.LogNormal(mu_sigma_seq_err, sigma_sigma_seq_err)
    )

    with numpyro.plate("samples", num_samples, dim=-2):
        raw_bs_eff = numpyro.sample("raw_bs_eff", dist.Normal(0, 1))
        raw_inaccurate_bs_eff = numpyro.sample(
            "raw_inaccurate_bs_eff", dist.Normal(0, 1)
        )
        raw_ox_eff = numpyro.sample("raw_ox_eff", dist.Normal(0, 1))
        raw_seq_err = numpyro.sample("raw_seq_err", dist.Normal(0, 1))

        bs_eff = numpyro.deterministic(
            "bs_eff", sigmoid(mu_bs_eff + sigma_bs_eff * raw_bs_eff)
        )
        inaccurate_bs_eff = numpyro.deterministic(
            "inaccurate_bs_eff",
            sigmoid(
                mu_inaccurate_bs_eff + sigma_inaccurate_bs_eff * raw_inaccurate_bs_eff
            ),
        )
        ox_eff = numpyro.deterministic(
            "ox_eff", sigmoid(mu_ox_eff + sigma_ox_eff * raw_ox_eff)
        )
        seq_err = numpyro.deterministic(
            "seq_err", sigmoid(mu_seq_err + sigma_seq_err * raw_seq_err)
        )

    return bs_eff, inaccurate_bs_eff, ox_eff, seq_err


def bs_likelihood(
    bs_total: Array,
    bs_c: Array,
    bs_eff: Array,
    inaccurate_bs_eff: Array,
    seq_err: Array,
    theta: Array,
    use_normal_approximation: bool = False,
) -> None:
    """Bisulphite sequencing likelihood.

    Args:
        bs_total: Number of C or T read-outs.
        bs_c: Number of C read-outs.
        bs_eff: Bisulfite conversion rate parameter.
        inaccurate_bs_eff: Inaccurate bisulphite conversion rate parameter.
        seq_err: Sequencing error rate parameter.
        theta: Methylation level parameter.
        use_normal_approximation: Whether to use normal approximation of Binomial. Defaults to False.
    """
    if use_normal_approximation:
        with numpyro.handlers.mask(mask=bs_total <= NORMAL_APPROXIMATION_N):
            numpyro.sample(
                "bs_num_c",
                dist.Binomial(
                    bs_total,
                    get_p_bs_c(theta, bs_eff, inaccurate_bs_eff, seq_err),
                ),
                obs=bs_c,
            )
        with numpyro.handlers.mask(mask=bs_total > NORMAL_APPROXIMATION_N):
            numpyro.sample(
                "bs_num_c_normal",
                dist.Normal(
                    bs_total * get_p_bs_c(theta, bs_eff, inaccurate_bs_eff, seq_err),
                    jnp.sqrt(
                        bs_total
                        * get_p_bs_c(theta, bs_eff, inaccurate_bs_eff, seq_err)
                        * (1.0 - get_p_bs_c(theta, bs_eff, inaccurate_bs_eff, seq_err))
                    ),
                ),
                obs=bs_c,
            )
    else:
        numpyro.sample(
            "bs_num_c",
            dist.Binomial(
                bs_total,
                get_p_bs_c(theta, bs_eff, inaccurate_bs_eff, seq_err),
            ),
            obs=bs_c,
        )


def oxbs_likelihood(
    oxbs_total: Array,
    oxbs_c: Array,
    bs_eff: Array,
    inaccurate_bs_eff: Array,
    ox_eff: Array,
    seq_err: Array,
    theta: Array,
    use_normal_approximation: bool = False,
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
        use_normal_approximation: Whether to use normal approximation of Binomial. Defaults to False.
    """
    if use_normal_approximation:
        with numpyro.handlers.mask(mask=oxbs_total <= NORMAL_APPROXIMATION_N):
            numpyro.sample(
                "ox_num_c",
                dist.Binomial(
                    oxbs_total,
                    get_p_oxbs_c(theta, bs_eff, inaccurate_bs_eff, ox_eff, seq_err),
                ),
                obs=oxbs_c,
            )
        with numpyro.handlers.mask(mask=oxbs_total > NORMAL_APPROXIMATION_N):
            numpyro.sample(
                "ox_num_c_normal",
                dist.Normal(
                    oxbs_total
                    * get_p_oxbs_c(theta, bs_eff, inaccurate_bs_eff, ox_eff, seq_err),
                    jnp.sqrt(
                        oxbs_total
                        * get_p_oxbs_c(
                            theta, bs_eff, inaccurate_bs_eff, ox_eff, seq_err
                        )
                        * (
                            1.0
                            - get_p_oxbs_c(
                                theta, bs_eff, inaccurate_bs_eff, ox_eff, seq_err
                            )
                        ),
                    ),
                ),
                obs=oxbs_c,
            )
    else:
        numpyro.sample(
            "ox_num_c",
            dist.Binomial(
                oxbs_total,
                get_p_oxbs_c(theta, bs_eff, inaccurate_bs_eff, ox_eff, seq_err),
            ),
            obs=oxbs_c,
        )


def luxglm_bs_oxbs_model(
    design_matrix: Array,
    bs_c: Array,
    bs_total: Array,
    oxbs_c: Array,
    oxbs_total: Array,
    bs_c_control: Array,
    bs_total_control: Array,
    oxbs_c_control: Array,
    oxbs_total_control: Array,
    alpha_control: Array,
    use_normal_approximation: bool = False,
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
        use_normal_approximation: Whether to use normal approximation of Binomial. Defaults to False.
    """
    num_samples = bs_c.shape[-2]
    num_predictors = design_matrix.shape[-1]
    num_modifications = 3
    num_cytosines = bs_c.shape[-1]

    bs_eff, inaccurate_bs_eff, ox_eff, seq_err = get_experimental_parameters(
        num_samples
    )

    theta_control = numpyro.sample("theta_control", dist.Dirichlet(alpha_control))

    with numpyro.handlers.scope(prefix="control"):
        bs_likelihood(
            bs_total_control,
            bs_c_control,
            bs_eff,
            inaccurate_bs_eff,
            seq_err,
            theta_control,
            use_normal_approximation,
        )
        oxbs_likelihood(
            oxbs_total_control,
            oxbs_c_control,
            bs_eff,
            inaccurate_bs_eff,
            ox_eff,
            seq_err,
            theta_control,
            use_normal_approximation,
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
        use_normal_approximation,
    )
    oxbs_likelihood(
        oxbs_total,
        oxbs_c,
        bs_eff,
        inaccurate_bs_eff,
        ox_eff,
        seq_err,
        theta,
        use_normal_approximation,
    )


def luxglm_bs_oxbs_two_step_control_model(
    bs_c_control: Array,
    bs_total_control: Array,
    oxbs_c_control: Array,
    oxbs_total_control: Array,
    alpha_control: Array,
    use_normal_approximation: bool = False,
) -> None:
    """LuxGLM BS/oxBS model.

    This model is used to estimate the posterior distributions of experimental parameters.

    Args:
        bs_c_control: Number of C read-outs from BS-seq for control cytosines.
        bs_total_control: Number of total read-outs from BS-seq for control cytosines.
        oxbs_c_control: Number of C read-outs from oxBS-seq for control cytosines.
        oxbs_total_control: Number of total read-outs from oxBS-seq for control cytosines.
        alpha_control: Pseudocounts for priors of control cytosine.
        use_normal_approximation: Whether to use normal approximation of Binomial. Defaults to False.
    """
    num_samples = bs_c_control.shape[-2]

    bs_eff, inaccurate_bs_eff, ox_eff, seq_err = get_experimental_parameters(
        num_samples
    )

    theta_control = numpyro.sample("theta_control", dist.Dirichlet(alpha_control))

    with numpyro.handlers.scope(prefix="control"):
        bs_likelihood(
            bs_total_control,
            bs_c_control,
            bs_eff,
            inaccurate_bs_eff,
            seq_err,
            theta_control,
            use_normal_approximation,
        )
        oxbs_likelihood(
            oxbs_total_control,
            oxbs_c_control,
            bs_eff,
            inaccurate_bs_eff,
            ox_eff,
            seq_err,
            theta_control,
            use_normal_approximation,
        )


def luxglm_bs_oxbs_two_step_noncontrol_model(
    design_matrix: Array,
    bs_c: Array,
    bs_total: Array,
    oxbs_c: Array,
    oxbs_total: Array,
    use_normal_approximation: bool = False,
) -> None:
    """LuxGLM BS/oxBS model.

    This model requires that the posterior distributions of experimental parameters has been estimated beforehand.

    Args:
        design_matrix: Design matrix.
        bs_c: Number of C read-outs from BS-seq for non-control cytosines.
        bs_total: Number of total read-outs from BS-seq for non-control cytosines.
        oxbs_c: Number of C read-outs from oxBS-seq for non-control cytosines.
        oxbs_total: Number of total read-outs from oxBS-seq for non-control cytosines.
        use_normal_approximation: Whether to use normal approximation of Binomial. Defaults to False.
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
        use_normal_approximation,
    )
    oxbs_likelihood(
        oxbs_total,
        oxbs_c,
        bs_eff,
        inaccurate_bs_eff,
        ox_eff,
        seq_err,
        theta,
        use_normal_approximation,
    )
