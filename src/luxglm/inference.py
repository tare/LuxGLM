"""inference.py."""

from collections.abc import Callable
from operator import itemgetter

from jax import Array, random
from jax.tree_util import tree_map
from jax.typing import ArrayLike
from numpyro.infer import ELBO, MCMC, NUTS, SVI, HMCGibbs, Trace_ELBO
from numpyro.infer.autoguide import AutoGuide, AutoNormal
from numpyro.infer.initialization import init_to_mean, init_to_sample
from numpyro.optim import Adam, _NumPyroOptim

from luxglm.dataclasses import LuxInputData, LuxResult
from luxglm.models import (
    luxglm_bs_oxbs_model,
    luxglm_bs_oxbs_two_step_control_model,
    luxglm_bs_oxbs_two_step_noncontrol_model,
)
from luxglm.utils import get_mcmc_summary

# ruff: noqa: PLR0914


def get_gibbs_fn(
    posterior_samples: dict[str, Array],
) -> Callable[[Array, list[str], list[str]], dict[str, ArrayLike]]:
    """Get gibbs_fn() to be used with HMCGibbs().

    This can be used for introducing unlearnable distributions in the model.

    Args:
        posterior_samples: Posterior samples of the parameters of interest.

    Returns:
        Function that returns a sample from the set of posterior samples.
    """
    num_samples = posterior_samples[next(iter(posterior_samples.keys()))].shape[0]

    def gibbs_fn(
        rng_key: Array,
        gibbs_sites: list[str],  # noqa: ARG001
        hmc_sites: list[str],  # noqa: ARG001
    ) -> dict[str, Array]:
        idx = random.randint(rng_key, (), 0, num_samples)
        return tree_map(itemgetter(idx), posterior_samples)

    return gibbs_fn


def run_nuts(
    key: Array,
    lux_input_data: LuxInputData,
    covariates: list[str],
    num_warmup: int = 1_000,
    num_samples: int = 1_000,
    num_chains: int = 4,
    *,
    two_steps_inference: bool = False,
) -> LuxResult:
    """Run NUTS.

    Args:
        key: PRNGKey.
        lux_input_data: Lux input data..
        covariates: List of covariates of interest.
        num_warmup: Number of warmup iterations. Defaults to 1000.
        num_samples: Number of sampling iterations. Defaults to 1000.
        num_chains: Number of chains. Defaults to 4.
        two_steps_inference: Whether to estimate experimental parameters independently. Defaults to False.

    Returns:
        LuxResult dataclass.
    """
    data = lux_input_data.get_data(covariates)

    bs_c = data.bs_c
    bs_total = data.bs_total
    oxbs_c = data.oxbs_c
    oxbs_total = data.oxbs_total

    bs_c_control = data.bs_c_control
    bs_total_control = data.bs_total_control
    oxbs_c_control = data.oxbs_c_control
    oxbs_total_control = data.oxbs_total_control

    alpha_control = data.alpha_control

    design_matrix = data.design_matrix.astype(float)

    if two_steps_inference:
        nuts_kernel = NUTS(
            luxglm_bs_oxbs_two_step_control_model, init_strategy=init_to_sample
        )
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
        )

        key, key_ = random.split(key, 2)
        mcmc.run(
            key_,
            bs_c_control=bs_c_control,
            bs_total_control=bs_total_control,
            oxbs_c_control=oxbs_c_control,
            oxbs_total_control=oxbs_total_control,
            alpha_control=alpha_control,
        )

        parameters_of_interest = {
            "bs_eff",
            "inaccurate_bs_eff",
            "ox_eff",
            "seq_err",
        }

        posterior_samples = {
            k: v for k, v in mcmc.get_samples().items() if k in parameters_of_interest
        }

        gibbs_fn = get_gibbs_fn(posterior_samples)

        nuts_kernel = NUTS(
            luxglm_bs_oxbs_two_step_noncontrol_model, init_strategy=init_to_sample
        )
        kernel = HMCGibbs(
            nuts_kernel, gibbs_fn=gibbs_fn, gibbs_sites=parameters_of_interest
        )
        mcmc_2 = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
        )

        key, key_ = random.split(key, 2)
        mcmc_2.run(
            key_,
            design_matrix,
            bs_c=bs_c,
            bs_total=bs_total,
            oxbs_c=oxbs_c,
            oxbs_total=oxbs_total,
        )

        inference_metrics = {
            "summary_controls": get_mcmc_summary(mcmc),
            "summary_noncontrols": get_mcmc_summary(mcmc_2),
        }
        posterior_samples = mcmc_2.get_samples() | mcmc.get_samples()

    else:
        nuts_kernel = NUTS(luxglm_bs_oxbs_model, init_strategy=init_to_sample)
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
        )

        key, key_ = random.split(key, 2)
        mcmc.run(
            key_,
            design_matrix,
            bs_c=bs_c,
            bs_total=bs_total,
            oxbs_c=oxbs_c,
            oxbs_total=oxbs_total,
            bs_c_control=bs_c_control,
            bs_total_control=bs_total_control,
            oxbs_c_control=oxbs_c_control,
            oxbs_total_control=oxbs_total_control,
            alpha_control=alpha_control,
        )

        inference_metrics = {"summary": get_mcmc_summary(mcmc)}
        posterior_samples = mcmc.get_samples()

    return LuxResult(
        lux_input_data.metadata_df.name.tolist(),
        covariates,
        inference_metrics,
        lux_input_data.count_df.index.tolist(),
        lux_input_data.control_count_df.index.tolist(),
        posterior_samples,
    )


def run_svi(
    key: Array,
    lux_input_data: LuxInputData,
    covariates: list[str],
    guide: AutoGuide | None = None,
    optim: _NumPyroOptim | None = None,
    loss: ELBO | None = None,
    num_steps: int = 10_000,
    num_samples: int = 1_000,
) -> LuxResult:
    """Run SVI.

    Args:
        key: PRNGKey.
        lux_input_data: Lux input data..
        covariates: List of covariates of interest.
        guide: Automatic guide. Defaults to `AutoNormal(luxglm_bs_oxbs_model, init_loc_fn=init_to_mean)`.
        optim: Optimizer. Defaults to `numpyro.optim.Adam(step.size=1e-1)`.
        loss: Loss function. Defaults to `Trace_ELBO(num_particles=10)`.
        num_steps: Number of optimization steps. Defaults to 10_000.
        num_samples: Number of samples from the guide. Defaults to 1_000.

    Returns:
        LuxResult dataclass.
    """
    data = lux_input_data.get_data(covariates)

    bs_c = data.bs_c
    bs_total = data.bs_total
    oxbs_c = data.oxbs_c
    oxbs_total = data.oxbs_total

    bs_c_control = data.bs_c_control
    bs_total_control = data.bs_total_control
    oxbs_c_control = data.oxbs_c_control
    oxbs_total_control = data.oxbs_total_control

    alpha_control = data.alpha_control

    design_matrix = data.design_matrix.astype(float)

    guide = guide or AutoNormal(
        luxglm_bs_oxbs_model, init_loc_fn=init_to_mean, init_scale=1
    )
    optim = optim or Adam(step_size=1e-1)
    loss = loss or Trace_ELBO(num_particles=10)
    svi = SVI(luxglm_bs_oxbs_model, guide, optim, loss=loss)

    key, key_ = random.split(key, 2)
    svi_result = svi.run(
        key_,
        num_steps,
        design_matrix,
        bs_c=bs_c,
        bs_total=bs_total,
        oxbs_c=oxbs_c,
        oxbs_total=oxbs_total,
        bs_c_control=bs_c_control,
        bs_total_control=bs_total_control,
        oxbs_c_control=oxbs_c_control,
        oxbs_total_control=oxbs_total_control,
        alpha_control=alpha_control,
    )
    params = svi_result.params

    key, key_ = random.split(key, 2)
    posterior_samples = guide.sample_posterior(
        key_, params, sample_shape=(num_samples,)
    )

    inference_metrics = {"params": params, "losses": svi_result.losses}

    return LuxResult(
        lux_input_data.metadata_df.name.tolist(),
        covariates,
        inference_metrics,
        lux_input_data.count_df.index.tolist(),
        lux_input_data.control_count_df.index.tolist(),
        posterior_samples,
    )
