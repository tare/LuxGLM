# LuxGLM: a probabilistic covariate model for quantification of DNA methylation modifications with complex experimental designs

## Overview

LuxGLM is a method for quantifying oxi-mC species with arbitrary covariate structures from bisulphite based sequencing data. LuxGLM's probabilistic modeling framework combines a previously proposed hierarchical generative model of Lux for oxi-mC-seq data and a general linear model component to account for confounding effects.

## Features

- Model-based integration and analysis of BS-seq/oxBS-seq/TAB-seq/fCAB-seq/CAB-seq/redBS-seq/MAB-seq/etc. data from whole genome, reduced representation or targeted experiments
- Accounts for confounding effects through general linear model component
- Considers nonideal experimental parameters through modeling, including e.g. bisulphite conversion and oxidation efficiencies, various chemical labeling and protection steps etc.
- Model-based integration of biological replicates
- Detects differential methylation using Bayes factors (DMRs)
- Full Bayesian inference using NumPyro

## Quick introduction

An usual LuxGLM pipeline has the following steps

1. Alignment of BS-seq and oxBS-seq data (e.g., [Bismark](http://www.bioinformatics.babraham.ac.uk/projects/bismark/) or [BSmooth](http://rafalab.jhsph.edu/bsmooth/))

2. Extraction of converted and unconverted counts (e.g., [Bismark](http://www.bioinformatics.babraham.ac.uk/projects/bismark/) or [BSmooth](http://rafalab.jhsph.edu/bsmooth/))

3. Integrative methylation analysis

4. Analysis of obtained methylation estimates, e.g., using Bayes factors

This documentation focuses on the third and fourth points.

## Installation

### PyPI

```console
$ pip install luglm
```

### GitHub

Install the version from the main branch as follows

```console
$ pip install git+https://github.com/tare/LuxGLM.git
```

## Usage

### Metadata

Count data and covariates are defined in the metadata file.

| name       | basal/tgf-beta | vitc | ra  | timepoint | count_file              | control_count_file             | control_definition_file |
| ---------- | -------------- | ---- | --- | --------- | ----------------------- | ------------------------------ | ----------------------- |
| TGFb_1_24h | 1              | 0    | 0   | 24        | wildtype/TGFb_1_24h.tsv | control/TGFb_1_24h_control.tsv | control_definitions.tsv |
| TGFb_1_38h | 1              | 0    | 0   | 38        | wildtype/TGFb_1_38h.tsv | control/TGFb_1_38h_control.tsv | control_definitions.tsv |
| TGFb_1_48h | 1              | 0    | 0   | 48        | wildtype/TGFb_1_48h.tsv | control/TGFb_1_48h_control.tsv | control_definitions.tsv |

The following columns are mandatory: `name`, `count_file`, `control_count_file`, and `control_definition`. Additionally, there has to be at least one covariate. In the above example, we have four covariates: `basal/tgf-beta`, `vitc`, `ra`, and `timepoint`.

### Control cytosines

The control cytosine data are supplied in the control count files. Each experiment will have its own file. The files contain location information (`chromosome` and `position`) and control type information (`control_cytosine`) for the control cytosines. Additionally, we have the number of Cs and and total number of read-outs from BS-seq and oxBS-seq experiments (`bs_c`, `bs_total`, `oxbs_c`, and `oxbs_total`).

| chromosome  | position | control_type | bs_c | bs_total | oxbs_c | oxbs_total |
| ----------- | -------- | ------------ | ---- | -------- | ------ | ---------- |
| Lambda_ctrl | 22924    | C            | 2    | 343      | 1      | 562        |
| Lambda_ctrl | 22928    | C            | 2    | 341      | 1      | 561        |
| Lambda_ctrl | 47359    | 5mC          | 3770 | 3857     | 4767   | 4877       |
| Lambda_ctrl | 47367    | 5mC          | 3895 | 3962     | 4855   | 4979       |
| Lambda_ctrl | 23789    | 5hmC         | 3792 | 3964     | 79     | 865        |
| Lambda_ctrl | 23794    | 5hmC         | 3901 | 4115     | 62     | 934        |

The prior knowledge on the control cytosines is supplied in the control definition file. Note that `control_type` is used to link the control count data and control definitions.

| control_type | C_pseudocount | 5mC_pseudocount | 5hmC_pseudocount |
| ------------ | ------------- | --------------- | ---------------- |
| C            | 998           | 1               | 1                |
| 5mC          | 1             | 998             | 1                |
| 5hmC         | 6             | 2               | 72               |

### Noncontrol cytosines

The non-control cytosine data are supplied in the count files. Each experiment will have its own file. The files contain location information (`chromosome` and `position`) for the non-control cytosines. Additionally, we have the number of Cs and and total number of read-outs from BS-seq and oxBS-seq experiments (`bs_c`, `bs_total`, `oxbs_c`, and `oxbs_total`).

| chromosome | position | bs_c | bs_total | oxbs_c | oxbs_total |
| ---------- | -------- | ---- | -------- | ------ | ---------- |
| chrX       | 7159069  | 1083 | 1563     | 850    | 2736       |
| chrX       | 7159186  | 1341 | 1534     | 2119   | 2719       |
| chrX       | 7159222  | 4949 | 5575     | 3886   | 4639       |
| chrX       | 7159235  | 4831 | 5588     | 4354   | 4641       |

### LuxGLM analysis

The following lines are sufficient to run LuxGLM

```python
import numpyro
from jax import random
from luxglm.inference import run_nuts
from luxglm.utils import get_input_data

numpyro.enable_x64()

# read input data
lux_input_data = get_input_data("metadata.tsv")

key = random.PRNGKey(0)
key, key_ = random.split(key)

# run LuxGLM
lux_result = run_nuts(
    key,
    lux_input_data,
    ["basal/tgf-beta"],
    num_warmup=1_000,
    num_samples=1_000,
    num_chains=4,
)

# ensure convergence
lux_result.inference_metrics["summary"].query("r_hat > 1.05")
```

To get the posterior samples of methylation levels of control and non-control cytosines one can call `lux_result.methylation_controls()` and `lux_result.methylation()`, respectively.

The posterior samples of experimental parameters can be obtained by calling `lux_result.experimental_parameters()`.

To study the effects of the covariates, one can get the posterior samples of coefficients of covariates using `lux_result.coefficients()`.

### Examples

Please see the [examples](examples/) directory for the tutorial notebooks.

## References

[1] T. Äijö, X. Yue, A. Rao and H. Lähdesmäki, “LuxGLM: a probabilistic covariate model for quantification of DNA methylation modifications with complex experimental designs.,” Bioinformatics, 32.17:i511-i519, Sep 2016.

[2] T. Äijö, Y. Huang, H. Mannerström, L. Chavez, A. Tsagaratou, A. Rao and H. Lähdesmäki, “A probabilistic generative model for quantification of DNA modifications enables analysis of demethylation pathways.,” Genome Biol, 17.1:1, Mar 2016.

[3] F. Krueger and S. R. Andrews, “Bismark: a flexible aligner and methylation caller for Bisulfite-Seq applications.,” Bioinformatics, 27.11:1571-1572, Jun 2011.

[4] K. D. Hansen, B. Langmead and R. A. Irizarry, “BSmooth: from whole genome bisulfite sequencing reads to differentially methylated regions.,” Genome Biol, 13.10:1, Oct 2012.

[5] D. Phan, N. Pradhan and M. Jankowiak, “Composable Effects for Flexible and Accelerated Probabilistic Programming in NumPyro.,” arXiv preprint 1912.11554, Dec 2019.
