# LuxGLM: a probabilistic covariate model for quantification of DNA methylation modifications with complex experimental designs

Overview
-------------
LuxGLM is a method for quantifying oxi-mC species with arbitrary covariate structures from bisulphite based sequencing data. LuxGLM's probabilistic modeling framework combines a previously proposed hierarchical generative model of **Lux** for oxi-mC-seq data and a general linear model component to account for confounding effects.

## Features
- Model-based integration and analysis of BS-seq/oxBS-seq/TAB-seq/fCAB-seq/CAB-seq/redBS-seq/MAB-seq/etc. data from whole genome, reduced representation or targeted experiments
- Accounts for confounding effects through general linear model component
- Considers nonideal experimental parameters through modeling, including e.g. bisulphite conversion and oxidation efficiencies, various chemical labeling and protection steps etc.
- Model-based integration of biological replicates
- Detects differential methylation using Bayes factors (DMRs)
- Full Bayesian inference implemented using **Stan**

## Quick introduction
An usual LuxGLM pipeline has the following steps

1. Alignment of BS-seq and oxBS-seq data (e.g., [Bismark](http://www.bioinformatics.babraham.ac.uk/projects/bismark/) or [BSmooth](http://rafalab.jhsph.edu/bsmooth/))

2.  Extraction of converted and unconverted counts (e.g., [Bismark](http://www.bioinformatics.babraham.ac.uk/projects/bismark/) or [BSmooth](http://rafalab.jhsph.edu/bsmooth/))

3. Integrative methylation analysis

4. Analysis of obtained methylation estimates, e.g., using Bayes factors

This documentation focus on points three and four.

## Prerequisites 
* Python 2.7 (https://www.python.org/)
* PyStan (http://pystan.readthedocs.org/en/latest/)
* NumPy (http://www.numpy.org/)
* SciPy (http://www.scipy.org/)

## Installation
To use our LuxGLM interface, one has to have PyStan installed. For instructions on installation of **PyStan**, please see the documentation of **PyStan** at http://pystan.readthedocs.io/en/latest/index.html.

In addition to the Python interface, LuxGLM can be used through other available **Stan** interfaces (see http://mc-stan.org/interfaces/ for a list of available interfaces). Please take into account that we only provide a Python interface for preparing the necessary inputs variables; that is, if **RStan** or other interface is used, then the user has to write approriate routines for parsing input variables from data files.

Using LuxGLM
-------------

A python script **luxglm<span></span>.py** is supplied for generating input variables from user-supplied data files and running the analysis. 

    usage: luxglm.py [-h] -c CONTROL_DATA -p CONTROL_PRIOR -d DATA_FILE -m DESIGN_FILE [-v]

    LuxGLM

     optional arguments:
     -h, --help                                       show this help message and exit
     -c CONTROL_DATA, --control_data CONTROL_DATA     file containing control cytosine data
     -p CONTROL_PRIOR, --control_prior CONTROL_PRIOR  file containing prior knowledge on control cytosines
     -d DATA_FILE, --data DATA_FILE                   file containing cytosine data
     -m DESIGN_FILE, --design_matrix DESIGN_FILE      file containing design matrix
     -o OUTPUT_FILE, --output OUTPUT_FILE             file for storing samples
     -v, --version                                    show program's version number and exit

For instance, the script **luxglm<span></span>.py** can be called as

    $ python luxglm.py -c control_data.txt -p control_prior.txt -d data.txt -m design_matrix.txt -o samples.p

In the following sections, we will cover describe the input files and their format.

#### Control cytosines
The files **control_data.txt** and **control_prior.txt** have the count data and prior information on the control cytosines of interest, respectively.

Each line (one line per control cytosine) in the file **control_data.txt** is composed of one or more sample-specific tab-separated blocks of four tab-separated nonnegative integers
>N<sub>BS</sub><sup>C</sup>\tN<sub>BS</sub>\tN<sub>oxBS</sub><sup>C</sup>\tN<sub>oxBS</sub>

That is, the number of Cs and the total BS-seq read-outs are listed first, which are followed by the number of Cs and total oxBS-seq read-outs.
Each line in **control_data.txt** should have exactly 4×N<sub>samples</sub> values separated with the tabs (one quadruple per sample). Importantly, each of the sample specified in **data.txt** (covered in the next section) should have its own control data. Moreover, the order of the sample-specific blocks in **control_data.txt** and **data.txt** is assumed to be the same.

The prior knowledge on the control cytosines is supplied in the file **control_prior.txt**. Although the hierarchical model allows that the control cytosines would have different priors between replicates, this is not implemented in the current version. Therefore, each line in **control_prior.txt** should have exactly three tab-separated values and the order of the rows, i.e., control cytosines, should be the same in **control_data.txt** and **control_prior.txt**.

#### Noncontrol cytosines
The files **data.txt** and **design_matrix.txt** have the count data of the noncontrol cytosines of interest and the covariate structure of the samples, respectively.

As with control cytosines in **control_data.txt**, each of the noncontrol cytosine has its own line in **data.txt**. Each line in **data.txt** is composed of one or more sample-specific tab-separated blocks of four tab-separated nonnegative integers
>N<sub>BS</sub><sup>C</sup>\tN<sub>BS</sub>\tN<sub>oxBS</sub><sup>C</sup>\tN<sub>oxBS</sub>

That is, the number of Cs and the total BS-seq read-outs are listed first, which are followed by the number of Cs and total oxBS-seq read-outs. Moreover, on each line there should be exactly 4×N<sub>samples</sub> values separated with the tabs (one quadruple per sample).

The file **design_matrix.txt** holds covariate information (of N<sub>covariates</sub>) of the N<sub>samples</sub> samples
>D<sub>1,1</sub>\tD<sub>1,2</sub>\t…\tD<sub>1,N<sub>covariates</sub></sub><br>
>D<sub>2,1</sub>\tD<sub>2,2</sub>\t…\tD<sub>2,N<sub>covariates</sub></sub><br>
>⋮<br>
>D<sub>N<sub>samples</sub>,1</sub>\tD<sub>N<sub>samples</sub>,2</sub>\t…\tD<sub>N<sub>samples</sub>,N<sub>covariates</sub></sub>

The order of the samples (i.e. rows) in **design_matrix.txt** should match with the order of blocks in **control_data.txt** and **data.txt**. 

#### Output
The obtained posterior samples are stored in the variable pickled in the file **samples.p**; the pickled variable is obtained using the *extract* method of the obtained *StanFit* instance (for more details, please see http://pystan.readthedocs.io/en/latest/api.html).

Most of the users are interested in posterior samples of methylation levels

 * **theta** for noncontrol cytosines and
 * **theta_control** for control cytosines
 
and posterior samples of the experimental parameters

 * **bsEff** for bisulphite conversion efficiency,
 * **bsBEff** for inaccurate bisulphite conversion efficiency,
 * **oxEff** for oxidation efficiency, and
 * **seqErr** for sequencing error probability.

For instance, the posterior mean estimate of **oxEff** can be obtained as

    >>> print samples['oxEff'].mean(0)

To study the effects of covariates, one can inspect the coefficient matrix **B**. For instance, we can assess differential methylation by studying **B** as described in the next section.

#### Estimation of Bayes factors using Savage-Dickey density ratio
Savage-Dickey density ratio is implemented by the *savagedickey* routine in **lux_routines.py**.

To demonstrate this let us consider a simple example of two conditions and six samples (three for each condition).
In this case, the design matrix input file would contain (assuming that the first three samples would correspond to the first condition)
>1\t0<br>
>1\t0<br>
>1\t0<br>
>0\t1<br>
>0\t1<br>
>0\t1

To assess whether the methylation status differs between the two conditions, we should study the estimated coefficient matrix **B**.
In more detail, we can call the *savagedickey* routine to compare the coefficients of the first and second covariates (remember that Y=DB and θ=softmax(Y))

    >>> from luxglm_routines import savagedickey
    >>> print savagedickey(samples['B'][:,0,:],samples['B'][:,1,:])

The routine will return a Savage-Dickey approximation of the Bayes factor.

## Examples
Two examples from the manuscript (**foxp3_time.py** and **foxp3_ra.py**) are provided. The scripts **foxp3_time.py** and **foxp3_ra.py** are modified from **luxglm<span></span>.py** to consider only subsets of our *Foxp3* data. These examples can be run as follows

    $ python foxp3_time.py -c Data/control_data.txt -p Data/control_prior.txt -d Data/data.txt -m Data/design_matrix.txt
    $ python foxp3_ra.py -c Data/control_data.txt -p Data/control_prior.txt -d Data/data.txt -m Data/design_matrix.txt

Please check **Data/control_data.txt**, **Data/control_prior.txt**, **Data/data.txt**, and **Data/design_matrix.txt** for input file examples. 

References
-------------
[1] T. Äijö, X. Yue, A. Rao and H. Lähdesmäki, “LuxGLM: a probabilistic covariate model for quantification of DNA methylation modifications with complex experimental designs.,” Bioinformatics, 32.17:i511-i519, Sep 2016.

[2] T. Äijö, Y. Huang, H. Mannerström, L. Chavez, A. Tsagaratou, A. Rao and H. Lähdesmäki, “A probabilistic generative model for quantification of DNA modifications enables analysis of demethylation pathways.,” Genome Biol, 17.1:1, Mar 2016.

[3] F. Krueger and S. R. Andrews, “Bismark: a flexible aligner and methylation caller for Bisulfite-Seq applications.,” Bioinformatics, 27.11:1571-1572, Jun 2011. 

[4] K. D. Hansen, B. Langmead and R. A. Irizarry, “BSmooth: from whole genome bisulfite sequencing reads to differentially methylated regions.,” Genome Biol, 13.10:1, Oct 2012. 

[5] Stan Development Team, “Stan Modeling Language Users Guide and Reference Manual, Version 2.14.0.,” http://mc-stan.org, Jan 2016. 

[6] Stan Development Team, “PyStan: the Python interface to Stan, Version 2.14.0.0.,” http://mc-stan.org, Jan 2016.
