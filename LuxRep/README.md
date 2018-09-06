# LuxRep

## Overview
LuxRep is a methylation analysis tool that models replicates from different bisulphite-converted DNA libraries from the same biological replicate (i.e., technical replicates) thereby allowing use of technical replicates of varying bisulphite conversion rates. Faster processing suited for genome wide analysis is achieved by using variational inference for model estimation. These features extend on **LuxGLM**, a probabilistic method for methylation analysis that handles complex sample designs.

## Features

* Model-based integration of biological and technical replicates
* Full Bayesian inference by variational inference implemented in __Stan__

## Quick introduction

An usual LuxRep pipeline has the following steps

1. Alignment of BS-seq data
2. Extraction of converted and unconverted counts
3. Methylation analysis
	1. Estimate experimental parameters from control cytosine data
	2. Quantify methylation modification and identify differential methylation

This documentation focuses on 3

## Installation

Below the environmental parameter $STAN_HOME refers to the source directory of CmdStan

LuxRep requires the following software

* __CmdStan__ (tested on version 2.18.0)
* __Python__ (tested on version 2.7.15)
* __pystan__ (tested on version 2.17.1.0)
* __Numpy__ (tested on version 1.15.1)
* __Scipy__ (tested on version 1.1.0)


## Using LuxRep

The first step in using LuxRep is to estimate the library-specific experimental parameters bisulphite conversion efficiency (bsEff) and sequecing error rate (seqErr) from control cytosine data. A python script **luxrep\_exp.py** is supplied for generating input files from user-supplied data files and for estimating the experimental parameters (includes compiling of relevant __stan__ code). 

	 usage: luxrep_exp.py [-h] -c CONTROL_DATA_TOTAL_READCOUNTS -m CONTROL_DATA_C_READCOUNTS -l $STAN_HOME
	 
	 Estimates experimental parameters bsEff and seqErr
	 
	 optional arguments:
	 -h, --help										show this help message and exit
	 -c CONTROL_DATA_TOTAL_READCOUNTS, --coverage_data CONTROL_DATA_TOTAL_READCOUNTS	file containing control cytosine total readcount data
	 -m CONTROL_DATA_C_READCOUNTS, --methylation_data CONTROL_DATA_C_READCOUNTS		file containing control cytosine methylation count (C) data
	 -l $STAN_HOME, --cmdstan_loc $STAN_HOME 						cmdstan directory with full pathname
	 -o OUTFOLDER, --outfolder OUTFOLDER							directory containing control output with full pathname
	 -v, --version										show program's version number and exit

For instance, luxrep\_exp.py can be called as

    python luxrep_exp.py -c data/TRs_cov.txt -m data/TRs_meth.txt -l $STAN_HOME

*Input*

The files **TRs\_cov.txt** and **TRs\_meth.txt** contain the total and "C" readcounts, respectively, from control cytosine.

Both files share the same format. The number of columns correspond to the number of bisulphite-coverted DNA libraries and the number of rows correspond to the number of control cytosine (the order of rows and columns should be the same in both files). The first line is the column header consisting of labels for each of the libraries.

Thus, for a dataset consisting of M libraries, a line in **TRs\_cov.txt** shows N<sub><sub>BS</sub>1</sub>\tN<sub><sub>BS</sub>2</sub>\t ... \tN<sub><sub>BS</sub>M</sub> corresponding to a single cytosine and, similarly, for the corresponding TRs\_meth.txt, N<sub><sub>BS,C</sub>1</sub>\tN<sub><sub>BS,C</sub>2</sub>\t ... \tN<sub><sub>BS,C</sub>M</sub>.

When using output coverage files from **Bismark**'s pipeline or one with the same format: `<chromosome> <start position> <end position> <methylation percentage> <count methylated> <count unmethylated>`, with the **bedtools** library the following bash scripts may be used to merge the coverage files into the format required by **luxrep\_exp.py**, where $control\_infile1, $control\_infile2, ..., $control\_infileM refer to the individual coverage files and $name1, $name2, ..., $nameM refer to the library labels for the column header (in the example below, M=3):

	# For the coverage file
	unionBedGraphs -i <(sortBed -i $control_infile1|awk '{$1="chr1";print $1 "\t" $2 "\t" $3 "\t" $5+$6}') <(sortBed -i $control_infile2 |awk '{$1="chr1";print $1 "\t" $2 "\t" $3 "\t" $5+$6}') <(sortBed -i $control_infile3 |awk '{$1="chr1";print $1 "\t" $2 "\t" $3 "\t" $5+$6}') -names $name1 $name2 $name3 -header | cut -f4- > control_cov.txt

	# For the methylation count file
	unionBedGraphs -i <(sortBed -i $control_infile1|awk '{$1="chr1";print $1 "\t" $2 "\t" $3 "\t" $6}') <(sortBed -i $control_infile2|awk '{$1="chr1";print $1 "\t" $2 "\t" $3 "\t" $6}') <(sortBed -i $control_infile3|awk '{$1="chr1";print $1 "\t" $2 "\t" $3 "\t" $6}') -names $name1 $name2 $name3 -header | cut -f4- > control_meth.txt

The examples above may be used for control data generated from the lambda phage genome (`$1="chr1"` converts the notation for the lambda chromosome in the first column into "chr1" so that **sortBed** recognizes it). **unionBedGraphs** then merges the sorted **bed** files and `cut -f4-` removes the first three columns (showing the genomic coordinates).

*Output*

The output files consist of __stan__-generated diagnostic and output files including **output.csv** which contains samples from the approximate posterior of the model parameters. The mean posterior for the parameters bsEff and seqErr from the latter is used as estimates of the experimental parameters used in the second module. 

The second step in using LuxRep is estimating the methylation levels of the noncontrol cytosine and determining differential methylation. A python script **luxrep.py** is supplied for generating input files from user-supplied data files and running the analysis (includes compiling of relevant __stan__ code). 

	 usage: luxrep.py -d NONCONTROL_DATA -s SAMPLE_LIST -m DESIGN_MATRIX -c EXPERIMENTAL_PARAMETERS -l $STAN_HOME
	 
	 Estimates experimental parameters bsEff and seqErr
	 
	 optional arguments:
	 -h, --help								show this help message and exit
	 -d NONCONTROL_DATA, --data NONCONTROL_DATA				file containing noncontrol cytosine data
	 -s SAMPLE_LIST, --sample_list SAMPLE_LIST				file containing sample number of libraries
	 -m DESIGN_MATRIX, --design_matrix					file containing design matrix
	 -c EXPERIMENTAL_PARAMETERS, --exp_params EXPERIMENTAL_PARAMETERS	directory containing output for control data with full pathname
	 -o OUTFOLDER, --outfolder OUTFOLDER					directory containing data analysis output with full pathname
	 -l $STAN_HOME, --cmdstan_loc $STAN_HOME 				cmdstan directory with full pathname
	 -v, --version								show program's version number and exit

For instance, luxrep.py can be called as

    python luxrep.py -d data/chunk_table_0.txt -s data/sample_list.txt -m data/design_matrix.txt -c $CWD/control_dir -l $STAN_HOME

*Input*

The file chunk\_table\_0.txt contains noncontrol cytosine data for all libraries in the dataset with data for each library corresponding to a two-column block, (i) total readcount and (ii) "C" count. The first row contains labels for the libraries and each row thereafter contains data for one cytosine. Each row contains 1 + 2xN<sub>library</sub> columns with the first column showing the genomic coordinates (in the format chromosome:start:end) and the rest showing N<sub><sub>BS</sub>1</sub>\tN<sub><sub>BS,C</sub>1</sub>\tN<sub><sub>BS</sub>2</sub>\tN<sub><sub>BS,C</sub>2</sub>\t ... \tN<sub><sub>BS</sub>M</sub>\tN<sub><sub>BS,C</sub>M</sub> for M libraries. The order of libraries should correspond with the order in the data files for the control cytosine used in the previous module.

When using output coverage files from **Bismark**'s pipeline or one with the same format (see description above in section for control data), with the **bedtools** library the following bash script may be used to merge the coverage files into the format required by **luxrep.py**, where $noncontrol\_infile1, $noncontrol\_infile2, ..., $noncontrol\_infileM refer to the individual coverage files and $name1, $name2, ..., $nameM refer to the library labels for the column header (in the example below, M=3):

	unionBedGraphs -i <(sortBed -i $noncontrol_infile1|awk '{print $1 "\t" $2 "\t" $3 "\t" $5+$6 ";" $6}') <(sortBed -i $noncontrol_infile2 |awk '{print $1 "\t" $2 "\t" $3 "\t" $5+$6 ";" $6}') <(sortBed -i $noncontrol_infile3|awk '{print $1 "\t" $2 "\t" $3 "\t" $5+$6 ";" $6}') -names $name1 $name2 $name3 -header -filler "0;0" | sed 's/'$'\t/:/;s/'$'\t/:/;s/;/'$'\t/g;s/chrom:start:end'$'\t//g' > table_input.txt	

As with the bash scripts for formatting input control data (section above), the total reads are summed from columns 5 and 6 (unmethylated and methylated read counts, respectively) and the individual coverage files are sorted by genomic coordinates (in this example with **sortBed**). The total and methylated read counts are first combined into one column (`$5+$6 ";" $6`) to comply with the format requirement of **unionBedGraphs**, then afterwards replaced with a tab. The output file from **unionBedGraphs** is further formatted so that the tabs separating the first three columns containing the genomic coordinates are replaced with colons, etc. 

The file **sample\_list.txt** lists the sample membership ("biological replicate") of each of the M libraries ("technical replicates"). The file contains one column and M rows in the same order of the libraries in the data files for the control and noncontrol cytosine (i.e., library<sub>row1</sub> = library<sub>column1</sub>). Each row contains one integer x<sub>_i_</sub> &in; 1 ... N<sub>sample</sub>, where _i_ &in; 1 ... M.

The file **design\_matrix.txt** contains covariate information (like in **LuxGLM**) but with one additional row on top and one column on the left for covariate and sample labels (biological replicates), respectively. The row order should follow the sample order in **sample\_list.txt** with N<sub>rows</sub> = N<sub>sample</sub>. The default order of the columns, after the row header, starts with the intercept ("base") followed by the covariate to be tested for differential methylation (in the given example, the covariate "case"), then, if present, followed by the other covariates.  

The directory specified by EXPERIMENTAL\_PARAMETERS contains the output files from the first module. The second module looks for the file **output.csv** and parses it for the posterior mean of the parameters bsEff and seqErr. 

*Output*

The directory specified by OUTFOLDER contains N<sub>cytosine</sub> subfolders each containing input, diagnostic and output files with the latter showing posterior samples of the model parameters for the corresponding cytosine. The second module includes a routine for computing a Bayes factor for differential methylation based on the covariate specified in **design\_matrix.txt**. A summary text file (in **.bed** format) lists the genomic coordinates for each cytosine in the input data file followed by its Bayes factor.

**References**
