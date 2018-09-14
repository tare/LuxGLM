## TUTORIAL

The following tutorial consists of instructions for using LuxRep to perform methylation analysis on a dataset consisting of 38 DNA libraries (from 22 biological samples). The coverage files consist of count data from loci in chr22.

#### Supplied files (in data folder)

*For estimating experimental parameters*

* **lambda\_fileList.txt** - list of bed files from the **Bismark** pipeline (or similar format) containing counts from control cytosine
* **nameList.txt** - space delimited list of labels for files listed in **lambda\_fileList.txt**
* **lambda\_bed\_files.tar.gz**	- compressed folder containing control cytosine bed files
	
*For methylation analysis*

* **hg19\_bed\_files.tar.gz** - compressed folder containing noncontrol cytosine bed files
* **sample_list.txt** -	file listing group membership (biological sample/replicate) of library, one line per library
* **design_matrix.txt** - file encoding design matrix
* **nameList.txt** - space-delimited list of labels for files listed in **lambda_fileList.txt**
	

#### Preliminaries

* Download LuxRep
* See **Installation** section of **README**

#### Estimating experimental parameters

	cd $LUXREP_HOME
	cd data
	
	# extract bed files (output from bismark pipeline) for control cytosine, one file for each library
	tar -xzvf lambda_bed_files.tar.gz
	
	# Estimate experimental parameters
	
	cd ..
	python luxrep_exp.py -f data/lambda_fileList.txt -n data/nameList.txt -o $PWD/controls -l $STAN_HOME	

#### Methylation analysis

*Preprocess noncontrol cytosine data files*

**unionBedGraphs** accepts bed files with 4 tab-delimited columns with the first three containing genomic coordinates. The commands below take the sum of columns 5 and 6 (from the **bismark** pipeline, these contain the counts for methylated and unmethylated cytosines, respectively) and, together with the first three columns, pipes the sum and methylated counts into an intermediate file. 

	cd $LUXREP_HOME
	cd data
	
	# extract bed files output from bismark pipeline for noncontrol cytosine, one file for each library
	
	tar -xzvf hg19_bed_files.tar.gz
	
	cd hg19_bed_files
	
	for i in {1..38}; do file=$i".bed"; ls $file; rm -f $i"_temp.bed"; cat $file | awk '{print $1 "\t" $2 "\t" $3 "\t" $5+$6 ";" $5}' > $i"_temp.bed"; done
	
	# combine data on total read count and methylated cytosine count, via the intermediate files 1_temp.bed ... M_temp.bed, in this case M=38
	
	unionBedGraphs -i $(cat ../fileList.txt) -names $(cat ../nameList.txt) -header -filler "0;0" | sed 's/'$'\t/:/;s/'$'\t/:/;s/;/'$'\t/g;s/chrom:start:end'$'\t//g' > ../hg19.tab
	
	cd ../..
	
	# Optional
	# split hg19.tab into smaller tables (size -s cytosine), like in Radmeth input tables, for parallelization 
	
    python split_inputTable.py -d data/hg19.tab -s 1000 -o $PWD/results

   The last line above allows splitting of the input table into smaller-sized tables (like in **Radmeth**) for parallelization. 

*Estimate methylation levels and calculate bayes factors for differential methylation*

By default, bayes factor is calculated for the significance of the second covariate in the design matrix (in the example **data/design_matrix.txt**, this is the "case" column).

	python luxrep.py -d results/counts_1.tab -s data/sample_list.txt -m data/design_matrix.txt -n data/nameList.txt -c $PWD/controls -o $PWD/results/1 -l $STAN_HOME

To test significance of other covariates (after an initial run), a routine is supplied that calculates bayes factors for a specified covariate using the generated **output.csv** files.

	# list output.csv files with full pathname for all cytosine to be included
	ls -d1 $PWD/results/1/*/output.csv > outputList.txt
	
	python 	# open python interpreter
	
	>>> import luxrep_routines
	>>> n = 3 # column number in design matrix corresponding to covariate of interest
	>>> file_list = 'outputList.txt' # file containing list of output.csv files
	>>> luxrep_routines.savagedickey2(file_list, n) # lists bayes factors and genomic positions in file bf_$n.bed

An example of the output file is supplied as **data/misc/bfs\_1.bed** (also in **data/misc** is **counts\_1.tab**, an output of **split_inputTable.py** and example input file to **luxrep.py**).
