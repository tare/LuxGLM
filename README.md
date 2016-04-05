# LuxGLM

## Prerequisites 
* Python 2.7 (https://www.python.org/)
* PyStan (http://pystan.readthedocs.org/en/latest/)
* NumPy (http://www.numpy.org/)
* SciPy (http://www.scipy.org/)

## Examples
Two examples from the manuscript (**foxp3_time.py** and **foxp3_ra.py**) are provided. The examples can be run as follows

    python foxp3_time.py -c Data/control_data.txt -p Data/control_prior.txt -d Data/data.txt -m Data/design_matrix.txt
    python foxp3_ra.py -c Data/control_data.txt -p Data/control_prior.txt -d Data/data.txt -m Data/design_matrix.txt

See also `python foxp3_time.py --help` and `python foxp3_ra.py --help`.
