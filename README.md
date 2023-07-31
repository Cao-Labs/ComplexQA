## ComplexQA: A Deep Graph Learning Approach for Protein Complex Structure Assessment
Updating

## Setup

#### Note, this software only works on linux environments for the time being

1. Create python virtual environment
	1. Virtualenv
		1. `pip install virtualenv` *`pip3` if you still have python2* 
		1. `python3 -m venv virtual-env-name` This creates a new virtual environment 
		1. `source virtual-env-name/bin/activate` This activates your new virtual environment 
	1. Conda 
		1. Download [Anaconda](https://www.anaconda.com/products/individual) *download the linux version to your linux machine* 
		1. Install Anaconda and follow the isntallation instructions, select yes for the init question at very end
		1. `conda activate base` to get into your 'base' environment, do not install packages to 'base' 
		1. `conda create -n virtual-env-name python=3.7`
		1. `conda activate virtual-env-name` this activates your new environment, this is where you install packages 
1. `pip install -r requirements.txt`
1. Navigate to the `ComplexQA`/ folder 
1. Run `python install.py` to complete setup 

## Execution
1. Navigate to ComplexQA folder (You can now run this script from anywhere!)
2. `python prepare_data.py -i ./QA_examples/Input -o 1JTD_interface.csv`
  - This command calculates the protein complex interface and saves the information to a csv file
    If you have reference fasta file
   `python prepare_data.py -i ./QA_examples/Input -o 1JTD_interface.csv`
3. `python prediction_complex.py ./QA_examples/Input/1JTD 1JTD_interface.csv ./TEST_OUT/`
  - This command runs the prediction and places a csv file in TEST_OUT/ folder


#### Notes for execution
- Currently, the input data must be in a folder even if you are only running one pdb. Please put pdbs in a folder named as the target name. 
```
QA_examples
└───Input
      └───target_name
          │   input_file_1.pdb
          │   input_file_1.pdb
          │   ...
```
- Currently only works on one `target_name` as shown above, will be updated soon


## Ideas 
* Distance map 
	* Calculate the differnece betwen the features as another feature vector. (i.e value 1 is diff between featuer 0 and 1, value 2 is diff between feature 1 and 2, so on) 
	* We could even make this a matrix by calculating the difference between feature 0 and all other features (this is a vector) then calculating the difference of feature 1 and all other features making a sort of distance map 
