# Installing PyBnesian (Enrique Valero-Leal)
This is a more extensive guide on how to install PyBnesian on an Ubuntu computer and minimise error in the installation. If any error is detected throughout it, please contact [Enrique-Val](https://github.com/Enrique-Val) on GitHub and we will try to publish a new version addressing it.


## Requirements of the Ubuntu computer for instaling PyBnesian
Before diving into installing Python packages, we need to ensure that certain C++ and OpenCL related packages are installed. We can do that with the following commands:

	sudo apt install cmake
	sudo apt install g++
	sudo apt install opencl-headers 
	sudo apt install ocl-icd-opencl-dev
	

## Configure an environemnt
Set up a virtual environment for Python 3.10. Conda is recommended and all the details about its intallation can be found at the [Anaconda website](https://www.anaconda.com/download).

**Important:** Once it is installed, execute the rest of the commands within your environment. You can access it via your terminal with command.

	conda activate myenv
	
When done correctly, you will read *(myenv)* right before your terminal prompt.

### Install pyarrow version 12.0.1. Posterior versions seem to cause trouble
Install pyarrow version 12.0.1 before installing PyBnesian. If not, PyBnesian will install later versions, which may cause trouble. In particular, we found that to be the cause of the error *undefined symbol _ZNK5arrow6Status8ToStringEv*

	pip install pyarrow==12.0.1


## Install PyBnesian
This could be done either with pip or from source. The first option is simpler, while we recommend the second option since the library might still need some small tweaks and this allows for recompiling it

### Option 1: Installing PyBnesian from pip
Run the command:

	pip install pybnesian
	

### Option 2: Installing pybnesian from source
First of all, clone the PyBnesian repo into your computer running the command:

	clone https://github.com/davenza/PyBNesian 
	
For the latest ongoing dev changes, you might consider alternatively cloning Enrique-Val's PyBnesian fork:

	https://github.com/Enrique-Val/PyBNesian
	
Enter into the newly created folder and run the installation file.
	
	cd PyBnesian
	python setup.py install


## Install pandas
A bug prevents PyBnesian from being imported if pandas is not installed and imported previously. As such make sure to install pandas running the following command and to import it in your project before importing PyBnesian.

	pip install pandas


## Useful links
Links that might be of help when dealing with PyBnesian installation.
1. [https://www.sasview.org/docs/old_docs/4.1.2/user/opencl_installation.html](https://www.sasview.org/docs/old_docs/4.1.2/user/opencl_installation.html)

## Acknowledgements
Thanks to David Atienza for developing PyBnesian in the first place. Also, Carlos Li assisted to find the commands for solving OpenCL headers related issues.
