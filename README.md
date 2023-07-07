# README #

### What is this? ###

This repository contains OpenFOAM + Python code that shows how to combine OpenFOAM and Python to get the Autoprogressive (SelfSim) [1] algorithm implemented in a finite volume context, as presented by Rodriguez et Al [2] at the OpenFOAM Workshop 18. The mechanics solver is OpenFOAM and the material model is given by a Python-based machine learning method. The OpenFOAM/Python interoperability is achieved via pythonPal4foam [3]. 

The test-case is a solid mechanics problem where a linear elastic steel plate with a hole which is stuck to a wall is deformed by applying a load to one of the boundaries. The material model is either a scikit-learn-based linear regression model or a TensorFlow-based neural network. Initially, the material model is trained with wrong material properties. After applying the SelfSim algorithm, the correct material model is learned. 

Note that this problem is 2D, therefore, some components of the strains and stresses are zero. For that reason, it is implemented 4 times:

1. 3x3LinReg: The material model is a scikit-learn-based linear regression that maps 3 strain components to 3 stress components.
2. 3x3NN: The material model is a TensorFlow-based neural network that maps 3 strain components to 3 stress components.
1. 6x6LinReg: The material model is a scikit-learn-based linear regression that maps 6 strain components to 6 stress components.
2. 6x6NN: The material model is a TensorFlow-based neural network that maps 6 strain components to 6 stress components.

### [Manual approach] How do I get set up? ###

In addition to an installation of OpenFOAM-9, a Python installation is required. The following Python packages are required, where the versions used to generate the results are:

* Python 3.8.12

* NumPy 1.18.5

* pybind11 2.8.1

* TensorFlow 2.4.0

* pip 21.3.1

* Matplotlib 3.3.1

* scikit-learn 1.0.1

* pandas 1.1.1

* tqdm 4.50.2

These libraries can be installed from the supplied pybind-no-gpu.yml file using the conda software (https://conda.io). Once conda is installed, the Python environment is installed with:

    conda env create -f pybind-no-gpu.yml

The conda environment can be activated with:

    conda activate pybind-no-gpu

Please be aware that the examples may not work with other versions of Python libraries, although they are likely to work with similar versions, e.g. Python 3.8.*.

The test case uses the solids4foam [2] toolbox. To install it, follow the instructions at https://solids4foam.github.io 

Once you have solids4foam on your system, set the following environment variable:

    export SOLIDS4FOAM_INST_DIR=<location_of_solids4foam>

In addition, two pybind11 environment variables must be defined, for example, as:

    export PYBIND11_INC_DIR=$(python3 -m pybind11 --includes)
    export PYBIND11_LIB_DIR=$(python3 -c 'from distutils import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')

Notice that you have to manually define the location of the SOLIDS4FOAM_INST_DIR in the "input_file.py" files for every case you want to run. The one on this code is configured to work on the specific PC where it was tested.

Once those environment variables have been defined, the OpenFOAM code included in the current repository can be compiled with the Allwmake script in the parent folder:

    ./Allwmake

If the Allwmake script gives the error “libpython or lpython not found” then please manually update the LD_LIBRARY_PATH environment variable with "export LD_LIBRARY_PATH=$PYBIND11_LIB_DIR:$LD_LIBRARY_PATH” and run the “./Allwmake” command again.


### How do I run the cases? ###

To run a particular tutorial, navigate to the parent folder for the specific tutorial and execute the provided Allrun script. For instance, to run the case where the material model maps 3 strain components to 3 stress components via a linear regression tutorial:

    cd 3x3LinReg
    ./Allrun

All the tutorials can be run with the following command (this takes ~48 hr on a 1-core modern system):

    ./Allrun


### Compatible OpenFOAM versions ###

The included code was tested with OpenFOAM-9 (it will probably work with others too).

If desired, the user can make the small changes required to get the cases to work with their particular version of OpenFOAM.


### Who do I talk to? ###

    Simon Rodriguez
    simon.rodriguezluzardo@ucdconnect.ie
    https://www.linkedin.com/in/simonrodriguezl/
    
    Philip Cardiff
    philip.cardiff@ucd.ie
    https://www.linkedin.com/in/philipcardiff/



### References ###

[1]	J. Ghaboussi, Soft computing in engineering. Boca Raton: CRC Press, Taylor & Francis Group, 2018.

[2] Simon Rodriguez, M. Celikin, Padraig Cunningham, and P. Cardiff, 'Learning constitutive models from mechanical tests with the Autoprogressive algorithm based on OpenFOAM and Python', 2023, doi: 10.13140/RG.2.2.34635.98080.

[3]	S. A. Rodriguez Luzardo and P. Cardiff, ‘A General Approach for Running Python Codes in OpenFOAM Using an Embedded PYBIND11 Python Interpreter’, OpenFOAM® J., vol. 2, pp. 166–182, Dec. 2022, doi: 10.51560/ofj.v2.79.

