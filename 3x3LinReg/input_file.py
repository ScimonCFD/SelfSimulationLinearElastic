# License
#  This program is free software: you can redistribute it and/or modify 
#  it under the terms of the GNU General Public License as published 
#  by the Free Software Foundation, either version 3 of the License, 
#  or (at your option) any later version.

#  This program is distributed in the hope that it will be useful, 
#  but WITHOUT ANY WARRANTY; without even the implied warranty of 
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 

#  See the GNU General Public License for more details. You should have 
#  received a copy of the GNU General Public License along with this 
#  program. If not, see <https://www.gnu.org/licenses/>. 

# Description
#  This file contains the declaration of several variables required in the 
#  implemented self-simulation algorithm.

# Authors
#  Simon A. Rodriguez, UCD. All rights reserved
#  Philip Cardiff, UCD. All rights reserved

import os

# Routes to the folders
ROUTE_THEORETICAL_MODEL = "./Theoretical/solids4foamPlateHole/"
ROUTE_NN_MODEL = "./NNBased/pythonNNBasePlateHole/"
ROUTE_TO_NEURAL_NETWORK_CODE = "./linearRegression/"

current_env=os.environ.copy()

PYBIND11_INC_DIR = "$(python3 -m pybind11 --includes)"
PYBIND11_LIB_DIR = "$(python3 -c 'from distutils import sysconfig; " \
                   + "print(sysconfig.get_config_var('LIBDIR'))')"
SOLIDS4FOAM_INST_DIR = "/home/simon/OpenFOAM/simon-9/solids4foam-release"

# Set environment variables
current_env["PYBIND11_INC_DIR"] = PYBIND11_INC_DIR
current_env["PYBIND11_LIB_DIR"] = PYBIND11_LIB_DIR
current_env["SOLIDS4FOAM_INST_DIR"] = SOLIDS4FOAM_INST_DIR

# Set several constants up
TOL_LOCAL_ITER = 1e-9 
TOTAL_ITERATIONS = 5 #Max local iterations 
TOTAL_LOAD_INCREMENTS = 15
TOTAL_NUMBER_PASSES = 10
SETS_IN_MOVING_WINDOW = 15
SEED = 2

# The strains in the training set will be a subset from the strains in the
# theoretical simulation
SUBSAMPLE_ORIGINAL_STRAINS = True

# Material properties
E = 100e9 #Young's modulus
v = 0.1 #Poisson's ratio
LAME_1 = E * v / ((1 + v) * (1 - 2 *v))
LAME_2 = E / (2 * (1 + v))