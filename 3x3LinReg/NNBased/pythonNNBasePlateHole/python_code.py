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
#  Linear isotropic Hookean law. Implemented via a scikit-learn linear 
#  regression model. The inputs are 3 strains, the outputs are 3 stresses.

# Authors
#  Simon A. Rodriguez, UCD. All rights reserved
#  Philip Cardiff, UCD. All rights reserved

from joblib import load

ML_model = load('ML_model.joblib')

#Load the scalers
x_scaler = load('x_scaler.joblib')
y_scaler = load('y_scaler.joblib')

def predict():
    new_epsilon = np.delete(epsilon, 5, 1)
    new_epsilon = np.delete(new_epsilon, 4, 1)
    new_epsilon = np.delete(new_epsilon, 2, 1)
    epsilon_scaled = x_scaler.transform(new_epsilon)
    sigma_scaled = ML_model.predict(epsilon_scaled)
    new_sigma = y_scaler.inverse_transform(sigma_scaled)
    sigma[:, 0] = new_sigma[:, 0]
    sigma[:, 1] = new_sigma[:, 1]
    sigma[:, 3] = new_sigma[:, 2]
    ##########################################################
