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
#  This routine trains a linear regression model to replace a linear elastic 
#  Hookean law, following the procedure presented in the selfSimulation 
#  algorithm and using OpenFOAM + Python via pythonPal4foam. 

# Authors
#  Simon A. Rodriguez, UCD. All rights reserved
#  Philip Cardiff, UCD. All rights reserved

import os
import numpy as np
import auxiliary_functions
from auxiliary_functions import *
from joblib import dump
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import random
import input_file
from input_file import *
from sklearn.preprocessing import MinMaxScaler
from joblib import load
from distutils.dir_util import mkpath
import time

# record start time
start = time.time()


# Create a folder to plot the results
mkpath(ROUTE_NN_MODEL + "Results/")

# Copy input_file.py to the neuralNetwork folder
terminal("cp input_file.py " + ROUTE_TO_NEURAL_NETWORK_CODE + "input_file.py")

# Run the theoretical simulation
terminal("cd " + ROUTE_THEORETICAL_MODEL + " && ./Allrun") 

# Create a mae loss function
mae = tf.keras.losses.MeanAbsoluteError()

# Seed everything
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Assemble the data set with the expected data
for load_inc in range(1, TOTAL_LOAD_INCREMENTS + 1):
    if(load_inc == 1):
        original_x_training_set = deserialise(ROUTE_THEORETICAL_MODEL + 
                                              str(int(load_inc)) + 
                                              "/", "epsilon")
    else:
        x_train_temp = deserialise(ROUTE_THEORETICAL_MODEL 
                                   + str(int(load_inc)) + "/", "epsilon")
        original_x_training_set = np.concatenate((original_x_training_set, 
                                                  x_train_temp), axis=0)
# Save the strains to a NumPy file
serialise(original_x_training_set, ROUTE_TO_NEURAL_NETWORK_CODE, 
          "original_x_training_set")

# Create and train the initial linear regression
terminal("cd " + ROUTE_TO_NEURAL_NETWORK_CODE + " && python main.py")

if (SUBSAMPLE_ORIGINAL_STRAINS):
    original_x_training_set = deserialise(ROUTE_TO_NEURAL_NETWORK_CODE, 
                                          "x_train")


# Bring the neural network and the scalers to the parent folder
terminal("cp " + ROUTE_TO_NEURAL_NETWORK_CODE + "ML_model.h5 " + 
          ROUTE_NN_MODEL)

terminal("cp " + ROUTE_TO_NEURAL_NETWORK_CODE + "*scaler.joblib " + 
          ROUTE_NN_MODEL)

# Load the initial neural network
ML_model= keras.models.load_model(ROUTE_NN_MODEL +"ML_model.h5")

# Load the stresses used at training stage
original_y_training_set = deserialise(ROUTE_TO_NEURAL_NETWORK_CODE, "y_train")    
    
# Suppress/hide the warnings from NumPy
np.seterr(invalid='ignore')

# Load the scalers
x_scaler = load(ROUTE_NN_MODEL + 'x_scaler.joblib')
y_scaler = load(ROUTE_NN_MODEL + 'y_scaler.joblib')
   
for pass_number in range(TOTAL_NUMBER_PASSES):
    for load_inc in range(TOTAL_LOAD_INCREMENTS):
        for i in range(TOTAL_ITERATIONS):
            print("\n \n Iteration number: ", i, "\n \n" )
            # Simulation A (load driven)
            # Create the flag for simulation A
            flag = 'simulationA'
            # Serialise the flag
            serialiseWordOrList(flag, ROUTE_NN_MODEL, "flag")
            # Run simulation A
            terminal("cd " + ROUTE_NN_MODEL + " && ./Allrun")
            
            #Copy DExpected to load_inc + 1 results
            terminal("cd " + ROUTE_THEORETICAL_MODEL + str(load_inc + 1) 
                     + "/ && cp D ../../../" + ROUTE_NN_MODEL 
                     + str(load_inc + 1) + "/DExpected")
            #Copy sigmaExpected to load_inc + 1 results
            terminal("cd " + ROUTE_THEORETICAL_MODEL + str(load_inc + 1) 
                     + "/ && cp sigma ../../../" + ROUTE_NN_MODEL 
                     + str(load_inc + 1) + "/sigmaExpected")
            #Copy epsilonExpected to load_inc + 1 results
            terminal("cd " + ROUTE_THEORETICAL_MODEL + str(load_inc + 1) 
                     + "/ && cp epsilon ../../../" + ROUTE_NN_MODEL 
                     + str(load_inc + 1) + "/epsilonExpected")
                   
            # With the fieldsManipulator routine and the field copied in the
            # previous lines, create Python files for those field (serialise 
            #those fields)
            terminal("cd NNBased \n cd pythonNNBasePlateHole \n " + 
                     " fieldsManipulator") 

            #Copy the results to a folder 'timestep + a'               
            terminal("cd " + ROUTE_NN_MODEL + " && cp -r " + 
                     str(load_inc + 1)  + " " + str(load_inc + 1) + "a" )    

            #Copy DExpected from load_inc + 1 to folder (timestep - 1)
            #(this is needed to set the known displacementes at the boundaries)
            if (i == 0):
                terminal("cd " + ROUTE_THEORETICAL_MODEL + str(load_inc + 1) + 
                         "/ && cp D ../../../" + ROUTE_NN_MODEL + 
                         str(load_inc) +"/DExpected")  
            
            # Remove results from Simulation A, to be able to run simulation B
            terminal("cd " + ROUTE_NN_MODEL + " && rm -r " 
                     + str(load_inc + 1) + " log* ")
            
            # Copy the load-driven D BC file to the D_Original file, so that
            # the load-driven simulation can be easily run later            
            terminal("cd " +  ROUTE_NN_MODEL + str(load_inc) + 
                     "/ && cp D D_original")
           
            # Set the known displacements at the boundary
            # Redefine the flag
            flag = 'BOUNDARY_CHANGE'
            # Serialise the new flag
            serialiseWordOrList(flag, ROUTE_NN_MODEL, "flag")  

            # Execute fieldsManipulator. That routine will see that the flag is
            # 'BOUNDARY_CHANGE' and will set the values at the boundaries as 
            # those given by displacement results from simulation A
            terminal("cd NNBased \n cd pythonNNBasePlateHole \n" + 
                     " fieldsManipulator")  
            
            if(load_inc == 0):
                #Change the values in the BC File to "Fixed Values". This also
                #eliminates 'gradient: uniform (0 0 0)
                change_bc_from_traction_to_d(ROUTE_NN_MODEL + 
                                             str(load_inc), True)
            
            else:
                #Change the values in the BC File to "Fixed Values"
                change_bc_from_traction_to_d(ROUTE_NN_MODEL + 
                                             str(load_inc), False)   

            # Simulation B (displacement driven)
            flag = 'simulationB'
            # Serialise the flag
            serialiseWordOrList(flag, ROUTE_NN_MODEL, "flag")  
            #Execute simulation B            
            terminal("cd " + ROUTE_NN_MODEL + " && ./Allrun")


            #Copy DExpected from load_inc + 1 to folder (timestep - 1)
            #(this is needed to serialise the fields to Python files)
            terminal("cd " + ROUTE_THEORETICAL_MODEL + str(load_inc + 1) 
                     + "/ && cp D ../../../" + ROUTE_NN_MODEL 
                     + str(load_inc + 1) + "/DExpected")
            #Copy sigmaExpected to load_inc + 1 results
            terminal("cd " + ROUTE_THEORETICAL_MODEL + str(load_inc + 1) 
                     + "/ && cp sigma ../../../" + ROUTE_NN_MODEL 
                     + str(load_inc + 1) + "/sigmaExpected")        
            #Copy epsilonExpected to load_inc + 1 results
            terminal("cd " + ROUTE_THEORETICAL_MODEL + str(load_inc + 1) 
                     + "/ && cp epsilon ../../../" + ROUTE_NN_MODEL 
                     + str(load_inc + 1) + "/epsilonExpected")

            # Execute fieldsManipulator. This will serialise fields to Python
            # files
            terminal("cd NNBased \n cd pythonNNBasePlateHole \n" + 
                     " fieldsManipulator")           
            
            #Copy the results to a folder 'timestep + B'  
            terminal("cd " + ROUTE_NN_MODEL + " && cp -r " + str(load_inc + 1) 
                     + " " + str(load_inc + 1) + "B" )
        
            # Load several fields
            #Load sigma from simul A
            sigma = deserialise(ROUTE_NN_MODEL  + str(load_inc + 1) + "a" 
                                + "/", "sigma")
            #Load epsilon from simul B
            epsilon = deserialise(ROUTE_NN_MODEL + str(load_inc + 1) + "B" 
                                  + "/", "epsilon")              
            #Load D from simul A
            D_simul_A = deserialise(ROUTE_NN_MODEL + str(load_inc + 1) + "a" 
                                    + "/", "D")  
            #Load D from simul B
            D_simul_B = deserialise(ROUTE_NN_MODEL + str(load_inc + 1) + "B" 
                                    + "/", "D")    
            
            #Load unused epsilon and sigma fields
            sigma_simul_B = deserialise(ROUTE_NN_MODEL  + str(load_inc + 1) 
                                        + "B" +  "/", "sigma")
        
            #Load epsilon from simul B
            epsilon_simul_A = deserialise(ROUTE_NN_MODEL + str(load_inc + 1) 
                                          + "a" +  "/", "epsilon")   
    
            #Deserialise expected fields
            if (i == 0):
                sigma_expected = deserialise(ROUTE_NN_MODEL  
                                             + str(load_inc + 1) 
                                             + "a" +  "/", "sigmaExpected")
            
                #Load epsilon from simul B
                epsilon_expected = deserialise(ROUTE_NN_MODEL 
                                               + str(load_inc + 1) 
                                               + "B" +  "/", "epsilonExpected")    
                
                D_expected = deserialise(ROUTE_NN_MODEL  + str(load_inc + 1) 
                                             + "a" +  "/", "DExpected")
                
            # Save some fields to plot results 
            if (pass_number == 0 and i == 0):
                serialise(sigma, ROUTE_NN_MODEL + "Results/", 
                          "sigma_simul_A_OriginalModel_LoadIncNum" + 
                          str(load_inc))
                serialise(epsilon, ROUTE_NN_MODEL + "Results/", 
                          "epsilon_simul_B_OriginalModel_LoadIncNum" + 
                          str(load_inc))
                serialise(sigma_simul_B, ROUTE_NN_MODEL + "Results/", 
                          "sigma_simul_B_OriginalModel_LoadIncNum" + 
                          str(load_inc))
                serialise(epsilon_simul_A, ROUTE_NN_MODEL + "Results/", 
                          "epsilon_simul_A_OriginalModel_LoadIncNum" + 
                          str(load_inc))
                serialise(D_simul_B, ROUTE_NN_MODEL + "Results/", 
                          "D_simul_B_OriginalModel_LoadIncNum" + 
                          str(load_inc))
                serialise(D_simul_A, ROUTE_NN_MODEL + "Results/", 
                          "D_simul_A_OriginalModel_LoadIncNum" + 
                          str(load_inc))

    
            if (pass_number == 0 and load_inc == 0 and i == 0):
                master_x_training_set = np.copy(original_x_training_set)
                master_y_training_set = np.copy(original_y_training_set)
                  
            else:
                if (i == 0):
                    master_x_training_set = deserialise(ROUTE_NN_MODEL, 
                                                   "master_x_training_set")

                    master_y_training_set = deserialise(ROUTE_NN_MODEL, 
                                                   "master_y_training_set")                

            # Create x_training_set and y_training_set
            if (pass_number == 0):
                x_training_set = np.concatenate((master_x_training_set, 
                                                 epsilon), axis=0)
                y_training_set = np.concatenate((master_y_training_set, 
                                                 sigma), axis=0)
                              
            else:   
                if (sigma.shape[0] * SETS_IN_MOVING_WINDOW > \
                    master_x_training_set.shape[0]):
                    x_training_set = np.concatenate((master_x_training_set, 
                                                     epsilon), axis=0)
                    y_training_set = np.concatenate((master_y_training_set, 
                                                     sigma), axis=0)
                    
                else:
                    x_training_set = np.concatenate(
                        (master_x_training_set[
                            (-1* SETS_IN_MOVING_WINDOW * 
                             epsilon.shape[0]):, :], epsilon), axis=0)
                    y_training_set = np.concatenate(
                        (master_y_training_set[
                            (-1* SETS_IN_MOVING_WINDOW * 
                             sigma.shape[0]):, :], sigma), axis=0)

            # Scale epsilon and sigma using the updated scaler 
            epsilon_scaled = x_scaler.transform(x_training_set)
            sigma_scaled = y_scaler.transform(y_training_set)          
          
            # Update the ML model 
            history = ML_model.fit(epsilon_scaled[:, np.newaxis, :], 
                                    sigma_scaled[:, np.newaxis , :], 
                                    epochs = NUMBER_OF_EPOCHS_OUTERLOOP, 
                                    validation_split = 0.02)
            
            # Save the new linear model 
            ML_model.save(ROUTE_NN_MODEL + "ML_model.h5")

            #Calculate some quantities to evaluate convergence
            diff = (mae(D_simul_A, D_expected).numpy()) / \
                (np.sum(D_expected * D_expected))**0.5
     
        
            if (i!=(TOTAL_ITERATIONS-1) and (diff > TOL_LOCAL_ITER)):
                terminal("cd " + ROUTE_NN_MODEL + "&& rm -r " + 
                         str(int(load_inc + 1)) + " log* *a *B")
                terminal("cd " +  ROUTE_NN_MODEL + str(load_inc) + 
                          "/ && cp D_original ./D") 
                print(" \n \n \n MAE(expected vs calculated D) at this " \
                      + "iteration is ", diff, "\n \n \n")        

            else:
                route_store_results = ROUTE_NN_MODEL + "loadpass" + \
                    str(pass_number + 1) + "_loadInc" +  str(load_inc + 1)
                
                terminal( "mkdir -p " + route_store_results)

                master_x_training_set = np.concatenate(
                    (master_x_training_set, epsilon), axis=0)
                master_y_training_set = np.concatenate(
                    (master_y_training_set, sigma), axis=0)
                
                serialise(x_training_set, ROUTE_NN_MODEL, "x_training_set")
                
                serialise(y_training_set, ROUTE_NN_MODEL, "y_training_set")
                
                serialise(master_x_training_set, ROUTE_NN_MODEL, 
                          "master_x_training_set")
                
                serialise(master_y_training_set, ROUTE_NN_MODEL, 
                          "master_y_training_set")
        
                terminal("cp " + ROUTE_NN_MODEL + "*.joblib " +" " + 
                         route_store_results + "/")
                terminal("cp -r " + ROUTE_NN_MODEL + "*a " + ROUTE_NN_MODEL 
                         + "*B " + " " + route_store_results)
                terminal("cd " + ROUTE_NN_MODEL + " && rm -r *B")

                terminal( "mv " + ROUTE_NN_MODEL + "log* " +  " " + 
                         route_store_results + "/")  
                terminal( "mv " + ROUTE_NN_MODEL + "*pkl " +  " " + 
                         route_store_results + "/")  
                terminal("cd " +  ROUTE_NN_MODEL + str(load_inc) + 
                         "/ && cp D_original ./D")
                terminal("cd " +  ROUTE_NN_MODEL + str(load_inc) + 
                         "/ && rm D_original DExpected")               
                terminal("cd " + ROUTE_NN_MODEL + " && rm -r " + 
                         str(load_inc + 1))
                terminal("cd " + ROUTE_NN_MODEL + " && mv " + str(load_inc + 1)
                         + "a" + " " + str(load_inc + 1))       
                break
            
    terminal("cd " + ROUTE_NN_MODEL + " && ./Allclean")
end = time.time()
print("Calculation is finished.")

with open('./report.txt', 'a') as f:
    f.write("Total calculation time: " + str(end-start) + " (s) \n or \n " + 
            str((end-start)/3600) + "(h)")
    f.close()
