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
#  Auxiliary functions required by the self-simulation algorithm.

# Authors
#  Simon A. Rodriguez, UCD. All rights reserved
#  Philip Cardiff, UCD. All rights reserved

import os
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle


def create_linear_reg(x_train, y_train):
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)

    return lin_reg


def serialise(object_to_serialise, route_resulting_file, name_resulting_file):
    #This function exports to npy

    with open(route_resulting_file + name_resulting_file + ".npy", 'wb') as f:
        np.save(f, object_to_serialise)
    f.close()
    

def serialise_word_or_list(object_to_serialise, route_resulting_file, 
                           name_resulting_file):
    #This function exports to pkl
    
    with open(route_resulting_file + name_resulting_file + ".pkl", 'wb') as f:
        pickle.dump(object_to_serialise, f)
    f.close()
    

def deserialise(route_to_file, name_file):
    #This function imports a .npy file
    with open(route_to_file  +  name_file + '.npy', 'rb') as f:
        temp = np.load(f, allow_pickle=True)
    f.close()

    return temp


def terminal(command):
    os.system(command)


def change_bc_from_traction_to_d(address, remove_gradient):
    # This function replaces the traction BC with known-displacement BC

    with open(address + "/D", "r") as f:
        lines = f.readlines()
    f.close()
    
    def remove_gradient_word(lines, index):
        dum = True
        i = 0
        while(dum):
            if ("gradient" in lines[index + i]):
                lines[index + i] = "\n"
                dum = False
                break
            else:
                i = i + 1
        return lines         
    
    text_to_replace =  "        relaxationFactor 1;\n"
    text_replacing_old = "        type            fixedValue;\n"

    for line in lines:
        if 'right' in line:
            index = lines.index(line)     
    if (remove_gradient):
        lines = remove_gradient_word(lines, index)
    for i in range(9):
        lines.pop(index+2)
    lines[index + 2] = lines[index + 2].replace(text_to_replace, 
                                                text_replacing_old)
    with open(address + "/D", "w") as f:  
        for line in lines:
            f.write(line)
    f.close()

    for line in lines:
        if 'down' in line:
            index = lines.index(line)
    if (remove_gradient):
        lines = remove_gradient_word(lines, index)
    for i in range(5):
        lines.pop(index+2)
    lines[index + 2] = lines[index + 2].replace(text_to_replace, 
                                                text_replacing_old)

    with open(address + "/D", "w") as f:  
        for line in lines:
            f.write(line)
    f.close()

    for line in lines:
        if 'up' in line:
            index = lines.index(line)
    if (remove_gradient):
        lines = remove_gradient_word(lines, index)
    for i in range(5):
        lines.pop(index+2)
    lines[index + 2] = lines[index + 2].replace(text_to_replace, 
                                                text_replacing_old)
    with open(address + "/D", "w") as f:  
        for line in lines:
            f.write(line)
    f.close()
    
    for line in lines:
        if 'hole' in line:
            index = lines.index(line)
    if (remove_gradient):
        lines = remove_gradient_word(lines, index)
    for i in range(5):
        lines.pop(index+2)
    lines[index + 2] = lines[index + 2].replace(text_to_replace, 
                                                text_replacing_old)
    with open(address + "/D", "w") as f:  
        for line in lines:
            f.write(line)
    f.close()