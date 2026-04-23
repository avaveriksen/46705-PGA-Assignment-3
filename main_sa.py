# -*- coding: utf-8 -*-
"""
Security Analysis Script
"""

import numpy as np
import PowerFlow_46705_sol as pf  # import Power Flow functions
import LoadNetworkData_sol as lnd # load the network data to global variables
max_iter = 30   # Iteration settings
err_tol = 1e-3
# Load the Network data ...
filename = "./TestSystem.txt"


lnd.LoadNetworkData(filename) # makes Ybus available as lnd.Ybus etc.

def System_violations(V,Ybus,Y_from,Y_to,lnd):
    # Inputs: 
    # V = results from the load flow
    # Ybus = the bus admittance matrix used in the load flow
    # Y_from,Y_to = tha admittance matrices used to determine the branch flows
    # lnd = the LoadNetworkData object for easy access to other model data
    
    #store variables as more convenient names
    br_f=lnd.br_f; br_t=lnd.br_t;   #from and to branch indices 
    ind_to_bus=lnd.ind_to_bus;      # the ind_to_bus mapping object
    bus_to_ind=lnd.bus_to_ind;      # the bus_to_ind mapping object    
    br_MVA = lnd.br_MVA             # object containing the MVA ratings of the branches 
    br_id  = lnd.br_id              # (you have to update LoadNetworkData for this) 


    #line flows and generators injection....
    S_to = V[br_t]*(Y_to.dot(V)).conj()         # the flow in the to end..
    S_from = V[br_f]*(Y_from.dot(V)).conj()     # the flow in the from end
    S_inj = V*(Ybus.dot(V)).conj()              # the injected power 
    SLD=lnd.S_LD                                # The defined loads on the PQ busses
    S_gen = S_inj + SLD                         # the generator arrays
    
    gen_ind = np.where(lnd.buscode>1)[0] #find the generator nodes
    gen_str = ' --> Generator at bus {:} overloaded ({})'
    violations = []
    
   ##################################################################################
   # YOUR CODE COMES HERE:
   # 1. Check flow in all branches (both ends) and report if limits are violated
   # 2. Check output of all generators and see if limits are exceeded
   # 3. Check voltages on all busses and see if it remains within pre-defined bounds
   ##################################################################################
    
    return violations
    
    


def apply_contingency_to_Y_matrices(Ybus,Yfr,Yto,fr_ind,to_ind,br_ind,Ybr_mat):
    # input:
    # The original admittance matirces: Ybus,Yfr,Yto
    # The from and to end indices for the branch (fr_ind, to_ind)
    # The indice for where the branch is in the branch list (br_ind)
    # The 2x2 admittance matrix for the branch Ybr_mat
    ##########################################################
    # This is important, you must copy the original matrices
    Ybus_mod = Ybus . copy ( ) # otherwise you will change the original Ybus matrix
    Yfr_mod = Yfr . copy ( ) # when ever you make changes to Ybus_mod
    Yto_mod = Yto . copy ( ) # using the .copy() function avoids this
    ##################################################################################
    # YOUR CODE COMES HERE:
    # 1. Remove the branch from the Ybus_mod matrix
    ind_Ybus = [fr_ind, to_ind]                           # array with the two Ybus indexes
    Ybus_mod_sliceix = np.ix_(ind_Ybus, ind_Ybus)   # indexes the slice of Ybus_mod we need
    Ybus_mod[Ybus_mod_sliceix] -= Ybr_mat                  # subtract the branch admittace from Ybus to remove branch
    # 2. Remove the branch from the Yto and Yfr matrices
    Yfr_mod[br_ind, fr_ind] = 0
    Yfr_mod[br_ind, to_ind] = 0
    Yto_mod[br_ind, fr_ind] = 0
    Yto_mod[br_ind, to_ind] = 0
    ##################################################################################
    return Ybus_mod,Yfr_mod,Yto_mod






#%%
##############################################################
#   Part I:                                                  #
#   Study the base case and display results (with % loading) #
#                                                            #
##############################################################

# Carry out load flow of the base case.....
V,success,n = pf.PowerFlowNewton(lnd.Ybus,lnd.Sbus,lnd.V0,lnd.pv_index,lnd.pq_index,max_iter,err_tol)
if success: # Display results if the power flow analysis converged
    pf.DisplayResults_and_loading(V,lnd) 




#%% 
##############################################################
#    Part II:                                                #
#    Simplified contingency analysis (only branch outages)   #
#                                                            #
##############################################################

print('*'*50)
print('*             Contingency Analysis               *')
print('*'*50)


for i in range(len(lnd.br_f)): #sweep over branches
    fr_ind = lnd.br_f[i]
    to_ind = lnd.br_t[i]
    br_ind = i
    Ybr_mat = lnd.br_Ymat[i]
    Ybus_mod,Yfr_mod,Yto_mod = apply_contingency_to_Y_matrices(lnd.Ybus,lnd.Y_fr,lnd.Y_to,fr_ind,to_ind,br_ind,Ybr_mat)
    #        
    str_status = '---------------------------------------------------------------\nTripping of branch {:} (bus {:} - bus {:})'.format(i+1,lnd.ind_to_bus[fr_ind],lnd.ind_to_bus[to_ind])
        
    # Carry out load flow of the base case.....
    try:
        V,success,n = pf.PowerFlowNewton(Ybus_mod,lnd.Sbus,lnd.V0,lnd.pv_index,lnd.pq_index,max_iter,err_tol,print_progress=False)
    except:
        str_ = ' --> Load Flow error (Jacobian) when branch {:} (bus {:} - bus {:}) is tripped'.format(i+1,lnd.ind_to_bus[fr_ind],lnd.ind_to_bus[to_ind])
        print(str_status+ ' [CONVERGENCE ISSUES!]')
        print(str_)
    else:  
        if success: # Display results if the power flow analysis converged
            violations = System_violations(V,Ybus_mod,Yfr_mod,Yto_mod,lnd)
            if not violations: #no violations, print status and move on
                print(str_status + ' [OK!]')
            else: # if violation, display them
                print(str_status + ' [Violations!]')
                for str_ in violations:
                    print(str_)
        else: #no convergence...
            str_ = ' --> No load-flow convergence when branch {:} (bus {:} - bus {:}) is tripped'.format(i+1,lnd.ind_to_bus[fr_ind],lnd.ind_to_bus[to_ind])
            print(str_status + ' [CONVERGENCE ISSUES!]')
            print(str_)
    