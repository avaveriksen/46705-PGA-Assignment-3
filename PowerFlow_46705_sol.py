#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
46705 - Power Grid Anlysis
This file contains the definitions of the functions needed to 
carry out Power Flow calculations in python.
"""

import numpy as np

# 1. the PowerFlowNewton() function
def PowerFlowNewton(Ybus,Sbus,V0,pv_index,pq_index,max_iter,err_tol,print_progress=True): #print progress toggles additional console output
    # Initialization of flag, counter and voltage
    success = 0
    n = 0
    V = V0
    if print_progress:
        print('  iteration      maximum P & Q mismatch (pu)')    
        print('  ---------      -----------------------')
    # Determine mismatch between initial guess and and specified value for P and Q
    F = calculate_F(Ybus,Sbus,V,pv_index,pq_index)
    # Check if the desired tolerance is reached
    success = CheckTolerance(F,n,err_tol,print_progress) 
    # Start Newton iterations
    while (not success) and (n < max_iter):
        # Update counter
        n += 1
        # Generate the Jacobian matrix        
        J_dS_dVm,J_dS_dTheta = generate_Derivatives(Ybus,V)
        J = generate_Jacobian(J_dS_dVm,J_dS_dTheta,pv_index,pq_index)
        # Compute step
        try:
            dx = np.linalg.solve(J, F)
        except Exception as e:
            if print_progress:
                print(f"Power Flow Newton: linalg.solve threw an exception: {e}")

        # Update voltages and check if tolerance is now reached
        V = Update_Voltages(dx,V,pv_index,pq_index)
        F = calculate_F(Ybus,Sbus,V,pv_index,pq_index)
        success = CheckTolerance(F,n,err_tol,print_progress)
    
    if success:
        if print_progress:
            print('The Newton Rapson Power Flow Converged in %d iterations!\n\n' % (n,))
    else:
        if print_progress:
            print('!!! No Convergence !!!\n Stopped after %d iterations without solution...' % (n,)) 
            
    return V,success,n
    

# 2. the calculate_F() function
def calculate_F(Ybus,Sbus,V,pv_index,pq_index):
    # Calculate the difference between current value and real value
    Delta_S = Sbus - V * (Ybus.dot(V)).conj()
    Delta_P = np.real(Delta_S)
    Delta_Q = np.imag(Delta_S)
    # Construct F to have the right structure of the indexes of different busses.
    F = np.concatenate((Delta_P[pv_index],Delta_P[pq_index], Delta_Q[pq_index]),axis=0)
    return F


# 3. the CheckTolerance() function
def CheckTolerance(F,n,err_tol,print_progress=True):
    # Check if maximum difference is smaller than tolerance
    normF = np.linalg.norm(F,np.inf)
    if normF < err_tol:
        success = 1
    else:
        success = 0
        
    if print_progress:    
        print(' %5d                %14.8f       ' %(n,normF))
    return success




# 4. the generate_Derivatives() function
def generate_Derivatives(Ybus,V):
    # Generate the derivatives used in the Jacobian
    J_ds_dVm = np.diag(V/np.absolute(V)).dot(np.diag((Ybus.dot(V)).conj())) + \
    np.diag(V).dot(Ybus.dot(np.diag(V/np.absolute(V))).conj())    
    J_dS_dTheta = 1j*np.diag(V).dot((np.diag(Ybus.dot(V))-Ybus.dot(np.diag(V))).conj())
    return J_ds_dVm,J_dS_dTheta


# 5. the generate_Jacobian() function
def generate_Jacobian(J_dS_dVm,J_dS_dTheta,pv_index,pq_index):
    # Generate Jacobian from dericatives of S    
    pvpq_ind = np.append(pv_index,pq_index)
    # Construct Jacobian from the indexes of the different nodes
    J_11 = np.real(J_dS_dTheta[np.ix_(pvpq_ind,pvpq_ind)])    
    J_12 = np.real(J_dS_dVm[np.ix_(pvpq_ind,pq_index)])
    J_21 = np.imag(J_dS_dTheta[np.ix_(pq_index,pvpq_ind)])
    J_22 = np.imag(J_dS_dVm[np.ix_(pq_index,pq_index)])
    J    = np.block([[J_11,J_12],[J_21,J_22]])  
    
    return J

    

# 6. the Update_Voltages() function
def Update_Voltages(dx,V,pv_index,pq_index):
    
    # Update the voltage with the new step
    Theta = np.angle(V)
    Vm = np.absolute(V)
         
    N1 = 0;  N2 = len(pv_index)
    N3 = N2; N4 = N3 + len(pq_index)
    N5 = N4; N6 = N5 + len(pq_index)

    if pv_index.size != 0:
        Theta[pv_index] += dx[N1:N2]
        
    if pq_index.size != 0:
        Theta[pq_index] += dx[N3:N4]
        Vm[pq_index] += dx[N5:N6]
    
    V = Vm * np.exp(1j*Theta)
    
    return V


####################################################
#  Displaying the results in the terminal window   #
####################################################
            
        
        
def DisplayResults_and_loading(V,lnd):    
    Ybus=lnd.Ybus; Y_from=lnd.Y_fr; Y_to=lnd.Y_to; br_f=lnd.br_f; br_t=lnd.br_t; 
    buscode=lnd.buscode; SLD=lnd.S_LD; MVA_base=lnd.MVA_base 
    br_MVA = lnd.br_MVA
    # Power to each bus by branch
    I_to = Y_to.dot(V) 
    S_to = V[br_t]*I_to.conj()
    to_loading = np.abs(S_to*MVA_base/br_MVA) *100
    P_to = np.real(S_to)
    Q_to = np.imag(S_to)    
    # Power from each bus by branch
    I_from = Y_from.dot(V)
    S_from = V[br_f]*I_from.conj()
    fr_loading = np.abs(S_from*MVA_base/br_MVA) * 100
    P_from = np.real(S_from)
    Q_from = np.imag(S_from)    
    # Power injected at each bus
    S_inj = V*(Ybus.dot(V)).conj()
    P_inj = np.real(S_inj)
    Q_inj = np.imag(S_inj)
    # Power injected by generators
    S_gen = S_inj + SLD                         # the generator injections compensated for loads at the same bus
    P_gen = np.real(S_gen)
    Q_gen = np.imag(S_gen)
    # Voltage magnitude and angle in degrees
    Vabs = np.absolute(V)
    Vang = np.angle(V, deg=True)        
    # Print bus results
    print(' ')
    print('=====================================================================================')
    print('| System MVA Base:  {: >9.1f} MVA                                                   |'.format(lnd.MVA_base))
    print('=====================================================================================')
    print('| Bus results                                                                       |')
    print('=====================================================================================')    
    print('  Bus       Bus          Voltage             Generation                  Load      ')    
    print('   #       Label    Mag(pu)  Ang(deg)   P (pu)  Q(pu)   loading     P (pu)  Q(pu) ')
    print(' -----   ---------  -------  -------  -------  --------  -------   -------  ------- ')

    k = 0; bus=k+1
    str_PQ =   '{:^7} {:^10} {: >7.3f}  {: >7.2f}   {:^7}   {:^7} {:^7}   {: >7.3f}   {: >7.3f}'
    str_PV =   '{:^7} {:^10} {: >7.3f}  {: >7.2f}  {: >7.3f}   {: >7.3f}  {: >7.2f}%   {:^7}   {:^7}'
    str_both = '{:^7} {:^10} {: >7.3f}  {: >7.2f}  {: >7.3f}   {: >7.3f}  {: >7.2f}%  {: >7.3f}   {: >7.3f}'
    for k in range(0,len(buscode)):
        
        bus = lnd.ind_to_bus[k]
        label = lnd.bus_labels[k]
        if buscode[k] == 1:
            print(str_PQ.format(bus,label,Vabs[k],Vang[k],'-','-','-',-P_inj[k],-Q_inj[k]))
#            print(' %2d   %7.3f  %7.2f     -        -      %7.3f  %7.3f' % (bus,Vabs[k],Vang[k],-P_inj[k],-Q_inj[k]))
        elif np.abs(SLD[k]) == 0.0: #if there is no load on the generator bus -> display gen results only
            if buscode[k] == 3: #if reference bus---
                bus_star = '{:}{:}{:}'.format('*',bus,'*')
                print(str_PV.format(bus_star,label,Vabs[k],Vang[k],P_inj[k],Q_inj[k],np.abs(S_inj[k])*lnd.MVA_base/lnd.Gen_MVA[k]*100.0,'-','-'))
            else:
                print(str_PV.format(bus,label,Vabs[k],Vang[k],P_inj[k],Q_inj[k],np.abs(S_inj[k])*lnd.MVA_base/lnd.Gen_MVA[k]*100.0,'-','-'))
        else: #if there is also load on the generator bus -> split load and generation data
            PLD = SLD[k].real; QLD = SLD[k].imag;
            if buscode[k] == 3: #if reference bus---
                bus_star = '{:}{:}{:}'.format('*',bus,'*')
                print(str_both.format(bus_star,label,Vabs[k],Vang[k],P_gen[k],Q_gen[k], np.abs(S_gen[k])*lnd.MVA_base//lnd.Gen_MVA[k]*100.0,PLD,QLD))
            else:
                print(str_both.format(bus,label,Vabs[k],Vang[k],P_gen[k],Q_gen[k], np.abs(S_gen[k])*lnd.MVA_base/lnd.Gen_MVA[k]*100.0, PLD,QLD))
            
    # Print branch results        
    print(' ')
    print(' ==============================================================================')
    print('| Branch flow                                                                 |')
    print(' ==============================================================================')    
    print('|Branch  From   To        From Bus Injection              To Bus Injection    |')    
    print('|  #     Bus    Bus    P (pu)   Q (pu)  loading        P (pu)   Q (pu) loading|')
    print(' -----  -----  -----  -------  -------- -------        ------- ------- -------')      
    
    for h in range(0,len(S_to)):
        branch = h + 1
        str_ = '  {:^3} {:>6} {:>6}    {: >6.3f}  {: >6.3f}   {: >6.2f}%        {: >6.3f}  {: >6.3f}  {: >6.2f}%'
        print(str_.format(branch,lnd.ind_to_bus[br_f[h]],lnd.ind_to_bus[br_t[h]],P_from[h],Q_from[h],fr_loading[h]  ,P_to[h],Q_to[h],to_loading[h] ))
    return

def System_violations(V,lnd):
    # Inputs: 
    # V = results from the load flow    
    # lnd = the LoadNetworkData object for easy access to other model data
    
    #store variables as more convenient names
    br_f=lnd.br_f; br_t=lnd.br_t;   #from and to branch indices 
    ind_to_bus=lnd.ind_to_bus;      # the ind_to_bus mapping object
    bus_to_ind=lnd.bus_to_ind;      # the bus_to_ind mapping object    
    br_MVA = lnd.br_MVA             # object containing the MVA ratings of the branches 
    br_id  = lnd.br_id              # (you have to update LoadNetworkData for this)
    Ybus = lnd.Ybus;                # Ybus = the bus admittance matrix used in the load flow
    Y_from = lnd.Y_fr; Y_to = lnd.Y_to;  # Y_from,Y_to = tha admittance matrices used to determine the branch flows

    #line flows and generator injections:
    S_to = V[br_t]*(Y_to.dot(V)).conj()         # the flow in the to end..
    S_from = V[br_f]*(Y_from.dot(V)).conj()     # the flow in the from end
    S_inj = V*(Ybus.dot(V)).conj()              # the injected power 
    SLD=lnd.S_LD                                # The defined loads on the PQ busses
    S_gen = S_inj + SLD                         # the generator injections compensated for loads at the same bus
    
    # Generator limit violations:    
    gen_ind = np.where(lnd.buscode>1)[0] #find the generator nodes
    gen_str = ' --> Generator at bus {:} overloaded ({})'
    violations = []
    
    for g in gen_ind:
        g_loading = np.abs(S_gen[g])*lnd.MVA_base/lnd.Gen_MVA[g]
        pgen = np.real(S_gen[g])*lnd.MVA_base
        qgen = np.imag(S_gen[g])*lnd.MVA_base
        bus_nr = lnd.ind_to_bus[g]
        pmax = lnd.p_gen_max[g]*lnd.MVA_base
        qmax = lnd.q_gen_max[g]*lnd.MVA_base
        qmin = lnd.q_gen_min[g]*lnd.MVA_base
        if g_loading > 1.0001: #if gen violation, record it
            errtype = 'Loading = {:.2f}  [%]'.format(g_loading*100)
            violations.append(gen_str.format(bus_nr,errtype))
        if pgen / pmax > 1.0001:
            errtype = 'Pgen = {:.2f} > {:.2f} [MW]'.format(pgen,pmax)
            violations.append(gen_str.format(bus_nr,errtype))
        if qgen > qmax:
            errtype = 'Qgen = {:.2f}  > {:.2f} [MVAr]'.format(qgen,qmax)
            violations.append(gen_str.format(bus_nr,errtype))
        if qgen < qmin:
            errtype = 'Qgen = {:.2f}  < {:.2f} [MVAr]'.format(qgen,qmin)
            violations.append(gen_str.format(bus_nr,errtype))
                  
    # Branch limit violations:    
    br_str = ' --> Branch #{:} - from bus {:} to bus {:} (ID:{:}) overloaded (fr:{:.2f}%, to:{:.2f}%)'
    for i,fr,to,MVA in zip(range(len(br_f)),br_f,br_t,br_MVA):
        fr_loading = np.abs(S_from[i]*lnd.MVA_base/MVA)    
        to_loading = np.abs(S_to[i]*lnd.MVA_base/MVA)    
        if ((fr_loading > 1)|(to_loading > 1)):
            br_nr = i+1
            violations.append(br_str.format(br_nr,ind_to_bus[fr], ind_to_bus[to], br_id[i] ,fr_loading*100,to_loading*100))
    
    # Bus voltage limit violations
    ind_low = np.where(np.abs(V)< lnd.v_min)
    ind_high = np.where(np.abs(V)> lnd.v_max)

    V_h_str = ''
    for ind_h in ind_high[0]:
        V_h_str += '\n      Bus {:<4} ->  {:.3f} pu > V_max: {:.3f} pu'.format(lnd.ind_to_bus[ind_h],np.abs(V[ind_h]),lnd.v_max[ind_h])    

    V_l_str = ''
    for ind_l in ind_low[0]:
        V_l_str += '\n      Bus {:<4} ->  {:.3f} pu < V_min: {:.3f} pu'.format(lnd.ind_to_bus[ind_l],np.abs(V[ind_l]),lnd.v_min[ind_l])    
   
    if len(ind_high[0])>0:
        volt_str = ' --> Voltages too high at ...' + V_h_str
        violations.append(volt_str)
    
    if len(ind_low[0])>0:
        volt_str = ' --> Voltages too low at ...' + V_l_str
        violations.append(volt_str)
    
    return violations
