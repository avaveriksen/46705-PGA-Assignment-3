# -*- coding: utf-8 -*-
import numpy as np
import ReadNetworkData as rd

def LoadNetworkData(filename):
    global Ybus, Sbus, V0, buscode, ref, pq_index, pv_index, Y_fr, Y_to, br_f,br_t,br_v_ind,br_Y,S_LD,  \
           ind_to_bus, bus_to_ind, MVA_base, bus_labels, br_MVA, Gen_MVA, br_id, br_Ymat, bus_kv, \
           p_gen_max, q_gen_min, q_gen_max, v_min, v_max
    #read in the data from the file...
    bus_data,load_data,gen_data,line_data, tran_data,mva_base, bus_to_ind, ind_to_bus = \
    rd.read_network_data_from_file(filename)

    ######################################################################### 
    # Construct the Ybus matrix from elements in the line_data and trans_data
    # Construct the branch admittance matrices Y_fr and Y_to
    ##########################################################################
    MVA_base = mva_base   
    N = len(bus_data) #Number of buses
    M_lines = len(line_data)
    M_trans = len(tran_data)
    M_branches = M_lines + M_trans
    Ybus = np.zeros((N,N),dtype=complex)
    Gen_MVA = np.zeros(N) #keep track of generators MVA size  (bus indices used)
    p_gen_max = np.zeros(N)
    q_gen_max = np.zeros(N)
    q_gen_min = np.zeros(N)
    br_f = -np.ones(M_branches,dtype=int)
    br_t = -np.ones(M_branches,dtype=int)
    br_MVA = np.zeros(M_branches,dtype=float)
    br_id = [None]*M_branches
    br_Ymat = [None]*M_branches   
    Y_fr = np.zeros((M_branches,N),dtype=complex)
    Y_to = np.zeros((M_branches,N),dtype=complex)
      
    
    # Add lines to the Ybus
    for line,i in zip(line_data,range(len(line_data))):
        bus_fr, bus_to, id_, R, X, B, MVA_rate, X2, X0 = line #unpack
        ind_fr = bus_to_ind[bus_fr]    
        ind_to = bus_to_ind[bus_to] 
                
        # Formulate 2x2 component admittance matrix of lines
        Yps_mat = np.zeros((2,2),dtype=complex)
        Z_se = R + 1j*X; Y_se = 1/Z_se
        Y_sh_2 = 1j*B/2 
        Yps_mat[0,0] = Y_se + Y_sh_2;  Yps_mat[0,1] = -Y_se
        Yps_mat[1,0] = -Y_se;          Yps_mat[1,1] = Y_se + Y_sh_2
        
        # Update the bus admittance matrix
        ind_Ybus = np.array([ind_fr,ind_to])
        Ybus[np.ix_(ind_Ybus,ind_Ybus)] += Yps_mat
        
        # Update Branch admittance matrices
        Y_fr[i,ind_fr] =  Yps_mat[0,0]      
        Y_fr[i,ind_to] =  Yps_mat[0,1]
        Y_to[i,ind_to] =  Yps_mat[1,1]       
        Y_to[i,ind_fr] =  Yps_mat[1,0]        
        
        # keep track of branch data        
        br_f[i] = ind_fr
        br_t[i] = ind_to
        br_MVA[i] = MVA_rate
        br_id[i] = id_
        br_Ymat[i] = Yps_mat 
        
        
    # Add transformers to Ybus
    for line,i in zip(tran_data,range(len(line_data),M_branches)):
        bus_fr, bus_to, id_, R,X,n,ang1, MVA_rate,fr_co, to_co, X2, X0 = line #unpack
        ind_fr = bus_to_ind[bus_fr]  # get the matrix index corresponding to the bus    
        ind_to = bus_to_ind[bus_to]  # same here
                 
        # Formulate 2x2 component admittance matrix of transformers
        Yps_mat = np.zeros((2,2),dtype=complex)
        Zeq = R+1j*X; Yeq = 1/Zeq    # transformer impedance, pu on system base
        c = n*np.exp(1j*ang1/180*np.pi) # Complex pu. turns ratio
        Yps_mat[0,0] = Yeq/np.abs(c)**2;  Yps_mat[0,1] = -Yeq/c.conj()
        Yps_mat[1,0] = -Yeq/c;            Yps_mat[1,1] = Yeq
        
        # Update the bus admittance matrixs
        ind_Ybus = np.array([ind_fr,ind_to])
        Ybus[np.ix_(ind_Ybus,ind_Ybus)] += Yps_mat
        
        # Update Branch admittance matrices
        Y_fr[i,ind_fr] =  Yps_mat[0,0]      
        Y_fr[i,ind_to] =  Yps_mat[0,1]
        Y_to[i,ind_to] =  Yps_mat[1,1]       
        Y_to[i,ind_fr] =  Yps_mat[1,0]
        
        # keep track of branch data
        br_f[i] = ind_fr
        br_t[i] = ind_to
        br_MVA[i] = MVA_rate
        br_id[i] = id_
        br_Ymat[i] = Yps_mat 
        

    #create Sbus, V and other bus related arrays
    #Get the bus data
    bus_kv = []
    buscode = []
    bus_labels = []
    Vm = []
    theta = []
    v_min = []
    v_max = []
    for line in bus_data: #[bus_nr, label, vm_init, theta_init, buscode, kv_base, v_min, v_max]
        b_nr, label, v_init, theta_init, code, kv, v_low, v_high = line
        buscode.append(code)
        bus_labels.append(label)
        Vm.append(v_init)
        theta.append(theta_init)
        bus_kv.append(kv)
        v_min.append(v_low)
        v_max.append(v_high)
    
    buscode = np.array(buscode)
    bus_kv = np.array(bus_kv)
    Vm = np.array(Vm)
    theta = np.array(theta)
    v_min = np.array(v_min)
    v_max = np.array(v_max)
    V0 = Vm * np.exp(1j*np.deg2rad(theta))

    # Create the Sbus vector (bus injections) and load vector
    Sbus = np.zeros(N,dtype=complex)
    S_LD = np.zeros(N,dtype=complex)
        
    for line in load_data:
        bus_nr, PLD, QLD = line
        ind_nr = bus_to_ind[bus_nr]
        SLD_val =(PLD+1j*QLD)/MVA_base
        Sbus[ind_nr] += -SLD_val # load is a negative injection...
        S_LD[ind_nr] +=  SLD_val # Keep track of the loads
        
    for line in gen_data:
        bus_nr, MVA_size, p_gen, p_max, q_max, q_min,  X, X2, X0, Xn, ground = line
        ind_nr = bus_to_ind[bus_nr]
        Sinj = (p_gen)/MVA_base
        Sbus[ind_nr] += Sinj # gen is a positive injection...
        Gen_MVA[ind_nr] += MVA_size
        p_gen_max[ind_nr] += p_max/MVA_base
        q_gen_max[ind_nr] += q_max/MVA_base
        q_gen_min[ind_nr] += q_min/MVA_base

    
    pq_index = np.where(buscode== 1)[0]# Find indices for all PQ-busses 
    pv_index = np.where(buscode== 2)[0]# Find indices for all PV-busses
    ref = np.where(buscode== 3)[0] # Find index for ref bus
 
    return
    