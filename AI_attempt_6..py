# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 00:27:29 2020

@author: ashaa
"""
import numba
from numba import cuda, float32
import numpy as np
import math as m
from datetime import datetime
import csv
from imblearn.over_sampling import SMOTE

startTime = datetime.now()


TPB = 64

@cuda.jit
def matrixmultiply(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R):
    """CUDA kernel"""
    
    """TPB must be a multiple of 64 along with the array sizes"""
    
    """" A = data[i, j]
         B = weights for data --> layer2 size = (640, 3197)
         C = layer 2 size = (640)
         D = biases for data --> layer2 size = (640)
         E = layer 2 weights size = (640, 640)
         F = layer 3 (640)
         G = layer2 biases (640)
         H = layer3 weights (640, 640)
         I = layer 4 (640)
         J = layer 3 biases (640)
         K = layer 4 weights (640, 640)
         L = layer 5 (640)
         M = layer 4 biases (640)
         N = layer 5 weights (2, 640)
         O = final layer (2) *padded to 640
         P = layer 5 biases (2) *padded to 640
         Q = actual (1)
         R = extra array (640)
         """

    ##shared memory
    #data
    sA = cuda.shared.array(shape=(TPB), dtype=float32)
    
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    
    sW = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    sD = cuda.shared.array(shape=(TPB), dtype=float32)

    sL = cuda.shared.array(shape=(TPB), dtype=float32)
    

    x = cuda.grid(1)
    
    # #threadID
    tx = cuda.threadIdx.x

    # #blockID
    bx = cuda.blockIdx.x
    
    #number of blocks
    blck = cuda.gridDim.x
    
    # if x == 1 and bx == 3:
    #     set_trace()
    
    #multiply weights by first layer
    if x < C.shape[0]:
        tmp = 0.
        for i in range(int(A.shape[0] / TPB)):
            # Preload data into shared memory
            sA[tx] = A[tx+i*TPB]
        
            #loop through dim 1 of B
            for k in range(TPB):    
                sB[tx, k] = B[x, i*TPB+k]
            cuda.syncthreads()
            
            #perform calculation on loop through sA dim 0 and sB dim 1
            for j in range(TPB):
                tmp += sA[j] * sB[tx, j]
              
            cuda.syncthreads()
        cuda.syncthreads()   

        C[x] = tmp
        
    cuda.syncthreads()
    
    
    
    #add biases layer 1
    

    
    if x < C.shape[0]:
        #preload biases and layer sections
        sD[tx] = D[x]
        sL[tx] = C[x]
        cuda.syncthreads()
        
        #add bias to layer and replace original layer
        # C[x] = sD[tx] + sL[tx]
        
        tmp = 1/(1+m.exp(-1*(sD[tx]+sL[tx]))) 
        C[x] = tmp
    cuda.syncthreads()
    
    
    
    ##layer 2 multiply by weights
    #layer 2
    #layer 2 weights
    
    
    # if x < F.shape[0]:
    #     tmp = 0.
    #     for i in range(int(C.shape[0]/TPB)):
            
    #         #preload memory
    #         sL[tx] = C[tx+i*TPB]
            
    #         for k in range(TPB):    
    #             sW[tx, k] = E[x, i*TPB+k]
            
    #         cuda.syncthreads()
            
    #         #perform multiplication
    #         for j in range(TPB):
    #             tmp += sL[j] * sW[tx, j]
    #         cuda.syncthreads()
    #     cuda.syncthreads()
        
    #     F[x] = tmp
        
    # cuda.syncthreads()
    
    
    
    # #add layer 2 biases
    # if x < F.shape[0]:
    #     #preload biases and layer sections
    #     sD[tx] = G[x]
    #     sL[tx] = F[x]
    #     cuda.syncthreads()
        
    #     #add bias to layer and replace original layer
    #     # C[x] = sD[tx] + sL[tx]
    #     F[x] = 1/(1+m.exp(-1*(sD[tx]+sL[tx]))) 
    # cuda.syncthreads()
    
        
    
    # #layer 3 weights
    
    
    # if x < I.shape[0]:
    #     tmp = 0.
    #     for i in range(int(F.shape[0]/TPB)):
            
    #         #preload memory
    #         sL[tx] = F[tx+i*TPB]
            
    #         for k in range(TPB):    
    #             sW[tx, k] = H[x, i*TPB+k]
            
    #         cuda.syncthreads()
            
    #         #perform multiplication
    #         for j in range(TPB):
    #             tmp += sL[j] * sW[tx, j]
    #         cuda.syncthreads()
    #     cuda.syncthreads()
        
    #     I[x] = tmp
        
    # cuda.syncthreads()
   
    

    
    
    
    # #add layer 3 biases
    
    # #biases    
    
    # if x < I.shape[0]:
    #     #preload biases and layer sections
    #     sD[tx] = J[x]
    #     sL[tx] = I[x]
    #     cuda.syncthreads()
        
    #     #add bias to layer and replace original layer
    #     # I[x] = sD[tx]+sL[tx]
    #     tmp = 0.
    #     tmp = 1/(1+m.exp(-1*(sD[tx]+sL[tx]))) 
    #     I[x] = tmp
        
    # cuda.syncthreads()
    
    
    
    
    # #multiply layer 4 by weights
    # if x < L.shape[0]:
    #     tmp = 0.
    #     for i in range(int(I.shape[0]/TPB)):
            
    #         #preload memory
    #         sL[tx] = I[tx+i*TPB]
            
    #         for k in range(TPB):    
    #             sW[tx, k] = K[x, i*TPB+k]
            
    #         cuda.syncthreads()
            
    #         #perform multiplication
    #         for j in range(TPB):
    #             tmp += sL[j] * sW[tx, j]
    #         cuda.syncthreads()
    #     cuda.syncthreads()
        
    #     L[x] = tmp
        
    # cuda.syncthreads()
    

    
    
    
    # #add layer 4 biases
    
    # #biases    
    
    # if x < L.shape[0]:
    #     #preload biases and layer sections
    #     sD[tx] = M[x]
    #     sL[tx] = L[x]
    #     cuda.syncthreads()
        
    #     #add bias to layer and replace original layer
    #     # I[x] = sD[tx]+sL[tx]
    #     tmp = 1/(1+m.exp(-1*(sD[tx]+sL[tx]))) 
    #     L[x] = tmp
        
    # cuda.syncthreads()
    
    
    # #multiply layer 5 weights
    if x < P.shape[0]:
        tmp = 0.
        for i in range(int(L.shape[0]/TPB)):
            
            #preload memory
            sL[tx] = C[tx+i*TPB]
            
            for k in range(TPB):    
                sW[tx, k] = N[x, i*TPB+k]
            
            cuda.syncthreads()
            
            #perform multiplication
            for j in range(TPB):
                tmp += sL[j] * sW[tx, j]
            cuda.syncthreads()
        cuda.syncthreads()   
        O[x] = tmp
        
    cuda.syncthreads()
    
    
    # #add final biases
    
    if x < P.shape[0]:
        #preload biases and layer sections
        sD[tx] = O[x]
        sL[tx] = P[x]
        cuda.syncthreads()
        
        tmp = sD[tx]+sL[tx]
            
        O[x] = 1/(1+m.exp(-1*(tmp))) 
        
    cuda.syncthreads()
    

    # ###NEURAL NETWORK ENDS HERE
















    ### BACKPROPOGATION BEGINS HERE
       
        
    # each threads is responsible for calculating one of the previous layer
    # with respect to layer 5 activations
    if Q[0] == 2:
            
        if x < L.shape[0]:
            #preload data
            tmp = 0.
            for i in range(R.shape[0]/TPB):
                for k in range(TPB):
                    #load weights in by column
                    sB[tx, k] = N[i*TPB+k, x]
                cuda.syncthreads()
                
                
                #perform calculation
                for j in range(2):
                        tmp += -2*(O[j]-j)*O[j]*(1-O[j])*sB[tx, j]
                
                cuda.syncthreads()
                
            R[x] = tmp + L[x]
                
    
                        
            cuda.syncthreads()
            
            
            
        
        #with respect to layer 5 weights      
        if x < 640:
            
            #preload data            
            
            #previous layer
            sA[tx] = L[x]
            cuda.syncthreads()
            
            
            #do calculation
            for j in range(2):
                N[j, x] =  -2*(O[j]-j)*(O[j]-O[j]**2)*sA[tx]
                cuda.syncthreads()
    
            
            
        #with respect to layer 5 biases
        if x < 640:
                if x < 2:
                    P[x] =  -2*(O[x]-x)*(O[x]-O[x]**2)
                else:
                    pass
                cuda.syncthreads()

    
    if Q[0] == 1:
        if x < L.shape[0]:
            #preload data
            tmp = 0.
            for i in range(R.shape[0]/TPB):
                for k in range(TPB):
                    #load weights in by column
                    sB[tx, k] = N[i*TPB+k, x]
                cuda.syncthreads()
                
                
                #perform calculation
                for j in range(2):
                    if j == 0:
                        tmp += -2*(O[j]-1)*O[j]*(1-O[j])*sB[tx, j]
                    if j == 1:
                        tmp += -2*(O[j]-0)*O[j]*(1-O[j])*sB[tx, j]
                cuda.syncthreads()
                
            R[x] = tmp + L[x]
                
    
                        
            cuda.syncthreads()
    
            
            
        
        #with respect to layer 5 weights      
        if x < 640:
            
            #preload data            
            
            #previous layer
            sA[tx] = C[x]
            cuda.syncthreads()
            
            
            #do calculation
            for j in range(2):
                if j == 0:
                    N[j, x] =  -2*(O[j]-1)*(O[j]-O[j]**2)*sA[tx]
                if j == 1:
                    N[j, x] =  -2*(O[j]-0)*(O[j]-O[j]**2)*sA[tx]

                cuda.syncthreads()
    
            
            
        #with respect to layer 5 biases
        if x < 640:
                if x < 2:
                    if x == 0:
                        P[x] = -2*(O[x]-1)*(O[x]-O[x]**2)
                    if x == 1:
                        P[x] = -2*(O[x]-0)*(O[x]-O[x]**2)
                        
                else:
                    pass
                cuda.syncthreads()
    
    #next layer
    
    
    #each threads is responsible for calculating one of the previous layer
    #with respect to layer 4 weights      
    # if x < 640:
        
    #     #preload data            
        
    #     #previous layer
    #     sA[tx] = I[x]
    #     cuda.syncthreads()
        
        
    #     #do calculation
    #     for j in range(L.shape[0]):
    #         tmp = 0.
    #         tmp = -2*(L[j]-R[j])*(L[j]-L[j]**2)*sA[tx]
    #         K[j, x] =  tmp
    #     cuda.syncthreads()
        
        
        
        
    # # #with respect to layer 4 biases
    # if x < 640:
    #         #do calculation
    #         M[x] =  -2*(L[x]-R[x])*(1/(1+m.exp((-m.log((1/L[x])-1)))))*(1-(1/(1+m.exp((-m.log((1/L[x])-1))))))
    #         cuda.syncthreads()
    
    
    
    
    # #with respect to layer 4 activations
    # if x < 640:
    #     #preload data
    #     tmp = 0.
    #     for i in range(R.shape[0]/TPB):
    #         for k in range(TPB):    
    #             sW[tx, k] = K[i*TPB+k, x]
    #         cuda.syncthreads()
            
            
    #         #perform calculation
    #         for j in range(TPB):
    #                 tmp += -2*(L[x]-R[x])*L[x]*(1-L[x])*sW[tx, j]
            
    #         cuda.syncthreads()
            
    #     R[x] = I[x] + tmp
            
                    
    #     cuda.syncthreads()
    
    
    
    
    
    
    
    # #next layer
    
    
    # # with respect to layer 3 weights
    # if x < 640:
        
    #     #preload data            
        
    #     #previous layer
    #     sA[tx] = F[x]
    #     cuda.syncthreads()
        
        
    #     #do calculation
    #     for j in range(L.shape[0]):
    #         H[j, x] = -2*(I[j]-R[j])*(I[j]-(I[j]**2))*sA[tx]
    #         cuda.syncthreads()
        
        
        
        
    # # #with respect to layer 3 biases
    # if x < 640:
    #         #do calculation
    #         J[x] =  -2*(I[x]-R[x])*(I[x]-I[x]**2)
    #         cuda.syncthreads()
    
    
    
    
    # #with respect to layer 3 activations
    # if x < 640:
    #     #preload data
    #     tmp = 0.
    #     for i in range(R.shape[0]/TPB):
    #         for k in range(TPB):    
    #             sW[tx, k] = H[i*TPB+k, x]
    #         cuda.syncthreads()
            
            
    #         #perform calculation
            
    #         for j in range(TPB):
    #                 tmp += -2*(I[x]-R[x])*I[x]*(1-I[x])*sW[tx, j]
            
    #         cuda.syncthreads()
            
    #     R[x] = F[x] + tmp
            
                    
    #     cuda.syncthreads()
    
    
    
    
    
    
    
    
    # ##next layer
    
    
    # #with respect to layer 2 weights
    # if x < 640:
        
    #     #preload data            
        
    #     #previous layer
    #     sA[tx] = C[x]
    #     cuda.syncthreads()
        
        
    #     #do calculation
    #     for j in range(L.shape[0]):
    #         E[j, x] = -2*(F[j]-R[j])*(F[j]-F[j]**2)*sA[tx]
    #     cuda.syncthreads()
        
        
        
        
    # # #with respect to layer 2 biases
    # if x < 640:
    #         #do calculation
    #         G[x] =  -2*(F[x]-R[x])*(F[x]-F[x]**2)
    #         cuda.syncthreads()
    
    
    
    
    #with respect to layer 2 activations
    # if x < 640:
    #     #preload data
    #     tmp = 0.
    #     for i in range(R.shape[0]/TPB):
    #         for k in range(TPB):    
    #             sW[tx, k] = E[i*TPB+k, x]
    #         cuda.syncthreads()
            
            
    #         #perform calculation
            
    #         for j in range(TPB):
    #                 tmp += -2*(L[x]-R[x])*L[x]*(1-F[x])*sW[tx, j]
            
    #         cuda.syncthreads()
            
    #     R[x] = F[x] + tmp
            
                    
    #     cuda.syncthreads()
    
    
    
    
    
    #next layer
    
    
    # # with respect to layer 1 weights
    if x < 640:
        
        #preload data            
        
        #previous layer
        for i in range(5):
            sA[tx] = A[x+640*i]
            cuda.syncthreads()
            
            
            #do calculation
            for j in range(L.shape[0]):
                B[j, x+i*640] =  -2*(C[j]-R[j])*(C[j]-C[j]**2)*sA[tx]
            cuda.syncthreads()
        
        
        
        
    # #with respect to layer 1 biases
    if x < 640:
            #do calculation
            D[x] =  -2*(C[x]-R[x])*(C[x]-F[x]**2)
            cuda.syncthreads()
    



#test arrays

#test weights
B_o = np.random.uniform(-0.004, 0.004, (640, 3197))
B = np.pad(B_o, ((0, 0), (0, 3)), mode='constant')

#layer 1 biases
bias_o = np.random.uniform(-4, 4, (640))

#layer 2 weights
l2w = np.random.uniform(-0.0008, 0.00008, (640, 640))

#layer2 biases
l2b = np.random.uniform(-0.04, 0.04, (640))

#layer 3 weights
l3w = np.random.uniform(-0.01, 0.01, (640, 640))

#layer 3 biases
l3b = np.random.uniform(-0.5, 0.5, (640))

#layer 4 weights
l4w = np.random.uniform(-0.0005, 0.0005, (640, 640))


#layer 4 biases
l4b = np.random.uniform(-0.5, 0.5, (640))


#layer 5 weights
l5w = np.random.uniform(-0.0005, 0.0005, (2, 640))
l5w = np.pad(l5w, ((0, 638), (0, 0)), mode = 'constant')


#layer 5 biases
l5b = np.random.uniform(-0.5, 0.5, (2))
l5b = np.pad(l5b, (0, 638), mode = 'constant')


#extra array
extra = np.full((640), 0, np.float)


"""note:
    why does l5b and l5w and actual have to have padding if only 1 or 2 of the 
    values are being written to?"""







#initialize master arrays
l2wn = np.zeros((640, 640))
l2bn = np.zeros((640))
l3wn = np.zeros((640, 640))
l3bn = np.zeros((640))
l4wn = np.zeros((640, 640))
l4bn = np.zeros((640))
l5wn = np.zeros((640, 640))
l5bn = np.zeros((640))

data = np.empty((5087, 3198))

temp = 0
##OVERSAMPLING DATA
with open('/users/ashaa/onedrive/desktop/exotrain.csv', mode= 'r') as file:
    csvFile = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
    for row in csvFile:
        data[temp] = row
        temp += 1

label = np.empty((5087))
datan = np.empty((5087, 3197))
count = 0
for i in data:
    label[count] = i[0]
    count += 1
    
count = 0
for i in data:
    datan[count] = i[1:]
    count += 1
    
sm = SMOTE()
datan, label = sm.fit_resample(datan, label)

divide = 0.

tmp = 0
for i in datan:
    tmp += 1
    if tmp < 370:          

        ##initialize data and actual
        #data
        A = np.array(row[1:], np.float)
        A = np.pad(A, (0, 3), mode='constant')

        #actual
        actual = np.array([row[0]], np.float)
        actualo = np.pad(actual, (0, 639), mode='constant')
        
        
        #initialize global
        A_globe_mem = cuda.to_device(A)
    
        #global weights
        B_globe_mem = cuda.to_device(B)
        
        #global layer 2
        C_globe_mem = cuda.device_array((B_o.shape[0]), dtype='float32')
        
        #global layer 1 bias
        Bias_globe_mem = cuda.to_device(bias_o)
        
        #global layer2 weights
        l2w_globe_mem = cuda.to_device(l2w)
        
        #global layer 3
        l3_globe_mem = cuda.device_array((640))
        
        #global layer 2 biases
        l2b_globe_mem = cuda.to_device(l2b)
        
        #global layer 3 weights
        l3w_globe_mem = cuda.to_device(l3w)
        
        #global layer 3 biases
        l3b_globe_mem = cuda.to_device(l3b)
        
        #global layer 4
        l4_globe_mem = cuda.device_array((640), dtype='float32')
        
        #global layer 4 weights
        l4w_globe_mem = cuda.to_device(l4w)
        
        #globale layer 4 biases
        l4b_globe_mem = cuda.to_device(l4b)
        
        #global layer 5
        l5_globe_mem = cuda.device_array((640))
        
        #global layer 5 weights
        l5w_globe_mem = cuda.to_device(l5w)
        
        #global layer 5 biases
        l5b_globe_mem = cuda.to_device(l5b)
        
        #global layer 6
        l6_globe_mem = cuda.device_array((640), dtype='float32')
        
        #global actual
        actual = cuda.to_device(actualo)
        
        #global extra
        extra_globe_mem = cuda.to_device(extra)
        

        
        
        #call on kernel
        
        matrixmultiply[10, TPB](A_globe_mem, B_globe_mem, C_globe_mem, Bias_globe_mem,
                                  l2w_globe_mem, l3_globe_mem, l2b_globe_mem,
                                    l3w_globe_mem, l4_globe_mem, l3b_globe_mem,
                                  l4w_globe_mem, l5_globe_mem, l4b_globe_mem,
                                  l5w_globe_mem, l6_globe_mem, l5b_globe_mem,
                                  actual, extra_globe_mem)
        
        
        if m.isnan(l2w_globe_mem.copy_to_host()[0][0]) is True or m.isnan(l2b_globe_mem.copy_to_host()[0]) is True or m.isnan(l3w_globe_mem.copy_to_host()[0][0]) is True or m.isnan(l3b_globe_mem.copy_to_host()[0]) is True or m.isnan(l4w_globe_mem.copy_to_host()[0][0]) is True or m.isnan(l4b_globe_mem.copy_to_host()[0]) is True or m.isnan(l5w_globe_mem.copy_to_host()[0][0]) is True or m.isnan(l5b_globe_mem.copy_to_host()[0]) is True:
            continue
        
        divide += 1
        Bn = np.add(B_globe_mem.copy_to_host(), B)
        bias_on = np.add(bias_o, Bias_globe_mem.copy_to_host())
        l2wn = np.add(l2wn, l2w_globe_mem.copy_to_host())
        l2bn = np.add(l2bn, l2b_globe_mem.copy_to_host())
        l3wn = np.add(l3wn, l3w_globe_mem.copy_to_host())
        l3bn = np.add(l3bn, l3b_globe_mem.copy_to_host())
        l4wn = np.add(l4wn, l4w_globe_mem.copy_to_host())
        l4bn = np.add(l4bn, l4b_globe_mem.copy_to_host())
        l5wn = np.add(l5wn, l5w_globe_mem.copy_to_host())
        l5bn = np.add(l5bn, l5b_globe_mem.copy_to_host())
        
        # print(Bn.shape)
        
        # if tmp % 75 == 0:
        #     # print(tmp)
        #     with open('/users/ashaa/onedrive/desktop/exotrain.csv', mode= 'r') as file:
        #         csvFile = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        #         temp = 0.
        #         for row in csvFile:
        #             temp += 1
        #             if temp > 35:
        #                 break
        #             if temp < 35:
        #                 ##initialize data and actual
        #                 #data
        #                 A = np.array(row[1:], np.float)
        #                 A = np.pad(A, (0, 3), mode='constant')
                
        #                 #actual
        #                 actual = np.array([row[0]], np.float)
        #                 actualo = np.pad(actual, (0, 639), mode='constant')
                        
                        
        #                 #initialize global
        #                 A_globe_mem = cuda.to_device(A)
                    
        #                 #global weights
        #                 B_globe_mem = cuda.to_device(B)
                        
        #                 #global layer 2
        #                 C_globe_mem = cuda.device_array((B_o.shape[0]), dtype='float32')
                        
        #                 #global layer 1 bias
        #                 Bias_globe_mem = cuda.to_device(bias_o)
                        
        #                 #global layer2 weights
        #                 l2w_globe_mem = cuda.to_device(l2w)
                        
        #                 #global layer 3
        #                 l3_globe_mem = cuda.device_array((640))
                        
        #                 #global layer 2 biases
        #                 l2b_globe_mem = cuda.to_device(l2b)
                        
        #                 #global layer 3 weights
        #                 l3w_globe_mem = cuda.to_device(l3w)
                        
        #                 #global layer 3 biases
        #                 l3b_globe_mem = cuda.to_device(l3b)
                        
        #                 #global layer 4
        #                 l4_globe_mem = cuda.device_array((640), dtype='float32')
                        
        #                 #global layer 4 weights
        #                 l4w_globe_mem = cuda.to_device(l4w)
                        
        #                 #globale layer 4 biases
        #                 l4b_globe_mem = cuda.to_device(l4b)
                        
        #                 #global layer 5
        #                 l5_globe_mem = cuda.device_array((640))
                        
        #                 #global layer 5 weights
        #                 l5w_globe_mem = cuda.to_device(l5w)
                        
        #                 #global layer 5 biases
        #                 l5b_globe_mem = cuda.to_device(l5b)
                        
        #                 #global layer 6
        #                 l6_globe_mem = cuda.device_array((640), dtype='float32')
                        
        #                 #global actual
        #                 actual = cuda.to_device(actualo)
                        
        #                 #global extra
        #                 extra_globe_mem = cuda.to_device(extra)
                        
                        
                        
        #                 #call on kernel
        #                 matrixmultiply[10, TPB](A_globe_mem, B_globe_mem, C_globe_mem, Bias_globe_mem,
        #                                           l2w_globe_mem, l3_globe_mem, l2b_globe_mem,
        #                                             l3w_globe_mem, l4_globe_mem, l3b_globe_mem,
        #                                           l4w_globe_mem, l5_globe_mem, l4b_globe_mem,
        #                                           l5w_globe_mem, l6_globe_mem, l5b_globe_mem,
        #                                           actual, extra_globe_mem)
                        
        #                 # print('inside')
        #                 if m.isnan(l2w_globe_mem.copy_to_host()[0][0]) is True or m.isnan(l2b_globe_mem.copy_to_host()[0]) is True or m.isnan(l3w_globe_mem.copy_to_host()[0][0]) is True or m.isnan(l3b_globe_mem.copy_to_host()[0]) is True or m.isnan(l4w_globe_mem.copy_to_host()[0][0]) is True or m.isnan(l4b_globe_mem.copy_to_host()[0]) is True or m.isnan(l5w_globe_mem.copy_to_host()[0][0]) is True or m.isnan(l5b_globe_mem.copy_to_host()[0]) is True:
        #                     continue
                        
        #                 # print(l2w_globe_mem.copy_to_host())
        #                 divide += 1
        #                 Bn = np.add(B_globe_mem.copy_to_host(), B)
        #                 bias_on = np.add(bias_o, Bias_globe_mem.copy_to_host())
        #                 l2wn = np.add(l2wn, l2w_globe_mem.copy_to_host())
        #                 l2bn = np.add(l2bn, l2b_globe_mem.copy_to_host())
        #                 l3wn = np.add(l3wn, l3w_globe_mem.copy_to_host())
        #                 l3bn = np.add(l3bn, l3b_globe_mem.copy_to_host())
        #                 l4wn = np.add(l4wn, l4w_globe_mem.copy_to_host())
        #                 l4bn = np.add(l4bn, l4b_globe_mem.copy_to_host())
        #                 l5wn = np.add(l5wn, l5w_globe_mem.copy_to_host())
        #                 l5bn = np.add(l5bn, l5b_globe_mem.copy_to_host())
        #                 # print(divide)
        
        



# np.savetxt('/users/ashaa/onedrive/desktop/code/ai/l4w.csv', np.add(l4w, l4wn/divide))
# np.savetxt('/users/ashaa/onedrive/desktop/code/ai/l4b.csv', np.add(l4b, l4bn/divide))
np.savetxt('/users/ashaa/onedrive/desktop/code/ai/l1w.csv', np.add(B, Bn/divide))
np.savetxt('/users/ashaa/onedrive/desktop/code/ai/l1b.csv', np.add(bias_o, bias_on/divide))
# np.savetxt('/users/ashaa/onedrive/desktop/code/ai/l2w.csv', np.add(l2w, l2wn/divide))
# np.savetxt('/users/ashaa/onedrive/desktop/code/ai/l2b.csv', np.add(l2b, l2bn/divide))
# np.savetxt('/users/ashaa/onedrive/desktop/code/ai/l3w.csv', np.add(l3w, l3wn/divide))
# np.savetxt('/users/ashaa/onedrive/desktop/code/ai/l3b.csv', np.add(l3b, l3bn/divide))
np.savetxt('/users/ashaa/onedrive/desktop/code/ai/l5w.csv', np.add(l5w, l5wn/divide))
np.savetxt('/users/ashaa/onedrive/desktop/code/ai/l5b.csv', np.add(l5b, l5bn/divide))



print(divide)



print(datetime.now() - startTime)