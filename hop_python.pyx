import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "energy.h":
    cdef cppclass Energy[T]:
        Energy(int nLabel, int nVar, int nPair, int nHigher)
        void SetUnaryCost(T *)
        void SetPairCost(int*, T *)
        int SetOneHOP(int n, #number of nodes participating in this potential
            int* ind, # indices of participating nodes - needs to be allocated outside
            T* weights,  # w_i for each node in the potential - needs to be allocated outside
            T* gammas,  # gamma_l for all labels and gamma_max - do not allocate
            T Q) 
            
cdef extern from "expand.h":
    cdef cppclass AExpand[T]:
        AExpand(Energy *e, int maxIter)
        T minimize(int *solution, T* ee) 

def robust_pn(np.ndarray[np.float64_t, ndim=2, mode='c'] unary_cost,
              np.ndarray[np.int32_t, ndim=2, mode='c'] edges,
              np.ndarray[np.float64_t, ndim=1, mode='c'] pairwise_cost,
              HOP,
              MAX_ITER = 50):
              
    cdef int nLabel = unary_cost.shape[1]
    cdef int nVar = unary_cost.shape[0]
    cdef int nPair = edges.shape[0]
    cdef int nHigher = len(HOP)
    
    cdef Energy[double] *energy = new Energy[double](nLabel, nVar, nPair, nHigher)
    
    energy.SetUnaryCost(<double*>unary_cost.data)
    energy.SetPairCost(<int*>edges.data, <double*>pairwise_cost.data)
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] ind
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] w
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] gamma
#    
    for hop in HOP:
        ind = hop['ind']
        w = hop['w']
        gamma = hop['gamma']
        #cdef double Q = hop['Q']
        energy.SetOneHOP(len(hop['ind']), <int*>ind.data, <double*>w.data, <double*>gamma.data,hop['Q'])
#        
    cdef AExpand[double] *expand = new AExpand[double](energy, MAX_ITER)
    
    cdef double ee[3]
    cdef double E = 0.0
#    
    cdef np.npy_intp result_shape[1]
    result_shape[0] = nVar
#    
    cdef np.ndarray[np.int32_t, ndim=1] result = np.PyArray_SimpleNew(1, result_shape, np.NPY_INT32)
    cdef int * result_ptr = <int*>result.data
    E = expand.minimize(result_ptr, ee)
    del expand
    del energy
    return result
    


