import numpy as np
from scipy.stats import norm,multivariate_normal
from numpy.random import random
from math import nan
from aim import getbinidx,cdata

def coarsenprob_base(x,cparams,ctype):
    if ctype =='uniform':
        return cparams
    if ctype =='tail':
        # cparams[2]: between 0 and  1: maximum coarsening probability
        # cparams[1]: any real: location parameter (probability 1/2 of maximum)
        # cparams[0]: real: slope and direction parameter: positive values: increasing prob
        #                                                  negative values: decreasing prob
        #                                                  larger absolute value: steeper in/de- crease 
        return cparams[2]/(1+np.exp(-cparams[0]*(x-cparams[1])))
    if ctype =='central':
        """
        coarsen with bell-shaped missingness probabilities
        centered at cparams[1] with width parameter cparams[0] and
        a scaling factor cparams[2]
        """
        return 2*cparams[2]/(1+np.exp(cparams[0]*(x-cparams[1])**2))


def coarsenprob(x,cparams_list,ctype_list):
    #cparams_list and ctypes_list are lists of parameters for a single coarsening probability
    #function. The final coarsenprob is a noisy-of the probabilities returned by each function
    #in the list. 
    p=1
    
    for i in range(len(cparams_list)):
        p*=(1-coarsenprob_base(x,cparams_list[i],ctype_list[i]))
     
    return 1-p    
    
def sample(N,mu,Sigma,cparams,ctypes,**kwargs):
    #cparams and ctypes are lists of lists: outer list goes over the dimensions (1 or 2)
    #                             inner lists go over components of coarsening mechanisms
        
    dim=Sigma.shape[0]
    
    if dim==1 and not 'binbounds' in kwargs.keys():
        print("Specification of binbounds missing for sampling in mode 'binned'")
        return
    
    
    # Sample the complete data:
    if dim==1:
        q=norm.rvs(loc=mu, scale=np.sqrt(Sigma), size=N).reshape((N,1))
    if dim==2:
        q=multivariate_normal.rvs(mean=mu, cov=Sigma, size=N)
    
    # for ctype=binned, prepare the data lists:
    if dim==1:
        binbounds=kwargs['binbounds']
        points = []
        bincounts = np.zeros(len(binbounds)+1)
        # augmented binbounds with left and right pseudo infinities:
        if len(binbounds)>0:
            augbbs=np.hstack((binbounds[0]-1,binbounds,binbounds[len(binbounds)-1]+1))
    
    for i in range(N):
        coarsen=np.zeros(dim)
        for h in range(dim):
            if random()<coarsenprob(q[i,h],cparams[h],ctypes[h]):
                coarsen[h]=1 
        # Make sure coarsening indicator is 1 for at most one dimension:
        if np.sum(coarsen)==2:
            coarsen[np.random.randint(0,2)]=0
         
        # Perform the coarsening:
        for h in range(dim):
            if dim==2:
                if coarsen[h]==1:
                    q[i,h]=None
            if dim==1:
                if coarsen[h]==0:
                    points.append(q[i,h])
                if coarsen[h]==1:
                    if len(binbounds)>0:
                        bincounts[getbinidx(augbbs,q[i,h])]+=1
                    else:
                        bincounts[0]+=1
                    
    if dim==2:
        D=q
        cdtype='mvals'
    if dim==1:
        D=(points,binbounds,bincounts)
        cdtype='binned'
    return cdata(D,cdtype)

  
def prettyprintcp(cparams):
    rstring=""
    for cp in cparams:
        for ccp in cp:
            rstring+=str(ccp.transpose())
    return rstring
