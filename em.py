import numpy as np
from scipy.stats import norm,multivariate_normal
from numpy.random import random
from math import nan

def condexp(mu,sigma,l,u):
    # Conditional expectation of N(mu,sigma) given interval [l,u]
    # l=-np.inf and/or u=np.inf are admissible
    sn = norm()
    lt=(l-mu)/sigma
    ut=(u-mu)/sigma
    return -sigma*(sn.pdf(ut)-sn.pdf(lt))/(sn.cdf(ut)-sn.cdf(lt))+mu
    
            

def expstats1d(mu,sigma,cd,nticks):
    # Calculates the expected mean and variance given current model N(mu,sigma)
    # and coarse 'binned' data cd
    #
    # NOTE: function takes standard deviation sigma as input, but
    # returns variance Sigma! This is because scipy.stats.norm takes sigma
    # as input.
    
    nmusigma=norm(mu,sigma)
    
    # First the expected mean:
    emu=0
    for i in range(len(cd.data[0])):
        emu+=cd.data[0][i]
    if len(cd.data[1])==0:  # only a single bin  
        emu+=cd.data[2][0]*mu
    else: 
        emu+=cd.data[2][0]*condexp(mu,sigma,-np.inf,cd.data[1][0])
        for i in range(1,cd.data[1].size-1):
            emu+=cd.data[2][i]*condexp(mu,sigma,cd.data[1][i-1],cd.data[1][i])
        emu+=cd.data[2][cd.data[2].size-1]*condexp(mu,sigma,cd.data[1][cd.data[1].size-1],np.inf)
    emu=emu/cd.size
    
    # Now the expected variance (approximately by summing over grid) 
    
    eSigma=0
    for i in range(len(cd.data[0])):
        eSigma+=(cd.data[0][i]-emu)**2
    
    if len(cd.data[1])==0:  # only a single bin  
        eSigma+= cd.data[2][0]*(sigma**2)
    else:    
        # define pseudo infinity bin bounds:
       
        pminusinf=emu-5*sigma
        pplusinf=emu+5*sigma
   
        extbounds=np.hstack((pminusinf,cd.data[1],pplusinf))
        
        
       
        for b in range(len(extbounds)-1):
            binprob=nmusigma.cdf(extbounds[b+1])-nmusigma.cdf(extbounds[b])
            bsigma=0
            grid=np.linspace(extbounds[b],extbounds[b+1],nticks)
            for i in range(nticks-1):
                prob=(nmusigma.cdf(grid[i+1])-nmusigma.cdf(grid[i]))/binprob
                square=((grid[i+1]+grid[i])/2-emu)**2
                bsigma+=prob*square
                #bsigma+= (nmusigma.cdf(grid[i+1])-nmusigma.cdf(grid[i]))*(grid[i]-emu)**2  
            eSigma+=cd.data[2][b]*bsigma
    eSigma=eSigma/cd.size    
    
    return emu,eSigma


def expstats2d(mu,Sigma,cd):

    # Literature: Little,Rubin Section 8.2.1
    
    d=cd.alldata()
    
    # An array to store conditional expectations of missing values
    # A little wasteful in terms of space ...
    cexpcts=np.zeros(d.shape)
    for i in range(d.shape[0]):
        for h in [0,1]:
            if np.isnan(d[i,h]):
                cexpcts[i,h]=mu[h]+(Sigma[0,1]/Sigma[1-h,1-h])*(d[i,1-h]-mu[1-h]) 
            
   
    # Calculated expected mean:        
    emu=np.zeros(2)
    for i in range(d.shape[0]):
        for h in [0,1]:
            if np.isnan(d[i,h]):
                emu[h]+=cexpcts[i,h]
            else:
                emu[h]+=d[i,h]
    emu=emu/d.shape[0]
    
    # Calculate expected covariance matrix:
    esigma2=np.zeros((2,2))
    b=np.zeros(2)
    
    condvar=np.zeros(2) #conditional variance of one variable, given the other
    for h in [0,1]:
        condvar[h]=Sigma[h,h]-(Sigma[0,1]**2)/(Sigma[1-h,1-h])
    
    
    dexp=np.zeros(2)    
    
    for i in range(d.shape[0]):
        for h in [0,1]:
            if np.isnan(d[i,h]):
                dexp[h]=cexpcts[i,h]
                b[h]=condvar[h]
            else:
                dexp[h]=d[i,h]
                b[h]=0 
           
        esigma2[0,0]+=(dexp[0]-emu[0])**2 + b[0]
        esigma2[0,1]+=(dexp[0]-emu[0])*(dexp[1]-emu[1])
        esigma2[1,1]+=(dexp[1]-emu[1])**2 +b[1]
        
    esigma2[1,0]=esigma2[0,1]
    esigma2=esigma2/d.shape[0]
    
    return emu,esigma2



    
def em(cd,initsize=100,minchange=0.0001,maxits=50,**kwargs):
    
    change=minchange+1
    it=0
    
    if 'initgauss' in kwargs.keys():
        mu=kwargs['initgauss'][0]
        Sigma=kwargs['initgauss'][1]
    else: 
        mu,Sigma = cd.acameanvar(initsize)
    #print("EM initial mu: {} \n sigma: \n {} ".format(mu,Sigma))
    
    while change > minchange and it<maxits :
        if cd.dim == 1:
            munew,Sigmanew = expstats1d(mu,np.sqrt(Sigma),cd,5)
     #       print("EM update mu: {} \n sigma: \n {} ".format(munew,Sigmanew))
        if cd.dim == 2:     
            munew,Sigmanew = expstats2d(mu,Sigma,cd)
        change=np.max( (np.max((mu-munew)**2), np.max((Sigma-Sigmanew)**2) ))
        it+=1
        mu=munew
        Sigma=Sigmanew
   
    print("EM iterations: {}".format(it))
    return munew,Sigmanew

def em2d_disc(cd,b,e,c,perc):
    # b: number of bins
    # e: termination threshold for minimum change in Sigma
    # c: maximum number of iterations
    minchange=e
    
    dcd = dcdata(cd,b,1,perc)
    terminate = False
    itcount=0
    
    dcd.setfrac_expectation()
    dcd.setgauss()
    
    mu=dcd.mu
    Sigma=dcd.Sigma
    
    #print("mu: {}  Sigma: {}".format(mu,Sigma))
    
    
    while not terminate:
        dcd.setfrac_expectation()
        dcd.setgauss()
        nmu=dcd.mu
        nSigma=dcd.Sigma
        #print("mu: {}  Sigma: {}".format(nmu,nSigma))
        
        if error(mu,Sigma,nmu,nSigma)[1]<minchange or itcount==c:
            terminate = True
        mu,Sigma=nmu,nSigma    
    
    print("Iterations: {}".format(itcount))
    return dcd,mu,Sigma
    

def error(mu,sigma,mmu,ssigma):
    merr=np.linalg.norm(mu-mmu)
    serr=np.linalg.norm(sigma-ssigma)
    return merr,serr
