import numpy as np
from scipy.stats import norm,multivariate_normal
from numpy.random import random
from math import nan
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import utils as utl
import time

# class collecting several utility methods for 
# computations with 1-dim and 2-dim Gaussians
class NormalUtils: 
    def __init__(self,mu,Sigma):
        # Coefficients for approximation of Normal CDF, Abramowitz, Stegum, Handbook of 
        # Mathematical Functions 26.2.18:
        self.c=np.array([0.196854,0.115194,0.000344,0.019527])
        
        self.dim=len(mu)
        self.mu,self.Sigma=mu,Sigma
        
        
        if self.dim==2:  #the 2-d case
            self.Larr,self.L=self.getCholesky(self.Sigma)
        
    def getCholesky(self,Sigma):
        L=np.zeros((3))
        L[0]=np.sqrt(Sigma[0,0])  
        L[1]=Sigma[1,0]/L[0]
        L[2]=np.sqrt(Sigma[1,1]-(Sigma[1,0]**2)/Sigma[0,0])
        Lm=np.array([[L[0],0],[L[1],L[2]]])
        return L,Lm
    
    def getSigmafromCholesky(self,L):
        return np.matmul(L,L.transpose())
   
    # Approximation of the 1-d standard normal CDF:
   
    def gcdf(self,z):
        
        if z<0:
            zz = -z
        else:
            zz =z
        r=1
        for i in range(4):
            r+=self.c[i]*(zz**(i+1))
        r= 1-1/(2*(r**4))
        
        if z>0:
            return r
        else:
            return 1-r
    
    
    # Derivative of the standard normal (1-dim) CDF based on
    # Version 26.2.18 of approximation:
    def derivgcdf(self,z):
        if z<0:
            z*=-1
        num=self.c[0]
        for i in range(1,4):
            num+=(i+1)*self.c[i]*(z**i)
        denom=1
        for i in range(1,5):
            denom+=self.c[i-1]*(z**i)
        return (2*num)/(denom**5)  
    
    # Gradient of CDF of standard normal distribution (d=1 or d=2)
    # Gradient with respect to the arguments of the CDF
    def gradgcdf(self,z):
        if len(z)==1:
            return self.derivgcdf(z)
        if len(z)==2:
            grad=np.zeros(2)
            grad[0]=self.derivgcdf(z[0])*self.gcdf(z[1])
            grad[1]=self.derivgcdf(z[1])*self.gcdf(z[0])
            return grad
    
    def invL(self,L):
        result=np.zeros((2,2))
        result[0,0]=1/self.L[0,0]
        result[1,0]=-self.L[1,0]/(self.L[0,0]*self.L[1,1])
        result[1,1]=1/self.L[1,1]
        return result

    def z(self,x):
        if self.dim==1:
            return (x-self.mu)/np.sqrt(self.Sigma)
        if self.dim==2:
            # maps x=(x0,x1) to z s.t. resulting transformation is N(0,1) distributed
            return np.dot(self.invL(self.getCholesky(self.Sigma)[1]),x-self.mu)
    
    def zx(self,x):
        # compute only z[0] of z(self,x), which only depends on x[0]
        return (x-self.mu[0])/self.L[0,0]

    def linfunc(self,yor,x): 
        # the linear function that maps an x-value to
        # a y-value, such that (x,linfunc(yor,x)) is 
        # z((xor,yor)) for some xor
        return (-self.L[1,0]*x+yor-self.mu[1])/self.L[1,1]
    
    def gradlinfunc(self,yor,x):
        # the gradient of linfunc w.r.t. to the parameters (mu[1],L[1,0],L[1,1])
        result = np.zeros(3)
        result[0]=(-1/self.L[1,1]) # mu[1] deriv
        result[1]=(-x/self.L[1,1]) # L[1,0] deriv
        result[2]=((self.L[1,0]*x-yor+self.mu[1])/self.L[1,1]**2) #L[1,1] deriv
        return result
    

### Other general utilities:

def getbinidx(b,x):
    '''
    b is an array of (increasing) bin bounds
    x a scalar
    returns the index of the bin that x belongs to
    
    the b[0] and b[b.size-1] represent minus and plus "pseudo infinities"
    everthing less than b[1] (>b[b.size-2]) is considered to belong to a 
    left (right) unbounded bin
    '''
    #print("getbinidx for x: {} with binidxs {}".format(x,b))
    if x<b[1]:
        #print("return 0")
        return 0
    if x>b[b.size-2]:
        #print("return {}".format(b.size-2))
        return b.size-2
    
    lower=1
    upper=b.size-2
    while upper-lower>1:
        mid=int((lower+upper)/2)
        if x>b[mid]:
                lower=mid
        else:
                upper=mid
    #print("returning {}".format(lower))            
    return lower


class cdata:
    '''
    Coarse data. Can be one of two types:
    - 'mvals': 2-d data with missing values. At most one missing value 
       at each data point. Missing values represented by 'None'
    - 'binned': 1-d data where some of the datapoints are only given by bin membership with respect 
       to a given partition of the real line into bins. In the special case of a single bin, this 
       corresponds to 1-d data with missing values (=completely unobserved cases)
    '''
    def __init__(self,D,ctype):
        #if ctype = 'mvals': D is an Nx2 array of 2-dim data points 
        #if ctype = 'binned': 
        # D is a list of three elements:
        # D[0]: list of reals (fully observed data points)
        # D[1]: list of bin bounds in increasing order; empty list in the single bin case
        # D[2]: list of counts of observations in each bin. len(D[2])=len(D[1])+1, also covering
        #       counts for the left and right unbounded bins
        self.data = D 
        self.ctype=ctype
        if ctype=='mvals':
            self.dim = 2
        if ctype=='binned':
            self.dim=1

        if self.ctype=='mvals':
            self.size = self.data.shape[0]
        if self.ctype=='binned':
            self.size = len(self.data[0])+np.sum(self.data[2])
            
    def alldata(self):
        return(self.data)
    
    def minmax(self):
        '''
        Returns 2xd array with minimum/maximum non-nan values in the d dimensions
        '''
        if ctype=='mvals':
            mm=np.zeros((2,self.dim))
            for i in range(self.dim):
                    mm[0,i]=np.min(self.data[~np.isnan(self.data[:,i]),i])
                    mm[1,i]=np.max(self.data[~np.isnan(self.data[:,i]),i])
        if ctype=='binned': # include the bin bounds
            mm=np.zeros((2,1))
            mm[0,0]=np.min( (np.min(self.data[0]), self.data[1][0] ) )  
            mm[1,0]=np.max( (np.max(self.data[0]), self.data[1][len(self.data[1])-1] ) ) 
        return mm
    
    def percinc(self):
        '''
        type binned: proportion of imprecise (bin valued) observations
        type mvals: proportion of missing values (not: of cases that are incomplete)
        '''
        if self.ctype=='binned':
            return np.sum(self.data[2])/self.size
        if self.ctype=='mvals':
            return len(np.where(np.isnan(self.data))[0])/self.data.size
    
    def acameanvar(self,numpoints):
        '''
        The 'available case' mean and covariance based on a random sub-sample
        of numpoints of the data
        '''
        if numpoints<self.size:
            if self.ctype=='mvals':
                idx=np.random.choice(self.size,numpoints,replace=False)
                subdata=self.data[idx,:]
            if self.ctype=='binned':
                idx=np.random.choice(len(self.data[0]),numpoints,replace=False)
                subdata=np.array(self.data[0])[idx]
        else:
            if self.ctype=='mvals':
                subdata=self.data
            if self.ctype=='binned':
                subdata=self.data[0]
                
        amu=np.zeros(self.dim)
        for i in range(self.dim):
            if self.ctype=='mvals':
                amu[i]=np.mean( subdata[~np.isnan(subdata[:,i]),i] )
            if self.ctype=='binned':
                amu[i]=np.mean(subdata)
        
        if self.ctype=='binned':
            acov=np.var(subdata)
        if self.ctype=='mvals':
            acov=np.cov(subdata[ ~np.any(np.isnan(subdata),axis=1) ,: ] ,rowvar=False)     
        
        
        return amu,acov
       
    def carllikelihood(self,mu,Sigma):
        # calculated the 'face value' or 'car' log-likelihood of the data
        # under parameters mu,Sigma
        if self.ctype=='mvals':
            gaussx=norm(mu[0],Sigma[0,0])
            gaussy=norm(mu[1],Sigma[1,1])
            gaussxy=multivariate_normal(mu,Sigma)
            llik=0
            for i in range(self.data.shape[0]):
                p=self.data[i,:]
                if p[0]==None:
                    llik+=np.log(gaussy.pdf(p[1]))
                if p[1]==None:
                    llik+=np.log(gaussy.pdf(p[0]))
                if not any(p[:]==None):
                    llik+=np.log(gaussxy.pdf(p))
                return llik    
        if self.ctype=='binned':
            gauss=norm(mu,np.sqrt(Sigma))
            llik=0
            for d in range(len(self.data[0])):
                llik+=np.log(gauss.pdf(self.data[0][d])) 
            if len(self.data[1])>0:
                llik+=np.log( self.data[2][0]*gauss.cdf(self.data[1][0]))
                for b in range(len(self.data[1])-1):
                    llik+=np.log( self.data[2][b+1]*\
                                 (gauss.cdf(self.data[1][b+1])- gauss.cdf(self.data[1][b])))
                llik+=np.log( self.data[2][-1]*(  1-gauss.cdf(self.data[1][-1]) ))           
            return llik

    


class dcdata:
    '''
    Discrete version of cdata and data structures for managing fractional completion and 
    current Gaussian model
    
    
    nbins denotes the number of bounded bins, not counting the left and right unbounded bins. 
    The unbounded bins are intended to only cover "empty tails"
     
    All dimensions are partitioned into bins. Bin bound for the i'th dimension are specified in 
    binbounds[:,i]. In the following, 'bin' can refer to a bin in a given dimension, or to 
    a bin in the multi-dimensional space (= Cartesian product of 1-d bins).
    
    The number of bins is one more than there are binbounds:
    
    nbins[i]=binbounds[:,i].size+1

    For now: all dimensions have the same number of bins:  nbins[i]=nbins for all i.
    
    For visualizatin etc. in the 2-d case: 
    
    The indexing of the bins starts at the lower left corner: the bin containing the (x0,x1) with
    x0<min(binbounds[:,0]) and x1<min(binbounds[:,1]) has index [0,0] in matrix representations
    
    The following data is stored in matrices of dimension \times_i nbins[i]
    
   
    countsfull: matrix storing for each bin the complete data cases falling into the bin
    countsmarg: nbins x 2 matrix storing the counts of incomplete data cases falling into the respective 
                bins for each component: countsmarg[:,i] : counts for observed xi component (i=0,1)
    gausscdf: 2d-matrix storing the CDF of a (current) Gaussian distribution. gausscdf[i,j] contains the
              CDF value of the upper right corner of bin with indices [i,j], i.e., the CDF value of
              [binbounds[i,0],binbounds[j,1]]. For upper or right unbounded bins, the respective binbound 
              value is infty. 
    gaussprobs: the probability values of the bins according to the Gaussian distribution that also is
                represented by gausscdf
    fraccompl: 1d or 2d array storing fractional allocations of marginal counts to individual bins.
               Entries are raw (fractional) counts, not normalized to probabilities
    fraccomplbins (1d only): 2d array maintaining the fractional completions for all coarse data bins.
                this will be a "near diagonal" matrix, because 
                completions of the i'th coarse data bin must be contained in the range
                self.cbins_to_dbins[i,:] of discretization bins.
    fraccompl2 (2d only): 3d array maintaining the fractional completions for all incomplete data items 
               
)      
    '''
    def __init__(self,cd,nbins,nticks,initgauss,**kwargs):
        # cd is a cdata object
        # nbins: number of discretization bins (in each dimension), not counting the left and 
        #        right pseudo-unbounded bins
        # initgauss: initial parameters for the Gaussian distribution
        
        
        self.cd=cd
        
        self.ctype=cd.ctype
        self.nbins=nbins
        self.size=cd.size
        self.dim=cd.dim
        self.data=cd.alldata()
       
        if self.dim==1:
            #old: self.flexmass = len(np.where(np.isnan(d)==True)[0])
            self.flexmass = np.sum(self.data[2])
            self.fixmass = self.size-self.flexmass
        
        # Coefficients for approximation of Normal CDF, Abramowitz, Stegum, Handbook of 
        # Mathematical Functions 26.2.18:
        self.c=np.array([0.196854,0.115194,0.000344,0.019527])
        
        # create the bounds for the discretization bins:
        if 'usebinbounds' in kwargs.keys():
            # this only working for d=1 for now
            # it is assumed that the specified binbounds include left and right pseudo infinities
            bins=kwargs['usebinbounds']
            self.nbins=len(bins)-3
            self.binbounds=bins.reshape((len(bins),1))
        else:    
            self.binbounds=np.zeros((nbins+3,self.dim))
            # first, create an equal spaced partition over the part of the 
            # space actually covered by the data (adding a little margin to the 
            # minimal and maximal data points)
            # for the 'binned' case, consider the bounds of the data bins also 
            # as data points so that the discretization grid covers all data 
            # points and all data bins

            for h in range(self.dim):

                
                # First take the points +/- 3 standard deviations away from the mean (using precise data values)
                if self.ctype=='mvals':    
                    allvals=self.data[ ~np.isnan(self.data[:,h]) ,h]
                if self.ctype=='binned':
                    allvals=self.data[0]
                minbin=np.mean(allvals)-3*np.std(allvals)
                maxbin=np.mean(allvals)+3*np.std(allvals)
                
                # in the binned case, also consider the upper/lower binbounds of non-empty bins
                if (self.ctype=='binned' and len(self.data[1]>0)):
                    # Min and max occupied bin:
                    minidx=np.min(np.where(self.data[2]>0))
                    maxidx=np.max(np.where(self.data[2]>0))
                    if minidx==0:
                        minbound=self.data[1][0]
                    else: 
                        minbound=self.data[1][minidx-1]
                    if maxidx==len(self.data[2])-1:
                        maxbound=self.data[1][maxidx-1]
                    else: 
                        maxbound=self.data[1][maxidx]    
                    
                    minbin=np.min((minbin,minbound))
                    maxbin=np.max((maxbin,maxbound))
               
                self.binbounds[1:nbins+2,h]=np.linspace(minbin,maxbin,nbins+1)    

                # add pseudo-infinity bounds:
                w=self.binbounds[self.nbins+1,h]-self.binbounds[1,h]
                self.binbounds[0,h]=self.binbounds[1,h]-w
                self.binbounds[self.nbins+2,h]=self.binbounds[self.nbins+1,h]+w

        # For the ctype='binned' case, create a mapping that 
        # maps the coarse data bins to the range of discretization bins
        # with which they have a non-empty intersection. Note: it would be 
        # convenient to have that the discretization bins are a proper refinement
        # of the coarse data bins. That is difficult to align with the goal to 
        # have equal width discretization bins
        
        if self.ctype=="binned":
            n=len(self.data[2])
            self.cbins_to_dbins = np.zeros((n,2),dtype=int)
            
            low=0
            high=0
            for i in range(n-1):
                if i>0: # resetting the low bound by one, if the high bound overshot the binbound
                        # in the previous iteration
                    if self.binbounds[low]>self.data[1][i-1]:
                        low-=1
                
                while self.binbounds[high] < self.data[1][i]:
                    high+=1
                self.cbins_to_dbins[i,:]=(low,high-1)
                low=high
                
            if len(self.data[1])>0 and self.binbounds[low]>self.data[1][n-2]:
                low-=1
            high=len(self.binbounds)-2
            self.cbins_to_dbins[n-1,:]=(low,high)
       
        # define an additional vector of 'ticks' on dimension 0 used for approximate 
        # evaluation of gradient of Gaussian distribution
        # the ticks cover the bounded bins, as well as the "unbounded" bins 
        # with pseudo-infinite bounds 
        self.numticks=nticks #the number of ticks in each bin (including bin boundaries)
        self.xticks=np.zeros((self.numticks-1)*(self.nbins+2)+1)
        
        #self.xticks[0:self.numticks]=np.linspace(self.binbounds[0,0],self.binbounds[1,0],self.numticks)
        for i in range(self.nbins+2):
            self.xticks[i*(self.numticks-1):i*(self.numticks-1)+self.numticks]=\
            np.linspace(self.binbounds[i,0],self.binbounds[i+1,0],self.numticks)
           
        # NOTE: self.Sigma is the variance (1-d), resp. covariance matrix (2-d)
        # Be careful when functions in the 1-d case require the standard deviation
        # sqrt(Sigma)!
        #self.mu,self.Sigma=cd.acameanvar(initsize)
        self.mu=initgauss[0]
        self.Sigma=initgauss[1]
        

    
        nutils=NormalUtils(self.mu,self.Sigma)
        
        if self.dim==1:
            self.sigma=np.sqrt(self.Sigma)
        if self.dim==2:
            # Get the Cholesky parameters both as an array of length 3 and 2x2 array 
            self.sigma,self.sigmaL=nutils.getCholesky(self.Sigma)
        
        self.init_matrices(self.mu,self.sigma)
  
      
    def getValue(self,x,matrix,d):
        '''
        Returns for the real x the entry of the matrix (one of gaussprobs,countsfull,fraccompl)
        that represents the bin to which x belongs.
        d (density) is a boolean that specifies whether the value should be divided by the area of the bin
        
        Used for plotting/visualization
        '''
        
        idxs=[]
        for d in range(len(x)):
            idxs.append(getbinidx(self.binbounds[:,d],x[d]))
        """
        Need to reverse the list of indices, because the x- data axis corresponds to the
        columns in the matrices
        """    
        idxs=tuple(reversed(idxs))
        
        
        if matrix=="gaussprobs":
            r=self.gaussprobs[idxs]
        elif matrix=="countsfull":
            r=self.countsfull[idxs]/self.size
        elif matrix=="fraccompl":
            r=self.fraccompl[idxs]/self.size
        elif matrix=="onlycompletion":
            r=(self.fraccompl[idxs]-self.countsfull[idxs])/self.size
        elif isinstance(matrix,np.ndarray):
            r=matrix[idxs]
        else:
            r=0
     
        if d and np.min(idxs)>0 and np.max(idxs)<self.nbins+1: # this only for unbounded bins
            area = 1
            for d in range(self.dim):
                area*=(self.binbounds[idxs[d],d]-self.binbounds[idxs[d]-1,d])
            r=r/area
        return r   
        
 
    def setGaussparams(self,params):
        '''
        Setting the parameters for the Gaussian distribution from the 'internal' vector representation
        params used for gradient descent (i.e., for d=1: the mean and standard deviation, 
        for d=2: the mean and the 3 Cholesky parameters for the covariance)
        '''
        if len(params)==2:
            self.mu=np.array([params[0]])
            self.sigma=params[1]
            self.Sigma=self.sigma**2
        if len(params)==5:
            self.mu=params[[0,1]]
            self.sigma[0]=params[2]
            self.sigma[1]=params[3]
            self.sigma[2]=params[4]
            self.sigmaL[0,0]=params[2]
            self.sigmaL[1,0]=params[3]
            self.sigmaL[1,1]=params[4]
            self.Sigma=np.matmul(self.sigmaL,self.sigmaL.transpose())
            
    def setGaussNaturalParams(self,mu,Sigma):
        """
        Set the Gaussian parameters based on the 'natural' specification as
        mean vector and covariance matrix (or variance)
        """
        nutils=NormalUtils(mu,Sigma)
        self.mu=mu
        self.Sigma=Sigma
        if self.dim==1:
            self.sigma=np.sqrt(self.Sigma)
        if self.dim==2:
            self.sigma,selfsigmaL=nutils.getCholesky(self.Sigma)
            
        
    def setgauss(self):
        # update the gaussprobs matrix based on current mu,Sigma values
        if self.dim==1:
            mn = norm(self.mu,self.sigma)
        if self.dim==2: 
            mn = multivariate_normal(self.mu,self.Sigma)  
     
        
        if self.dim==1:
            # first the cdf:
            for i in range(len(self.gausscdf)):
                self.gausscdf[i]=mn.cdf(self.binbounds[i])
           
            # now the probabilities:
            self.gaussprobs[0]=self.gausscdf[1]
            for i in range(1,self.nbins+2):
                self.gaussprobs[i]=self.gausscdf[i+1]-self.gausscdf[i]
                
        if self.dim==2:

            # Calculate the cdf at all bin boundary points:
            cdfvals = np.zeros((self.nbins+3,self.nbins+3))
            for i in range(self.nbins+3):
                for j in range(self.nbins+3):
                    tick=time.time()
                    cdfvals[j,i]=mn.cdf([self.binbounds[i,0],self.binbounds[j,1]])
                    
                    
            # Now the probabilities:
            for i in range(self.nbins+2):
                for j in range(self.nbins+2):
                    self.gaussprobs[j,i]=cdfvals[j+1,i+1]-cdfvals[j,i+1]\
                    -cdfvals[j+1,i]+cdfvals[j,i]

        # apply correction for numerical inaccuracy leading to negative probs:
        self.gaussprobs=np.maximum(self.gaussprobs,np.zeros(self.gaussprobs.shape))
        
    def setfrac(self):
        '''
        Set the fractional completion (= incremental AI step). Handled separately for mvals and binned case. 
        '''
            
        if self.ctype == 'binned':
            if self.flexmass >0:
                self.fraccompl=np.copy(self.countsfull)
                for i in  range(len(self.data[2])):
                    idxs=self.cbins_to_dbins[i,:]
                    sl=slice(idxs[0],idxs[1]+1)
                    currcompl=self.countsfull[sl]+np.sum(self.fraccomplbins[:,sl],axis=0)\
                              -self.fraccomplbins[i,sl]
                    currmass=np.sum(currcompl)
                    flex=self.data[2][i]
                    q=toprobvec(self.gaussprobs[sl])
                    
                    p=toprobvec(currcompl)*(currmass/(currmass+flex))
                    pnew=utl.distributeflex(p,q)
                    self.fraccomplbins[i,sl]= toprobvec(pnew-p)*flex
                    self.fraccompl+=self.fraccomplbins[i,:]
                    
        if self.ctype=='mvals':
            
            for i in range(2):

                self.setfracslice(0)
                self.setfracslice(1)
              
                self.fraccompl=self.countsfull+np.sum(self.fraccompl2,axis=2) 

            
            
        # correct for possible components that are >0 while gaussprobs==0:
        # self.fraccompl[np.where(self.gaussprobs==0)]=0
        # also round to 0 components that are below a threshold:
        self.fraccompl[np.where(self.fraccompl<1.0e-5)]=0
        
    
    def setfracslice(self,k):
        if k==0:
            for i in range(self.nbins+2):
                flexmass = self.countsmarg[i,k]
                if flexmass >0:
                    q=toprobvec(self.gaussprobs[:,i])
                    currfix=np.sum(self.countsfull[:,i])+np.sum(self.fraccompl2[:,i,1])
                    p=toprobvec(self.countsfull[:,i]+self.fraccompl2[:,i,1])*(currfix/(currfix+flexmass))
                    pnew=utl.distributeflex(p,q)
                    self.fraccompl2[:,i,0]= toprobvec(pnew-p)*flexmass
        if k==1:
            for j in range(self.nbins+2):
                flexmass = self.countsmarg[j,k]
                if flexmass >0:
                    q=toprobvec(self.gaussprobs[j,:])
                    currfix=np.sum(self.countsfull[j,:])+np.sum(self.fraccompl2[j,:,0])
                    p=toprobvec(self.countsfull[j,:]+self.fraccompl2[j,:,0])*(currfix/(currfix+flexmass))
                    pnew=utl.distributeflex(p,q)
                    self.fraccompl2[j,:,1]= toprobvec(pnew-p)*flexmass
    
    def setfrac_expectation(self):
        '''
        Sets the fraccompl according to the expectation given the current gaussprobs
        '''
        self.fraccompl=self.countsfull.copy()
        
        if self.dim==1:
            self.fraccompl+=self.flexmass*self.gaussprobs
            
        if self.dim==2:
            for k in (0,1):
                for i in range(self.nbins+2):
                    flexmass = self.countsmarg[i,k]
                    if flexmass >0:
                        if k==0:
                            q=toprobvec(self.gaussprobs[:,i])
                            self.fraccompl[:,i]+=flexmass*q
                        else:
                            q=toprobvec(self.gaussprobs[i,:])
                            self.fraccompl[i,:]+=flexmass*q



        
    def getKL(self):
        klval=utl.KL_array(self.fraccompl.flatten()/self.size, self.gaussprobs.flatten())
        return klval
    
    def getEntropy(self):
        nonzeros=np.where(self.fraccompl>0)
        empdistr=self.fraccompl[nonzeros]/np.sum(self.fraccompl[nonzeros])
        return - np.sum(empdistr*np.log(empdistr))
        
    def getKLmap(self):
        if self.dim==1:
            klmap=np.zeros(self.nbins+2)
            for i in range(self.nbins+2):
                klmap[i]=utl.klbase(self.fraccompl[i]/self.size,self.gaussprobs[i])
            
        if self.dim==2:    
            klmap=np.zeros((self.nbins+2,self.nbins+2))
            for i in range(1,self.nbins+1):
                for j in range(1,self.nbins+1):
                    klmap[j,i]=utl.klbase(self.fraccompl[j,i]/self.size,self.gaussprobs[j,i])
        return klmap        
    
    def testKL(self):
        '''
        Only for debugging
        '''
        if np.sum(self.fraccompl[np.where(self.gaussprobs==0)])>0:
            print("Infinite KL:")
            print("fraccompl is {}".format(self.fraccompl[np.where(self.gaussprobs==0)]))
            print("at indices {}".format(np.where(self.gaussprobs==0)))
        else:
            print("KL o.k.: {}".format(self.getKL()))
            print("Weight check: {},{}".format(np.sum(self.fraccompl.flatten()/self.size), np.sum(self.gaussprobs.flatten())))
                 
    
    def getDensity(self,a):
        '''
        Returns a matrix representing the density version of the matrix a (typically the current
        gaussprob matrix):
        a must be a (nbins+2)x(nbins+2) matrix 
        For bounded bins, result contains the probability divided by the area of the bin.
        Unbounded bins have value 0 (only included in return matrix for format compatibility)
        Values are not normalized by dividing by total area
        '''
        
        if self.dim==1:
            result = np.zeros(self.nbins+2)
            for i in range(1,self.nbins+1):
                result[i]=a[i]/(self.binbounds[i]-self.binbounds[i-1])
                
        if self.dim==2:
            result = np.zeros((self.nbins+2,self.nbins+2))
            for i in range(1,self.nbins+1):
                for j in range(1,self.nbins+1):
                    result[j,i]=a[j,i]/((self.binbounds[j,1]-self.binbounds[j-1,1])\
                                        *(self.binbounds[i,0]-self.binbounds[i-1,0]))
        return result      
    

        
    def init_matrices(self,mu,Sigma):
        
        if self.ctype=='binned':
            self.countsfull = np.zeros((self.nbins+2))
            self.gaussprobs=np.zeros((self.nbins+2))
            self.gausscdf=np.zeros((self.nbins+3))
            self.fraccompl=np.zeros(self.nbins+2)
            # fraccomplbins contains the fractional completions of the coarse data bin counts
            # over the discretization bins. This will be a "near diagonal" matrix, because 
            # completions of the i'th coarse data bin must be contained in the range
            # self.cbins_to_dbins[i,:] of discretization bins.
            self.fraccomplbins = np.zeros((len(self.data[2]),self.nbins+2))
            self.gaussgrad=np.zeros((self.nbins+2,2))# gradient with regard to (mu,sigma)
            
        if self.ctype=='mvals':
            self.countsfull = np.zeros((self.nbins+2,self.nbins+2))
            self.gaussprobs=np.zeros((self.nbins+2,self.nbins+2))
            self.countsmarg = np.zeros((self.nbins+2,self.dim))
            self.fraccompl=np.zeros((self.nbins+2,self.nbins+2))      
            self.fraccompl2=np.zeros((self.nbins+2,self.nbins+2,2))
            #gradient with regard to the 
            #5 components (mu[0],mu[1],sigma[0,0],sigma[1,1],sigma[1,2])
            #gradient also evaluated at the virtual 'infinites' binbounds
            #therefore here dimensions self.nbins+2, not self.nbins+1 as in self.dim=1 case
            self.gaussgrad=np.zeros((self.nbins+2,self.nbins+2,5)) 
            
        # Initialize the count matrices
        # In the following assuming that only one component can be nan
        if self.ctype=='binned':
            for m in range(len(self.data[0])):
                idx=getbinidx(self.binbounds,self.data[0][m])
                self.countsfull[idx]+=1   
        if self.ctype=='mvals':
            for m in range(self.data.shape[0]):
                if np.isnan(self.data[m,0]): 
                    idx=getbinidx(self.binbounds[:,1],self.data[m,1])
                    self.countsmarg[idx,1]+=1     
                elif np.isnan(self.data[m,1]):
                    idx=getbinidx(self.binbounds[:,0],self.data[m,0])
                    self.countsmarg[idx,0]+=1
                else:
                    idx0=getbinidx(self.binbounds[:,0],self.data[m,0])
                    idx1=getbinidx(self.binbounds[:,1],self.data[m,1])
                    self.countsfull[idx1,idx0]+=1
            
                        
        # Initialize matrix for bin-probabilities according to Gaussian distribution:
        self.setgauss()
    
    
        
 
    def setgrad(self):
        # compute Gaussian probabilities and the partial derivatives of the probabilities of the bins
        # 
        # for d=1 the distribution parameters are mu and sigma 
        # 
        # for d=2 the parameters are:
        #     mu[0],mu[1],l11,l21,l22 
        # where l11,l21,l22 are the entries in the Cholesky decomposition of the
        # covariance matrix Sigma: Sigma=LL^T, where L=sigmaL=[[l11,0],[l21,l22]]
        # The choice of a lower triangular from in the Cholesky decomposition leads
        # to an asymmetric treatment of the parameters mu[0] and mu[1], and l11 and l22
       
        
        nutils = NormalUtils(self.mu,self.Sigma) 
                
        if self.dim==1:
            sn = norm()
            zleft=nutils.z(self.binbounds[0])
            for i in range(self.nbins+2):
                zright=nutils.z(self.binbounds[i+1])
                
                dPhidzleft=sn.pdf(zleft)
                dPhidzright=sn.pdf(zright)
                  
                # gradient with respect to mu and sigma    
                self.gaussgrad[i,0]=(dPhidzright-dPhidzleft)*(-1/self.sigma)
                self.gaussgrad[i,1]=dPhidzright*(-zright/self.sigma)- dPhidzleft*(-zleft/self.sigma)
                
                zleft=zright
                
        if self.dim==2:
            
            sn = norm()
            
            # The transformed x-coordinates binbounds[:,0]:
            txbounds = nutils.zx(self.binbounds[:,0])
            # The transformed y-coordinates of 2-d binbounds (binbounds[j,0],binbounds[i,1])
            tybounds=np.zeros((self.nbins+3,self.nbins+3))
            for j in range(self.nbins+3):
                lf= lambda x: nutils.linfunc(self.binbounds[j,1],x)
                for i in range(self.nbins+3):
                    tybounds[j,i] = lf(txbounds[i])
                                    
            
            # First the partial derivatives w.r.t. sigmaL[0,0] and mu[0]
            
            # The density values of the standard normal at the transformed binbounds[:,0]   
            phivals = sn.pdf(txbounds)
                   
            for j in range(self.nbins+2):
                for i in range(self.nbins+2):
                    
                    gaussprobright=sn.cdf(tybounds[j+1,i+1])-sn.cdf(tybounds[j,i+1])
                    gaussprobleft=sn.cdf(tybounds[j+1,i])-sn.cdf(tybounds[j,i])
                    self.gaussgrad[j,i,2]=\
                    phivals[i+1]*gaussprobright*(self.mu[0]-self.binbounds[i+1,0])/self.sigma[0]**2\
                    -phivals[i]*gaussprobleft*(self.mu[0]-self.binbounds[i,0])/self.sigma[0]**2
                    self.gaussgrad[j,i,0]=\
                    phivals[i+1]*gaussprobright*(-1/self.sigma[0])\
                    -phivals[i]*gaussprobleft*(-1/self.sigma[0])
               
            
            # Now the probabilities and the partial derivatives w.r.t. mu[1], sigmaL[1,0], sigmaL[1,1]
            # This uses an approximation of the integral by averaging over self.numticks values 
            # at the self.xticks
            
            # The transformed xticks and their density values:
            tticksbounds = nutils.zx(self.xticks)
            phivals = sn.pdf(tticksbounds)
            
            # The gradients at the pseudo -infty y-bounds: 
            dPhidL=np.vectorize(nutils.derivgcdf)\
                                (nutils.linfunc(self.binbounds[0,1],tticksbounds))
            dLdparams=np.zeros((3,tticksbounds.shape[0]))
            for t in range(tticksbounds.shape[0]):
                dLdparams[:,t]=nutils.gradlinfunc(self.binbounds[0,1],tticksbounds[t])
            gradlower= phivals*dPhidL*dLdparams
            
            for j in range(self.nbins+2):
                dPhidL=np.vectorize(nutils.derivgcdf)\
                                (nutils.linfunc(self.binbounds[j+1,1],tticksbounds))
                dLdparams=np.zeros((3,tticksbounds.shape[0]))
                for t in range(tticksbounds.shape[0]):
                    dLdparams[:,t]=nutils.gradlinfunc(self.binbounds[j+1,1],tticksbounds[t])
                gradupper= phivals*dPhidL*dLdparams
                
                for i in range(self.nbins+2):
                    grad=np.zeros(3)
                    startidx=i*(self.numticks-1)
                    for h in range(self.numticks):
                        grad+=gradupper[:,startidx+h]-gradlower[:,startidx+h]
                    grad/=self.numticks
                    self.gaussgrad[j,i,(1,3,4)]=grad
                    
                gradlower=gradupper
                
       
          
            
    def getKLgrad(self):
        """
        Get the gradient of the KL distance between fraccompl
        and gaussprobs with regard to the distribution parameters
        """
        if self.dim==1:
           
            grad=np.zeros(2)
            for i in range(self.nbins+2):
                if self.gaussprobs[i]>0:
                    # The gradient will be normalized in the M step, so 
                    # no need to normalize self.fraccompl to a probability
                    # distribution here
                    grad+=(-self.fraccompl[i]/self.gaussprobs[i])*self.gaussgrad[i]
                else:
                    if self.fraccompl[i] >0:
                        print("Warning: infinite KL encountered!") 
                        
        if self.dim==2:
                        
            grad=np.zeros(5)
            for j in range(self.nbins+2):
                for i in range(self.nbins+2):
                    if self.gaussprobs[j,i]>0:
                        grad+=(-self.fraccompl[j,i]/self.gaussprobs[j,i])*self.gaussgrad[j,i]
                    else:
                        if self.fraccompl[j,i] >0:
                            print("Warning: infinite KL encountered!")
                     
        return grad
    
    def setGaussandgetKL(self,p):
        """
        p an array of two, resp five,  parameters
        """
        self.setGaussparams(p)
        self.setgauss()
        return self.getKL()
    
    def Mstep(self,stepsize,tval):
        # Perform a linesearch for the parameters (self.mu, self.sigma) minimizing KL(fraccompl,gaussprobs(mu,sigma))
        # in the direction of the current gradient
        # stepsize: initial step for linesearch in direction of gradient
        # tval: threshold for termination of linesearch
        
        if (not np.isfinite(self.getKL()) ):
            print("Infinite KL at beginning of Mstep; Safe reset of mu,Sigma")
            mu,Sigma=self.muSigma_from_fraccompl()
            self.setGaussNaturalParams(mu,Sigma)
            self.setgauss()
            return self.mu,self.Sigma
            
        self.setgrad()
        
        if self.dim==1:
            current=np.hstack([self.mu,self.sigma])
        if self.dim==2:
            current=np.hstack([self.mu,self.sigma])
        
        # Note: this is the gradient of the KL distance, 
        grad=self.getKLgrad()
        
        if (np.linalg.norm(grad)>0):
            grad*=(1/np.linalg.norm(grad))
        
        leftbound=current
        
        leftkl=self.setGaussandgetKL(current)
        
      
        rightbound=current-stepsize*grad
        rightkl=self.setGaussandgetKL(rightbound)
        
        
        lastrightkl=leftkl
        newstep=stepsize
        
       
        while rightkl<lastrightkl:
            lastrightkl=rightkl
            newstep*=2
            rightbound-=newstep*grad
            rightkl=self.setGaussandgetKL(rightbound)
           
        terminate=False
        
        #print("initial leftb: {}  rightb: {}".format(leftbound,rightbound))
        
        while not terminate:
            midbound=0.5*(rightbound+leftbound)
            midkl=self.setGaussandgetKL(midbound)
            #print("KL: left {}  mid {} right {}".format(leftkl,midkl,rightkl))
            if midkl>leftkl:
                rightbound=midbound
                rightkl=midkl
            else:
                midgrad=self.getKLgrad()
                #print("midgrad: {}  dot: {}".format(midgrad,np.dot(grad,midgrad)))
                if np.dot(grad,midgrad)>0:
                    leftbound=midbound
                    leftkl=midkl
                else:
                    rightbound=midbound
                    rightkl=midkl
            terminate=np.linalg.norm(rightbound-leftbound)<tval
       
        if rightkl<leftkl:
            self.setGaussparams(rightbound)
        else:
            self.setGaussparams(leftbound)
        self.setgauss()
        
        #print("mu,Sigma after M step: {},{}".format(self.mu,self.Sigma))
        
        return self.mu,self.Sigma    
    
    def muSigma_from_fraccompl(self):
        """
        Computes a rough estimate of mu and Sigma from the current fraccompl.
        This is a very rough estimate intended only as a 'safe' initialization.
        - considers only the marginals in each dimension, so that Sigma is diagonal in the 2d case
        - bins are identified with their center points
        """
        mu=[]
        Sigma=[]
        for d in range(self.dim):
            refpoints=np.zeros(self.fraccompl.shape[d])
            for i in range(len(refpoints)):
                refpoints[i]=(self.binbounds[i,d]+self.binbounds[i+1,d])/2
            #print("refpoints: {}".format(refpoints))
            if self.dim==1:
                fracmarg=self.fraccompl
            if self.dim==2:
                fracmarg=np.sum(self.fraccompl,axis=1-d)
            #print("fraccompl: {}".format(fracmarg))
            mu_d=np.sum(refpoints*fracmarg)/self.size
            mu.append(mu_d)
            Sigma_d=np.sum(fracmarg*((refpoints-mu_d)**2))/self.size
            Sigma.append(Sigma_d)
            
        mu=np.array(mu)
        Sigma=np.diag(Sigma)
        return mu,Sigma    
        
def toprobvec(z):
    if np.sum(z)== 0: # in this case return uniform distribution vector
        return np.ones(z.size)/z.size
    return z/np.sum(z)
    
def tocountvec(p,c):
        return c*p          
    
def plotheat(dcd):
    # Only for 2-d dcd
    toplot = (dcd.gaussprobs,dcd.getDensity(dcd.gaussprobs),dcd.fraccompl,dcd.getKLmap())
    titles = ("gaussprobs","gaussdensity","fraccompl","KLmap")
    fig,axes=plt.subplots(1,4,figsize=(20,3))
    for ax,m,t in zip(axes.ravel(),toplot,titles):
        #ax.imshow(m)
        ax.set_title(t)
        sns.heatmap(m,ax=ax)
    plt.show()
    
def plotstate(dcd,toplot,coords,d):
    '''
    toplot a list of strings from "gaussprobs","fraccompl","countsfull","KLmap"
    '''
    if dcd.dim==1:
        plotstate1d(dcd,toplot,coords,d)
    if dcd.dim==2:
        plotstate2d(dcd,toplot,coords,d)
        
def plotstate1d(dcd,toplot,coords,d):
    xrange=coords
    for st in toplot:
        if (st=="gaussprobs"):
            lstring=r'$P_{\theta}$'
        elif (st=="fraccompl"):
            lstring=r'$P_c$'
        elif (st=="countsfull"):
            lstring="exact observations"
        elif (st=="onlycompletion"):
            lstring="completion of binned"
        else:
            lstring=st
        plt.plot(xrange,vals_for_real(dcd,xrange,st,d),label=lstring)
#     plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),\
#           ncol=4, fontsize=15, fancybox=True, shadow=True) # use this for separate legend outside    
    plt.legend()
    plt.show()
    
def plotstate2d(dcd,toplot,coords,d): 
    numcols=2
    numrows=int(np.ceil(len(toplot)/numcols))
    fig,axes=plt.subplots(numrows,numcols,figsize=(15,5*numrows))
    for ax,tp in zip(axes.ravel(),toplot):
        #ax.imshow(m)
        if not isinstance(tp,np.ndarray):
            ax.set_title(tp)
        im=ax.pcolormesh(coords[0],coords[1],vals_for_real(dcd,coords,tp,False))
        fig.colorbar(im,ax=ax)
    plt.show()
    
        
def vals_for_real(dcd,coords,matrix,d):
    '''
    matrix is a string
    d a boolean specifying whether absolute or density values should be returned
    when dcd is 1-d: coords contains an array
    when dcd is 2-d: coords=(xx,yy) containing 2-d arrays of the same shape
    '''
    if dcd.dim ==1:
        r=np.zeros(len(coords))
        for i in range(r.shape[0]):
            r[i]=dcd.getValue(np.array([coords[i]]),matrix,d)
            
    if dcd.dim ==2:
        r=np.zeros(coords[0].shape)
        for j in range(r.shape[0]):
            for i in range(r.shape[1]):
                a = dcd.getValue([coords[0][j,i],coords[1][j,i]],matrix,d)
                r[j,i]=a
    return r                  

def dmS(mu1,Sigma1,mu2,Sigma2):
    '''
    Computes a difference between the two parameters (mu1,Sigma1) and (mu2,Sigma2)
    '''
    dS=np.max(np.abs(Sigma1-Sigma2))
    dm=np.max(np.abs(mu1-mu2))
    
    return np.max((dS,dm))

     
    
def aim(cd, # coarse data
        initgauss, # initial mu,Sigma values
        nbins=10, # number of bins 
        nticks=5, # number of "ticks" per bin for approximating integral
        e=1.05, # kl termination threshold 
        maxits=20, # max iteration termination condition
        lrate=0.05, # "learning rate" for M step
        tv=0.001, # distance threshold for termination of linesearch in M step
        toplot=None, # list of things to plot at each iteration
        coords=None, # coordinates for plotting (array or meshgrid)
        plotdensity=False, # Boolean whether plots should use densities rather than absolute values
        **kwargs):
  
    dcd = dcdata(cd,nbins,nticks,initgauss,**kwargs)
              
    #dcd.printme()
    #print("Initial mu,Sigma: {},{}".format(dcd.mu,dcd.Sigma))
    klvals=[]
    allSigmas=[]
    allmus=[]
    terminate=False
  
    oldkl= float("INF")
    
    itcount = 0
    
    timeAI=0
    timeM=0
    
    while not terminate:    
        itcount+=1    
        #print("fraccompl sum: {}".format(np.sum(dcd.fraccompl)))
        
        tick=time.time()
        dcd.setfrac()
        timeAI+= time.time()-tick
        
        currentkl=dcd.getKL()
        klvals.append(currentkl) 
        
        tick=time.time()
        mu,Sigma=dcd.Mstep(lrate,tv)
        timeM+= time.time()-tick
        
        if not toplot==None:
            plotstate(dcd,toplot,coords,plotdensity)  
            print("t: {} mu: {} \n Sigma: {}".format(itcount,mu,Sigma))
            
        allSigmas.append(Sigma)
        allmus.append(mu)
            
        if (currentkl==0) or (oldkl/currentkl < e) or itcount == maxits:
            terminate=True
        else:
            oldkl=currentkl
           
    print("bins: {} its: {}  Time AI: {}   Time M: {}  Percentage M: {}".\
          format(nbins,itcount,timeAI,timeM,timeM/(timeAI+timeM))) 
    
    
    #return dcd,mu,Sigma,currentkl,np.stack(allSigmas),np.stack(allmus),klvals
    return dcd,mu,Sigma,currentkl,np.stack(allSigmas),np.stack(allmus),klvals

def score(true,learned,indx):
    if indx==-1:
        if len(true)==6: # delete one of Sigma[0,1]=Sigma[1,0] components for scoring
            true=np.delete(true,4)
            learned=np.delete(learned,4)
        return np.sum((true-learned)**2)
    else:
        return np.sqrt(np.sum((true[indx]-learned[indx])**2))
