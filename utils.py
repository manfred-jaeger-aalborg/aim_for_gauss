import numpy as np

def klbase(p,q):
    if p==0:
        return 0
    else:
        result =  p*np.log(p/q)
        if q==0 and p>0:
            print("Warning: infinite kl in klbase for p: {} q: {}".format(p,q))
        if np.isnan(result):
            print("Warning: nan in klbase for p: {} q: {}".format(p,q))
        return result

    
def KL_array(p,q):
    #print("KL_array {} , {}".format(p,q))
    if len(p)!=len(q):
        print("arrays of unequal length in KL_array")
        return None
    kl=0
    for i in range(len(p)):
        kl+=klbase(p[i],q[i])
    if kl<0:
        print("Negative kl! sums: p: {} q: {}".format(np.sum(p),np.sum(q)))
    return kl#/np.log(len(p))    
    
def KL_comps(p,q):
    if len(p)!=len(q):
        print("arrays of unequal length in KL_array")
        return None
    result=np.zeros(len(p))
    for i in range(len(p)):
        result[i]=klbase(p[i],q[i])
    return result    
    
def distributeflex(p,q):
    '''
    q: a probability vector
    p: a sub-probability vector with the same length as q. p.sum <= 1
    returns: vector pfull for which pfull.sum=1, pfull[i]>=p[i] for all i, 
    and pfull minimizes KL(p,q) under these constraints
    '''
    #print("distributeflex with P: {} \n  q: {}".format(p,q))

    qzeros=np.where(q==0)
    qnzeros=np.where(q!=0)
    if len(qzeros[0])>0:
        #print("warning: q vector has zero components {}".format(qzeros))
        if np.max(p[qzeros])>0:
            print("Warning: distributeflex with q[i]=0<p[i]")
        
    pnz=p[qnzeros]
    qnz=q[qnzeros]
    
    if np.max(pnz/qnz) <= 1:
        return q
    
    
    resultnz = np.copy(pnz)
    flexleft = 1-np.sum(p)

    #iterations =0
    while flexleft >0:
        #print("flexleft: {}".format(flexleft))
        ratios=np.around(resultnz/qnz,decimals=12)

        rvals=np.sort(np.unique(ratios))
       
        mins=np.where(ratios<=rvals[0])[0]
        #print("mins: {}".format(mins))
        
        psub=resultnz[mins]
        qsub=qnz[mins]
        #print("psub: {}".format(psub))
        #print("qsub: {}".format(qsub))
        

        if len(rvals)==1:
            print("Warning: rvals={} from inputs \n p: {} \n {} q: {}".format(rvals,p,q))
            
        d=rvals[1]*qsub-psub
        if np.any(d<0):
            print("Warning: negative values in d vector: {}".format(d))
            print("rvals: {}".format(rvals))
            print("psub: {}".format(psub))
            print("qsub: {}".format(qsub))
            
        dsum=np.sum(d)
        #print("d: {}".format(d))
            
        if dsum < flexleft:
            resultnz[mins]+=d
            flexleft-= dsum  
        else:
            resultnz[mins]+=(flexleft/np.sum(qsub))*qsub
            flexleft=0

        #iterations+=1    
   # s=np.sum(result[np.where(q==0)])
   #  if s>0:
   #      print("Warning: positive mass on zero components of q")
   #      print("p: {}".format(p[np.where(q==0)]))
   #      print("Result: {}".format(result[np.where(q==0)]))

   #  if result[np.where(result<0)].size >0:
   #      print("Warning: negative components in result of distributeflex")
   #      print("p: {}".format(p))
   #      print("q: {}".format(q))

    result=np.copy(p)
    result[qnzeros]=resultnz

    #print("result: {}".format(result))
    
    return result




