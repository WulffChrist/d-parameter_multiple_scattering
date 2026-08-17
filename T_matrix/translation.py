import numpy as np
from scipy.special import lpmv,spherical_jn,spherical_yn,factorial
from sympy.physics.wigner import gaunt

def jl(l,rho,der=False):
        if der:
            return spherical_jn(l,rho,True)
        else:
            return spherical_jn(l,rho)

def h1l(l,rho,der=False):
    if der:
        return spherical_jn(l,rho,True)+1j*spherical_yn(l,rho,True)
    else:
        return spherical_jn(l,rho)+1j*spherical_yn(l,rho)

def ap(l,m):
    return np.sqrt(((l+m+1)*(l-m+1))/((2*l+1)*(2*l+3)))

def am(l,m):
    return np.sqrt(((l+m)*(l-m))/((2*l+1)*(2*l-1)))

def bp(l,m):
    return np.sqrt(((l+m+2)*(l+m+1))/((2*l+1)*(2*l+3)))

def bm(l,m):
    return np.sqrt(((l-m)*(l-m-1))/((2*l+1)*(2*l-1)))

def beta0(l,m,d,k):
    x = np.cos(d[1])
    normalization = np.sqrt( ( (2*l+1) / (4*np.pi) ) * (factorial(l-m) / factorial(l+m)) )
    return np.sqrt(4*np.pi)*(-1)**(l+m) * normalization * lpmv(-m,l,x) * np.exp(-1j*m*d[2]) * jl(l,k*d[0])

def alpha0(l,m,d,k):
    x = np.cos(d[1])
    normalization = np.sqrt( ( (2*l+1) / (4*np.pi) ) * (factorial(l-m) / factorial(l+m)) )
    return np.sqrt(4*np.pi)*(-1)**(l+m) * normalization * lpmv(-m,l,x) * np.exp(-1j*m*d[2]) * h1l(l,k*d[0])

def indx(l,m):
    return l*(l + 1) + m

def scalar(lstop,d,k,type):
    Nomega = np.size(k)
    lmodes = lstop**2+2*lstop

    lstopp = 2*lstop
    lmodesp = (2*lstopp)**2+2*(2*lstopp)

    alpha = np.zeros((Nomega,lmodesp+1,lmodes+1), dtype=np.complex64)
    alphastar = np.zeros((Nomega,lmodesp+1,lmodes+1), dtype=np.complex64)

    for lp in range(0,lstopp+1):
        for mp in range(-lp,lp+1):
            q = indx(lp,mp)

            if type == 1:
                alpha[:,q,0] = beta0(lp,mp,d,k)
                alphastar[:,q,0] = beta0(lp,mp,d,np.conj(k))
            elif type == 3:
                alpha[:,q,0] = alpha0(lp,mp,d,k)
                alphastar[:,q,0] = (-1)**lp*alpha0(lp,mp,d,np.conj(k))
            else:
                print('this type does not exist')
                exit

    for l in range(1,lstop+1):

        for m in range(0,l+1):
            p = indx(l,m)

            for lp in range(0,lstopp+1-l):
                if m==0:
                    mpstart = 0
                else:
                    mpstart = -lp
                
                for mp in range(mpstart,lp+1):
                    q = indx(lp,mp)

                    if l == m:
                        if bm(lp+1,mp-1) == 0:
                            alphalp = 0
                            alphalpstar = 0
                        else:
                            alphalp = alpha[:,indx(lp+1,mp-1),indx(l-1,m-1)]
                            alphalpstar = alphastar[:,indx(lp+1,mp-1),indx(l-1,m-1)]
                        
                        alpha[:,q,p] = (bp(lp-1,mp-1)*alpha[:,indx(lp-1,mp-1),indx(l-1,m-1)]+bm(lp+1,mp-1)*alphalp)/bp(l-1,m-1)
                        alphastar[:,q,p] = (bp(lp-1,mp-1)*alphastar[:,indx(lp-1,mp-1),indx(l-1,m-1)]+bm(lp+1,mp-1)*alphalpstar)/bp(l-1,m-1)

                    else:
                        if am(l-1,m) == 0:
                            alphal = 0
                            alphalstar = 0
                        else:
                            alphal = alpha[:,q,indx(l-2,m)]
                            alphalstar = alphastar[:,q,indx(l-2,m)]
                        if ap(lp-1,mp) == 0:
                            alphalp = 0
                            alphalpstar = 0
                        else:
                            alphalp = alpha[:,indx(lp-1,mp),indx(l-1,m)]
                            alphalpstar = alphastar[:,indx(lp-1,mp),indx(l-1,m)]
                        
                        alpha[:,q,p] = (-am(l-1,m)*alphal+ap(lp-1,mp)*alphalp+am(lp+1,mp)*alpha[:,indx(lp+1,mp),indx(l-1,m)])/ap(l-1,m)
                        alphastar[:,q,p] = (-am(l-1,m)*alphalstar+ap(lp-1,mp)*alphalpstar+am(lp+1,mp)*alphastar[:,indx(lp+1,mp),indx(l-1,m)])/ap(l-1,m)
                    
                    if m>0 or mp>0:
                        
                        if type == 1:
                            alpha[:,q-2*mp,p-2*m]=(-1)**(mp+m)*np.conj(alphastar[:,q,p])
                        if type == 3:
                            alpha[:,q-2*mp,p-2*m]=-(-1)**(l+lp+mp+m)*np.conj(alphastar[:,q,p])
    return alpha[:,0:lmodes+1,: ]

def Cmatrix(lstop,d,k,type):
    Nomega = np.size(k)
    lmodes = 2*(lstop**2+2*lstop)

    C = np.zeros((Nomega,lmodes,lmodes), dtype=np.complex64)
    alpha = scalar(lstop,d,k,type)
    
    for l in range(1,lstop+1):

        for m in range(-l,l+1):
            p = 2*(indx(l,m)-1)

            for lp in range(1,lstop+1):

                for mp in range(-lp,lp+1):
                    q = 2*(indx(lp,mp)-1)


                    term2A = (np.sqrt( (l-m) * (l+m+1) )*np.sqrt( (lp-mp) * (lp+mp+1) )*alpha[:,indx(lp,mp+1),indx(l,m+1)]
                            if ( (l-m) * (l+m+1)  * (lp-mp) * (lp+mp+1) ) != 0 else 0 )    
                    term3A = (np.sqrt( (l+m) * (l-m+1) )*np.sqrt( (lp+mp) * (lp-mp+1) )*alpha[:,indx(lp,mp-1),indx(l,m-1)]
                            if ( (l+m) * (l-m+1) * (lp+mp) * (lp-mp+1) ) != 0 else 0 )
                    

                    C[:,q,p] = (
                        1/2 * np.sqrt( 1 / ( lp*(lp+1) * l*(l+1) ) ) 
                        * ( 2*mp*m * alpha[:,indx(lp,mp),indx(l,m)]
                        + term2A
                        + term3A )
                    )

                    term1B = (2*m*np.sqrt( (lp-mp) * (lp+mp) ) * alpha[:,indx(lp-1,mp),indx(l,m)]
                            if ( (lp-mp) * (lp+mp) ) != 0 else 0)
                    term2B = (np.sqrt( (l-m) * (l+m+1) )*np.sqrt( (lp-mp) * (lp-mp-1) )*alpha[:,indx(lp-1,mp+1),indx(l,m+1)]
                            if ( (l-m) * (l+m+1) * (lp-mp) * (lp-mp-1) ) != 0 else 0)       
                    term3B = (np.sqrt( (l+m) * (l-m+1) )*np.sqrt( (lp+mp) * (lp+mp-1) )*alpha[:,indx(lp-1,mp-1),indx(l,m-1)]
                            if ( (l+m) * (l-m+1) * (lp+mp) * (lp+mp-1) ) != 0 else 0)
                    
                    C[:,q+1,p] = (
                        -1j*1/2 * np.sqrt( (2*lp+1) / (2*lp-1) * 1 / ( lp*(lp+1) * l*(l+1) ) ) 
                        * ( term1B
                        + term2B
                        + term3B )
                    )

                    C[:,q+1,p+1] = C[:,q,p]
                    C[:,q,p+1] =  C[:,q+1,p]
    
    return C

def gaunt_coeffs(lstop): #gaunt coefficients Chew eq. (D.23b)
    lmodes = lstop**2+2*lstop # total modes without polarization

    lpps = 2*lstop+1 # approximate amount of total l''

    A = np.zeros((lmodes,lmodes,lpps))
    
    for l in range(1,lstop+1):

        for m in range(-l,l+1):
            p = indx(l,m)-1 #index for l and m, -1 since we start at l and l' = 1

            for lp in range(1,lstop+1):
                
                for mp in range(-lp,lp+1):
                    q = indx(lp,mp)-1 # index for l' and m'

                    for lpp in range(lpps):
                        # Chew eq. (D.23b)
                        A[q,p,lpp] = (-1)**m*gaunt(l,lp,lpp,-m,mp,m-mp,prec=64) # using sympy which doesnt include (-1)^m by itself
    return A

def abcoeff(lstop): # coefficients to be multiplied by spherical harmonics in Chew eqs. (D22) and (D23)
    lmodes = lstop**2+2*lstop # total number of modes withour polarization
    lppmax = lstop+lstop # approximate number of maximum l''. Just needs to be higher than the amount needed
    
    a = np.zeros((lmodes,lmodes,lppmax), dtype=np.complex128) #for making A coefficients
    b = np.zeros((lmodes,lmodes,lppmax), dtype=np.complex128) #for making B coefficients
    alpps = np.zeros((lmodes,lmodes,lppmax),dtype=int) #l'' for A coefficients
    blpps = np.zeros((lmodes,lmodes,lppmax),dtype=int) #l'' for B coefficents
    
    A = gaunt_coeffs(lstop)

    for l in range(1,lstop+1):

        for m in range(-l,l+1):
            p = indx(l,m)-1 #index for l and m, -1 since we start at l and l' = 1

            for lp in range(1,lstop+1):

                for mp in range(-lp,lp+1):
                    q = indx(lp,mp)-1 #index for l' and m'
                    
                    aidx = 0 # a index
                    bidx = 0 # b index

                    for lpp in range(np.abs(l-lp),l+lp+2): # range of l''
                        fac = 1j ** (lp-l) * 4 * np.pi / ( 2*lp * (lp+1)) # factor in front of sums in eqs. (D.22) and (D.23)
                        
                        if (((lpp + lp + l) % 2) != 0): # if sum is odd B coefficients are non zero due to l''-1
                            blpps[q,p,bidx] = lpp
                            if lpp == 0: # gaunt coeff is 0 if l'' - 1 = -1, thus if l'' is 0 then B = 0
                                B = 0
                            else:            
                                term1B = (-np.sqrt( (lp-mp) * (lp+mp+1) * (lpp+m-mp) * (lpp+m-mp-1) ) * A[indx(lp,mp+1)-1,p,lpp-1] # term 1
                                        if ( (lp-mp) * (lp+mp+1) * (lpp+m-mp) * (lpp+m-mp-1) ) != 0 else 0) # check to avoid NaN values or complaints from np.sqrt function
                                
                                term2B = (np.sqrt( (lp+mp) * (lp-mp+1) * (lpp-m+mp) * (lpp-m+mp-1) ) * A[indx(lp,mp-1)-1,p,lpp-1] # term 2
                                        if ( (lp+mp) * (lp-mp+1) * (lpp-m+mp) * (lpp-m+mp-1) ) != 0 else 0)# check to avoid NaN values or complaints from np.sqrt function
                                
                                term3B = ( 2*mp * np.sqrt( (lpp-m+mp) * (lpp+m-mp) ) * A[q,p,lpp-1] # term 3
                                        if ( (lpp-m+mp) * (lpp+m-mp) ) > 0 else 0)# check to avoid NaN values or complaints from np.sqrt function

                                B = np.sqrt( ( 2*lpp+1 ) / ( 2*lpp-1 ) ) * ( term1B + term2B + term3B ) # with prefactor
                                
                            if (np.abs(m-mp) > lpp): # This condition is necessary or associated legendre polynomials won't work
                                b[q,p,bidx] = 0
                            else:
                                b[q,p,bidx] = fac * 1j**(lpp) * B # prefactor inside sum
                            bidx += 1
                                
                        else: # if sum is even A coefficients are non zero
                            alpps[q,p,aidx] = lpp
                            
                            if (np.abs(m-mp) > lpp): # This condition must be fulfilled
                                a[q,p,aidx] = 0
                            else:
                                a[q,p,aidx] = fac * 1j**(lpp) * ( l*( l+1 ) + lp*( lp+1 ) - lpp*( lpp+1 ) ) * A[q,p,lpp] # prefactor inside sum
                            aidx += 1
    return a,b,alpps,blpps

def Cmatrix_chew(lstop,d,k,type,a,b,alpps,blpps): # full addition translation coefficients of Chew eqs (D.22) and (D.23)
    # a,b,alpps,blpps as inputs so we only need to calculate them once
    Nomega = np.size(k)
    lmodes = 2*(lstop**2+2*lstop)

    C = np.zeros((Nomega,lmodes,lmodes), dtype=np.complex128) # matrix structure same as single sphere t-matrices
    
    #a,b,alpps,blpps = abcoeff(lstop)
    for l in range(1,lstop+1):

        for m in range(-l,l+1):
            p = 2*(indx(l,m)-1)# indices account for polarization -1 since we start at l=1

            for lp in range(1,lstop+1):

                for mp in range(-lp,lp+1):
                    q = 2*(indx(lp,mp)-1)# indices account for polarization

                    sumA = 0 # sum for A coefficients
                    sumB = 0 # sum for B coefficients
                    for i in range(np.size(alpps,axis=2)): # number of l''
                        lpp = alpps[int(q/2),int(p/2),i] # value of l'', Indices are halved since alpps does not depend on polarization
                        if (i > 0 and lpp == 0) or (np.abs(m-mp) > lpp): #ensuring we don't go outside of relevant l'' and m m' condition
                            continue
                        else:
                            if type == 1:
                                bessel = jl(lpp,k*d[0]) # both types of bessel functions
                            if type == 3:
                                bessel = h1l(lpp,k*d[0])
                            
                            if lpp == 0:
                                normalization = np.sqrt(1.0 / (4.0 * np.pi)) # normalizaiton for spherical harmonics l''=0
                            else:
                                normalization = np.sqrt( ( (2*lpp+1) / (4*np.pi) ) * ( factorial(lpp-(m-mp)) / factorial(lpp+(m-mp)) ) ) # and for l'' > 0
                            
                            swf = normalization * lpmv(m-mp,lpp,np.cos(d[1])) * np.exp(1j*(m-mp)*d[2]) * bessel # scalar wave functions
                            
                            sumA += a[int(q/2),int(p/2),i]*swf # summing for A
                        
                    for i in range(np.size(blpps,axis=2)): # number of l''
                        lpp = blpps[int(q/2),int(p/2),i] # value of l''
                        if (i > 0 and lpp == 0) or (np.abs(m-mp) > lpp): #ensuring we don't go outside of relevant l'' and m m' condition
                            continue
                        else:
                            if type == 1:
                                bessel = jl(lpp,k*d[0]) # both types of bessel functions
                            if type == 3:
                                bessel = h1l(lpp,k*d[0])
                            
                            if lpp == 0:
                                normalization = np.sqrt(1.0 / (4.0 * np.pi)) # normalizaiton for spherical harmonics l''=0
                            else:
                                normalization = np.sqrt( ( (2*lpp+1) / (4*np.pi) ) * ( factorial(lpp-(m-mp)) / factorial(lpp+(m-mp)) ) ) # and for l'' > 0  *lpp*(lpp+1)
                            
                            swf = normalization * lpmv(m-mp,lpp,np.cos(d[1])) * np.exp(1j*(m-mp)*d[2])*bessel # scalar wave functions
                            
                            sumB += b[int(q/2),int(p/2),i]*swf # summing for B

                    # prefactor to get matrices in the convention of Brian Stout(2002) fully normalized VSH
                    C[:,q,p] = np.sqrt(lp*(lp+1)/(l*(l+1)))*sumA # the first and every other row is A B A B
                    C[:,q+1,p] = np.sqrt(lp*(lp+1)/(l*(l+1)))*sumB
                    
                    C[:,q+1,p+1] = C[:,q,p] # and every other is B A B A 
                    C[:,q,p+1] =  C[:,q+1,p]

                    # Structure of C-matrix is according to eq. (2.185) in my Masters Thesis_notes
                    # Rows have lp,mp and columns have l,m
    
    return C