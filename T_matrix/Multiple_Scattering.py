import numpy as np
from scipy.special import lpmv,spherical_jn,spherical_yn,factorial
from scipy.linalg import inv
from scipy.interpolate import CubicSpline
import scipy.io
from pathlib import Path

from .Drude import *
from .translation import *
from .sph2cart import *

class Mie():
    hc=197.46358707 #eV*nm
    def __init__(self,omega,epsout,lmax,physics = 'C'):
        self.omega = omega
        self.epsout = epsout
        self.lmax = lmax
        self.physics = physics
        #self.Nskip = Nskip
        self.a,self.b,self.alpps,self.blpps = abcoeff(lmax)
    
    def jl(l,rho,der=False): #makes it easier to call spherical bessel function
        if der:
            return spherical_jn(l,rho,True)
        else:
            return spherical_jn(l,rho)

    def h1l(l,rho,der=False): #spherical hankel function
        if der:
            return spherical_jn(l,rho,True)+1j*spherical_yn(l,rho,True)
        else:
            return spherical_jn(l,rho)+1j*spherical_yn(l,rho)
    
    def riccati(l,rho): #riccati Bessel functions
        jl = spherical_jn(l,rho)
        yl = spherical_yn(l,rho)
        djl = spherical_jn(l,rho,'true')
        dyl = spherical_yn(l,rho,'true')
        psi = rho*jl
        dpsi = jl+rho*djl
        xi = psi+1j*rho*yl
        dxi = jl+1j*yl+rho*(djl+1j*dyl)

        return psi,dpsi,xi,dxi

    def dintermat(self,rs):
            script_dir = Path(__file__).parent
            if rs == 2:
                mat_path = script_dir / 'rs=2.mat'
            if rs == 4:
                mat_path = script_dir / 'rs=4.mat'

            omegap = rs**(-1.5)*47.1552
            
            data = scipy.io.loadmat(mat_path)
            omegadat = data['omegav']
            omegadat = omegadat[0,:]*omegap
            d = data['dperp']
            d = d[0,:]
            cs = CubicSpline(omegadat, d)
            return cs(self.omega)

    def dinter(self,omegadata,data):
                cs = CubicSpline(omegadata, data)
                return cs(self.omega)
    
    def a_b(self,R,em): # Mie scattering coefficients vectorized Bohren Hoffman eq. (4.56,4.57)
        ed = self.epsout

        kd = self.omega/Mie.hc*np.sqrt(ed)
        xd = kd*R
        km = self.omega/Mie.hc*np.sqrt(em)
        xm = km*R

        lstop = self.lmax

        N_omega = np.size(xd)
        a = np.zeros((N_omega,lstop), dtype=np.complex128)
        b = np.zeros((N_omega,lstop), dtype=np.complex128)

        for i in range(lstop):
            l=i+1

            jld = spherical_jn(l,xd)
            jlm = spherical_jn(l,xm)
            h1ld = h1l(l,xd)
            _,dpsid,_,dxid= Mie.riccati(l,xd)
            _,dpsim,_,_= Mie.riccati(l,xm)

            a[:,i]=((em*jlm*dpsid
                    -ed*jld*dpsim)
                    /(em*jlm*dxid
                    -ed*h1ld*dpsim))
    
            b[:,i]=((jlm*dpsid
                    -jld*dpsim)
                    /(jlm*dxid
                    -h1ld*dpsim))
    
        return a,b
    
    def Qa_b(self,R,em,fd): # Mie scattering coefficients for surface response approximation

        ed = self.epsout

        kd = self.omega/Mie.hc*np.sqrt(ed)
        xd = kd*R
        km = self.omega/Mie.hc*np.sqrt(em)
        xm = km*R

        if fd == 'dp':
            dorth = 0.4
        if fd == 'dm':
            dorth = -0.4
        if fd == 'Ag':
            omegadata, dAgR, dAgI = np.loadtxt(fr'T_matrix\AgFeibelmanParams.txt', unpack=True)
            dAg = dAgR + 1j*dAgI
            #dorth = dAg[::self.Nskip]
            dorth = self.dinter(omegadata,dAg)
            #dorth = self.dinter(4)
        if fd == 'Na':
            omegadata, dNaR, dNaI = np.loadtxt(fr'T_matrix\NaFeibelmanParams.txt', unpack=True)
            dNa = dNaR + 1j*dNaI
            #dorth= dNa[::self.Nskip]
            dorth = self.dinter(omegadata,dNa)
            #dorth = self.dinter(2)
        dpar = 0

        lstop = self.lmax

        N_omega = np.size(xd)
        a = np.zeros((N_omega,lstop), dtype=np.complex128)
        b = np.zeros((N_omega,lstop), dtype=np.complex128)

        for i in range(lstop):
            l=i+1

            dparR = dpar/R
            dorthR = l*(l+1)*dorth/R

            jld = spherical_jn(l,xd)
            jlm = spherical_jn(l,xm)
            h1ld = h1l(l,xd)
            psid,dpsid,xid,dxid= Mie.riccati(l,xd)
            psim,dpsim,_,_= Mie.riccati(l,xm)

            a[:,i]=((em*jlm*dpsid
                    -ed*jld*dpsim
                    +(em-ed)*(jld*jlm*dorthR+dpsid*dpsim*dparR))
                    /(em*jlm*dxid
                    -ed*h1ld*dpsim
                    +(em-ed)*(h1ld*jlm*dorthR+dxid*dpsim*dparR)))
    
            b[:,i]=((jlm*dpsid
                    -jld*dpsim
                    +(xm**2-xd**2)*jld*jlm*dparR)
                    /(jlm*dxid
                    -h1ld*dpsim
                    +(xm**2-xd**2)*h1ld*jlm*dparR))
    
        return a,b
    
    def c_d(self,R,em): # Mie transmission coefficients vectorized
        k2=self.omega/Mie.hc*np.sqrt(self.epsout)
        x=k2*R
        m=np.sqrt(em)/np.sqrt(self.epsout)

        lstop = self.lmax

        N_omega = np.size(x)
        c = np.zeros((N_omega,lstop), dtype=np.complex128)
        d = np.zeros((N_omega,lstop), dtype=np.complex128)

        for i in range(lstop):
            l=i+1

            h1 = Mie.h1l(l,x)
            jl = spherical_jn(l,x)
            jlm = spherical_jn(l,m*x)
            _,dpsi,_,dxi= Mie.riccati(l,x)
            _,dpsim,_,_= Mie.riccati(l,m*x)
            
            c[:,i]=((jl*dxi
            -h1*dpsi)
            /(jlm*dxi
            -h1*dpsim))
            
            d[:,i]=((m*jl*dxi
            -m*h1*dpsi)
            /(m**2*jlm*dxi
            -h1*dpsim))
    
        return c,d

    def Qc_d(self,R,em,fd): # Mie scattering coefficients for surface response approximation

        ed = self.epsout

        kd = self.omega/Mie.hc*np.sqrt(ed)
        xd = kd*R
        km = self.omega/Mie.hc*np.sqrt(em)
        xm = km*R

        if fd == 'dp':
            dorth = 0.4
        if fd == 'dm':
            dorth = -0.4
        if fd == 'Ag':
            omegadata, dAgR, dAgI = np.loadtxt(fr'T_matrix\AgFeibelmanParams.txt', unpack=True)
            dAg = dAgR + 1j*dAgI
            #dorth = dAg[::self.Nskip]
            dorth = self.dinter(omegadata,dAg)
            #dorth = self.dinter(4)
        if fd == 'Na':
            omegadata, dNaR, dNaI = np.loadtxt(fr'T_matrix\NaFeibelmanParams.txt', unpack=True)
            dNa = dNaR + 1j*dNaI
            #dorth= dNa[::self.Nskip]
            dorth = self.dinter(omegadata,dNa)
            #dorth = self.dinter(2)
        dpar = 0

        lstop = self.lmax

        N_omega = np.size(xd)
        c = np.zeros((N_omega,lstop), dtype=np.complex128)
        d = np.zeros((N_omega,lstop), dtype=np.complex128)

        for i in range(lstop):
            l=i+1

            dparR = dpar/R
            dorthR = l*(l+1)*dorth/R

            jld = spherical_jn(l,xd)
            jlm = spherical_jn(l,xm)
            h1ld = h1l(l,xd)
            psid,dpsid,xid,dxid= Mie.riccati(l,xd)
            psim,dpsim,_,_= Mie.riccati(l,xm)

            c[:,i]=((jld*dxid
                    -h1ld*dpsid)
                    /(jlm*dxid
                    -h1ld*dpsim+(xm**2-xd**2)*h1ld*jlm*dparR))
    
            d[:,i]=np.sqrt(em*ed)*((jld*dxid
                    -h1ld*dpsid)
                    /(em*jlm*dxid
                    -ed*h1ld*dpsim+(em-ed)*( h1ld*jlm*dorthR+dxid*dpsim*dparR )))
    
        return c,d

    def single_sphere(self,R,epsin,fd): # T matrix for a single sphere

        if self.physics == 'C':
            a,b=self.a_b(R,epsin) # mie coefficients dependent on radius R
        if self.physics == 'Q':
            a,b=self.Qa_b(R,epsin,fd)
        # The single sphere T-matrix is a diagonal matrix made on of the Mie coefficients with a negative sign
        lstop = np.size(a,1)
        Nomega = np.size(a,0)

        lmodes = 2*(lstop**2+2*lstop) # total number of modes with polarization
        
        T = np.zeros((Nomega,lmodes,lmodes), dtype=np.complex128) 
        # data structure of T-matrix should be the same as Treams as per "T-matrix represantion of optical scattering response: Suggestion for a data format"
        # Creator of Treams Dominik Beutel is also cited in this paper
        # lmodes has the structure of l -> m -> polarization 
        idx = 0
        for l in range(1,lstop+1):
            for m in range(-l,l+1):
                T[:,idx,idx]=-b[:,l-1] # TE modes. First index is frequency and l-1 is due to l starting at 1
                T[:,idx+1,idx+1]=-a[:,l-1] # TM modes
                idx+=2 # accounts for polarization
        return T
    
    def transmission_matrix(self,R,epsin,fd): # T matrix for transmission of a single sphere
        # same as for scattering except now it is c and d coefficients and no negative sign since it doesnt reverse direction as scattering
        
        if self.physics == 'C':
            c,d = self.c_d(R,epsin) # mie coefficients dependent on radius R
        if self.physics == 'Q':
            c,d = self.Qc_d(R,epsin,fd)
        lstop = np.size(c,1)
        Nomega = np.size(c,0)

        lmodes = 2*(lstop**2+2*lstop) # total number of modes with polarization
        
        T = np.zeros((Nomega,lmodes,lmodes), dtype=np.complex128)
        idx = 0
        for l in range(1,lstop+1):
            for m in range(-l,l+1):
                T[:,idx,idx]=c[:,l-1]
                T[:,idx+1,idx+1]=d[:,l-1]
                idx+=2 # accounts for polarization
        return T
        
    def multi_sphere(self,Rs,pos,epsin,fd): # multiple sphere T-matrix, following the derivation of Brian Stout(2002)
        # multi sphere T-matrix is a concatenated matrix of T-matrices T(i,j) each holding the contribution of sphere j to the scattered field of sphere i
        ks = self.omega/self.hc*np.sqrt(self.epsout) # k-vectors. Using epsout since we are looking at scattered fields
        Nomega = np.size(self.omega) # number of omegas
        N = np.size(Rs) # number of scatterers
        
        lstop = self.lmax
        lmodes = 2*(lstop**2+2*lstop) # number of modes with polarization
        
        Tsize = N*lmodes # Total size of Tmatrix
        
        T = np.zeros((Nomega,Tsize,Tsize),dtype = np.complex128) # structure of final 
        Tdiag = np.zeros((Nomega,Tsize,Tsize),dtype = np.complex128) # T-matrix with single scatterer T-matrices on diagonal
        for i in range(N):
            for j in range(N):
                row_start, row_end = i * lmodes, (i + 1) * lmodes # Defines the (i,j) part of the Tmatrix
                col_start, col_end = j * lmodes, (j + 1) * lmodes
                if Nomega == 1:
                    T1 = self.single_sphere(Rs[j],epsin[j],fd[j]) # each column holds T1(j)
                else:
                    T1 = self.single_sphere(Rs[j],epsin[j,:],fd[j]) # each column holds T1(j)
                if i == j:
                    Tdiag[:,row_start:row_end,col_start:col_end] = T1 # to create diagonal T-matrix
                    for w in range(Nomega):
                        T[w,row_start:row_end,col_start:col_end] = np.identity(lmodes) # diagonals are just identity matrices
                else:
                    rij_xyz = np.array([pos[i,0]-pos[j,0],pos[i,1]-pos[j,1],pos[i,2]-pos[j,2]]) # position vector from scatterer j to i
                    rij = cart2sph(rij_xyz[0],rij_xyz[1],rij_xyz[2]) # transforms to spherical coordinates
                    Cij3 = Cmatrix_chew(lstop,rij,ks,3,self.a,self.b,self.alpps,self.blpps) # translation of scattered fields is done with hankel functions therefore the "3"
                    #Cij3 = Cmatrix(lmax,rij,ks,3) # Translation matrix from Brian Stout
                    for w in range(Nomega):
                        T[w,row_start:row_end,col_start:col_end] = - Cij3[w,:,:] @ T1[w,:,:] # off diagonal elemts of tmatrix to be inverted see Brian Stout(2002) eq. (10)

        # Diagonal similarity scaling S^-1 (I - CT) S with s_l = |h_l(k R_i)|, so all entries are O(1)
        ks_arr = np.atleast_1d(ks)
        ls = np.concatenate([[l]*2*(2*l+1) for l in range(1, lstop+1)])  # l of each mode (l -> m -> pol)
        for w in range(Nomega):
            s = np.concatenate([np.abs(spherical_jn(ls, ks_arr[w]*Rj) + 1j*spherical_yn(ls, ks_arr[w]*Rj))
                                for Rj in np.atleast_1d(Rs)])
            Mt = T[w,:,:]*s[None,:]/s[:,None]            # scaled system matrix
            T[w,:,:] = s[:,None]*inv(Mt)/s[None,:]       # = (I - CT)^-1, as before
            # this gives the T-matrix as in Brian Stout (2002) or in my Masters Thesis_notes eq. (2.173)
        return T,Tdiag # returns diagonal T-matrix for future use

        
        #for w in range(Nomega):
        #    T[w,:,:] = inv(T[w,:,:]) # inversion of Tmatrix
        #    # this gives the T-matrix as in Brian Stout (2002) or in my Masters Thesis_notes eq. (2.173)
        #return T,Tdiag # returns diagonal T-matrix for future use. a,b,alpps,blpps also for further use to avoid having to calculate twice

    def scattering_coeffs(self,Rs,pos,a_inc,T,Tdiag):
        # For calculating scattering coefficients
        # Takes T and Tdiag as inputs. For a single sphere T would just be the single sphere T-matrix
        N = np.size(Rs) # number of scatterers
        Tsize = np.size(a_inc,1) # total size of T-matrix. Even for Nomega = 1, a_inc should have shape (1,Tsize)
        Nomega = np.size(self.omega)

        lstop = int(np.sqrt(1+0.5*Tsize/N)-1) # lmax can be found by solving for it in the equation below
        lmodes = 2*(lstop**2+2*lstop)

        ks = self.omega/self.hc*np.sqrt(self.epsout) # k-vectors using epsout since we are looking at scattered fields

        a_inc_i = np.zeros_like(a_inc,dtype=np.complex128) # array for translated incoming fields
        T_scat = np.zeros_like(T,dtype=np.complex128) # T-matrix for scattered fields instead of just excitation fields

        for w in range(Nomega):
            T_scat[w,:,:] = Tdiag[w,:,:] @ T[w,:,:]  # We find the scattering coefficients from the excitation field coefficients by multpiplying by Tdiag

        for j in range(N): # for loop for translating incident field as is Stout(2002) eq. (12,13) or Beutel(2023) eq. (19,20)
                start, end = j * lmodes, (j + 1) * lmodes
                if N == 1:
                    rj0_xyz = np.array([pos[0],pos[1],pos[2]]) # coordinates for single sphere
                    rj0 = cart2sph(rj0_xyz[0],rj0_xyz[1],rj0_xyz[2])
                else:
                    rj0_xyz = np.array([pos[j,0],pos[j,1],pos[j,2]]) # coordinates of ith sphere
                    rj0 = cart2sph(rj0_xyz[0],rj0_xyz[1],rj0_xyz[2]) # converting to spherical coordinates
                Cj01s = Cmatrix_chew(lstop,rj0,ks,1,self.a,self.b,self.alpps,self.blpps) # translation coefficients with Bessel functions
                #Cj01s = Cmatrix(lmax,rj0,ks,1)

                for w in range(Nomega):
                    a_inc_i[w,start:end] = Cj01s[w,:,:] @ a_inc[w,start:end] # creating translated incicent field coefficients
        
        p = np.zeros((Nomega,Tsize),dtype = np.complex128) # scattering coefficients
        for w in range(Nomega):
            p[w,:] = T_scat[w,:,:] @ a_inc_i[w,:] # found by multiplying total T-matrix with translated incident field
        return p

    def transmission_coeffs(self,Rs,pos,epsin,fd,a_inc,T):
        N = np.size(Rs)

        Tsize = np.size(a_inc,1)
        Nomega = np.size(self.omega)

        lmax = int(np.sqrt(1+0.5*Tsize/N)-1)
        lmodes = 2*(lmax**2+2*lmax)

        ks = self.omega/self.hc*np.sqrt(self.epsout)

        a_inc_i = np.zeros_like(a_inc,dtype=np.complex128)
        T_trans = np.zeros_like(T,dtype=np.complex128)

        Tdiag = np.zeros((Nomega,Tsize,Tsize),dtype = np.complex128) 

        for i in range(N):
                start, end = i * lmodes, (i + 1) * lmodes

                if Nomega == 1:
                    Tdiag[:,start:end,start:end] = self.transmission_matrix(Rs[i],epsin[i],fd[i])
                else:
                    Tdiag[:,start:end,start:end] = self.transmission_matrix(Rs[i],epsin[i,:],fd[i])
            
                if N == 1:
                    rj0_xyz = np.array([pos[0],pos[1],pos[2]])
                else:
                    rj0_xyz = np.array([pos[i,0],pos[i,1],pos[i,2]])
                rj0 = cart2sph(rj0_xyz[0],rj0_xyz[1],rj0_xyz[2])
                Cj01s = Cmatrix_chew(lmax,rj0,ks,1,self.a,self.b,self.alpps,self.blpps)
                #Cj01s = Cmatrix(lmax,rj0,ks,1)

                for w in range(Nomega):
                    a_inc_i[w,start:end] = Cj01s[w,:,:] @ a_inc[w,start:end]
        
        for w in range(Nomega):
            T_trans[w,:,:] = Tdiag[w,:,:] @ T[w,:,:]

        p = np.zeros((Nomega,Tsize),dtype = np.complex128)
        for w in range(Nomega):
            p[w,:] = T_trans[w,:,:] @ a_inc_i[w,:]
        return p

    def C_ext(self,a_inc,Rs,pos,epsin,fd):
        ks = self.omega/self.hc*np.sqrt(self.epsout) # k-vectors
        Nomega = np.size(self.omega) #
        N = np.size(Rs) # number of scatterers
        lstop = self.lmax
        
        lmodes = 2*(lstop**2+2*lstop) # number of modes with polarization

        C = np.zeros(Nomega)

        if N == 1:
            T = self.single_sphere(Rs,epsin,fd)
            C = -2*np.pi/(ks**2)*np.trace(np.real(T),axis1=1,axis2=2)
        else:
            T,Tdiag = self.multi_sphere(Rs,pos,epsin,fd)
            
            T_scat = np.zeros_like(T,dtype=np.complex128)
            for w in range(Nomega):
                T_scat[w,:,:] = Tdiag[w,:,:] @ T[w,:,:]
        
            a_inc_i = a_inc[:,0:lmodes]

            Cj01s = np.zeros((N,Nomega,lmodes,lmodes),dtype=np.complex128)
            C0i1s = np.zeros((N,Nomega,lmodes,lmodes),dtype=np.complex128)
            for j in range(N):
                rj0_xyz = np.array([pos[j,0],pos[j,1],pos[j,2]])
                rj0 = cart2sph(rj0_xyz[0],rj0_xyz[1],rj0_xyz[2])
                Cj01s[j,:,:,:] = Cmatrix_chew(lstop,rj0,ks,1,self.a,self.b,self.alpps,self.blpps) 

                r0i_xyz = -np.array([pos[j,0],pos[j,1],pos[j,2]])
                r0i = cart2sph(r0i_xyz[0],r0i_xyz[1],r0i_xyz[2])
                C0i1s[j,:,:,:] = Cmatrix_chew(lstop,r0i,ks,1,self.a,self.b,self.alpps,self.blpps)

            for w in range(Nomega):
                for i in range(N):
                    Ci = 0
                    row_start, row_end = i * lmodes, (i + 1) * lmodes
                    for j in range(N):
                        col_start, col_end = j * lmodes, (j + 1) * lmodes
                        Tij = C0i1s[i,w,:,:] @ T_scat[w,row_start:row_end,col_start:col_end] @ Cj01s[j,w,:,:]
                        Ci += np.real(np.dot(np.conjugate(a_inc_i[w,:]) , Tij @ a_inc_i[w,:]))

                    C[w] += -1/(ks[w]**2)*Ci
        return C
    
    def C_scat(self,a_inc,Rs,pos,epsin,fd):
        ks = self.omega/self.hc*np.sqrt(self.epsout) # k-vectors
        Nomega = np.size(self.omega) #
        N = np.size(Rs) # number of scatterers
        lstop =self.lmax

        lmodes = 2*(lstop**2+2*lstop) # number of modes with polarization

        T,Tdiag = self.multi_sphere(Rs,pos,epsin,fd)
        p = self.scattering_coeffs(Rs,pos,a_inc,T,Tdiag)
        C = np.zeros(Nomega)

        if N == 1:
            for w in range(Nomega):
                C[w] = 1/(ks**2)*np.real(np.dot(np.conjugate(p[w,:]),p[w,:]))
        else:
            Cij1s = np.zeros((N,N,Nomega,lmodes,lmodes),dtype=np.complex128)
            for i in range(N):
                for j in range(N):
                    rij_xyz = np.array([pos[i,0]-pos[j,0],pos[i,1]-pos[j,1],pos[i,2]-pos[j,2]]) # position vector from scatterer j to i
                    rij = cart2sph(rij_xyz[0],rij_xyz[1],rij_xyz[2])
                    Cij1s[i,j,:,:,:] = Cmatrix_chew(lstop,rij,ks,1,self.a,self.b,self.alpps,self.blpps)

            for w in range(Nomega):
                for i in range(N):
                    i_start, i_end = i * lmodes, (i + 1) * lmodes
                    Ci = 0
                    for j in range(N):
                        j_start, j_end = j * lmodes, (j + 1) * lmodes
                        Ci += np.dot(np.conjugate(p[w,i_start:i_end]),Cij1s[i,j,w,:,:] @ p[w,j_start:j_end])
                    C[w] += 1/(ks[w]**2)*np.real(Ci)
        return C
    
    def plane_wave(self,E0,N):
        # plane wave expansion coefficients for fully normalized VSH
        lstop = self.lmax
        lmodes = 2*(lstop**2+2*lstop)
        Nomega = np.size(self.omega)
        a = np.zeros((Nomega,N*lmodes),dtype = np.complex128) # concatanated array of coefficients
        
        for i in range(N):
            for l in range(1,lstop+1):
                for m in range(-l,l+1):
                    # entire normalization = np.sqrt( ( (2*l+1) / (4*np.pi * l * (l+1)) ) * (factorial(l-m) / factorial(l+m)) )
                    if m == 1:
                        normalization = np.sqrt( ( (2*l+1) / (4*np.pi * l * (l+1)) ) * ( 1 / (l*(l+1)) ) ) # for m = 1 we get this normalization
                        ElTE = 1j*normalization*2*np.pi*1j**l*E0*l*(l+1) # expansion coefficients TE modes
                        ElTM = 1j*normalization*2*np.pi*1j**l*E0*l*(l+1) # expansion coefficients TM modes
                        # These are the same but we get a 90* phase difference from the difference in m = -1 coefficients and e^(im\phi)
                    if m == -1:
                        normalization = np.sqrt( ( (2*l+1) / (4*np.pi * l * (l+1)) ) * ( l*(l+1) ) )
                        ElTE = 1j*normalization*2*np.pi*1j**l*E0 # l*(l+1) dissapears due to associated legendre function parity
                        ElTM = -1j*normalization*2*np.pi*1j**l*E0 # difference in signs

                    p = 2*(indx(l,m)-1) # index accounts for polarization and -1 due to l starting at 1
                    # indx function is within translation.py
                    if np.abs(m) != 1: # only m = -1,1 modes are nonzero
                        a[:,p+lmodes*i] = 0
                        a[:,p+lmodes*i+1] = 0
                    else:
                        a[:,p+lmodes*i] = ElTE
                        a[:,p+lmodes*i+1] = ElTM
        return a

    def M_N(self,m,l,rho,theta,phi,bessel):

        #Bessel functions
        zl = bessel(l,rho)
        zlmin1 = bessel(l-1,rho) # for future recurrence
        zlplus1 = bessel(l+1,rho)
        dricc = rho*zlmin1-l*bessel(l,rho)#Recurrence relation DLMF 10.51.2 makes it more numerically stable

        #Legendre function
        x = np.cos(theta)
        Plm = lpmv(m,l,x)

        Plm_sin = np.zeros_like(theta)
        dPlm_dtheta = np.zeros_like(theta)

        if (l+m) == 0:
            Plm_1 = np.zeros_like(theta)
        else:
            Plm_1 = lpmv(m,l-1,x)

        mask0 = theta < 1e-9
        maskpi = theta > np.pi-1e-9
        general_case = ~(mask0 | maskpi)

        if m == 1:
            Plm_sin[mask0] = l*(l+1)/2
            Plm_sin[maskpi] = (-1)**(l+1)*l*(l+1)/2

            dPlm_dtheta[mask0] = l*(l+1)/2
            dPlm_dtheta[maskpi] = (-1)**(l)*l*(l+1)/2
        elif m == -1:
            Plm_sin[mask0] = -1/2
            Plm_sin[maskpi] = (-1)**(l)/2

            dPlm_dtheta[mask0] = -1/2
            dPlm_dtheta[maskpi] = (-1)**(l+1)/2

        if np.any(general_case):
            Plm_sin[general_case] = Plm[general_case] / np.sin(theta[general_case])

            dPlm = ( l*x[general_case]*lpmv(m,l,x[general_case]) - (l+m) * Plm_1[general_case]) / (x[general_case]**2-1)
            dPlm_dtheta[general_case] = -np.sin(theta[general_case]) * dPlm

        M = np.array([
            np.zeros_like(theta),
            1j * m * zl * Plm_sin * np.exp(1j*m*phi),
            -zl * dPlm_dtheta * np.exp(1j*m*phi)
        ])
        
        mask = rho<1e-6
        dricc_rho = dricc/rho
        if l == 1:
            dricc_rho[mask] = 2.0/3.0
        else:
            dricc_rho[mask] = 0

        N = np.array([
            ((l*(l+1))/(2*l+1)) * (zlmin1+zlplus1) * Plm * np.exp(1j*m*phi),
            dricc_rho * dPlm_dtheta * np.exp(1j*m*phi),
            1j * m * dricc_rho * Plm_sin * np.exp(1j*m*phi)
        ])

        normalization = np.sqrt( ( (2*l+1) / (4*np.pi * l * (l+1)) ) * (factorial(l-m) / factorial(l+m)) )
        M = normalization * M
        N = normalization * N

        return M,N
    

    def Efields(self,x,y,z,a_inc,Rs,pos,epsin,fd):
        N = np.size(Rs)

        T,Tdiag = self.multi_sphere(Rs,pos,epsin,fd) # inputs for a_scat and a_in

        a_scat = self.scattering_coeffs(Rs,pos,a_inc,T,Tdiag) # Scattering coefficients
        a_in = self.transmission_coeffs(Rs,pos,epsin,fd,a_inc,T) # Field coefficients inside the spheres
        kout = self.omega/Mie.hc*np.sqrt(self.epsout) # k-vectors outside spheres
        kins = self.omega/Mie.hc*np.sqrt(epsin) # k-vectors inside

        r,theta,phi = cart2sph(x,y,z) # mesh in spherical coordinates
        rhoout = kout*r # rho outside spheres for incident field

        Ematrix = np.zeros_like(r,dtype=np.complex128)
        Ematrix = np.array([Ematrix,Ematrix,Ematrix]) # Efield matrices has three elements for three different directions

        E_inc = np.zeros_like(Ematrix,dtype=np.complex128)
        E_scat_xyz = np.zeros_like(Ematrix,dtype=np.complex128)
        E_in_xyz = np.zeros_like(Ematrix,dtype=np.complex128)
        
        lmodes = 2*(self.lmax**2+2*self.lmax) # number of modes
        
        for l in range(1,self.lmax+1): # first incident field is created for global origin
                for m in range(-l,l+1):    
                    p = 2*(indx(l,m)-1)

                    Minc,Ninc = self.M_N(m,l,rhoout,theta,phi,Mie.jl) # vector spherical wave functions for incident field
                    E_inc += a_inc[0,p]*Minc+a_inc[0,p+1]*Ninc # this is only done once since we done need one for each scatterer
                    
        E_inc_xyz = E_sph2cart(E_inc,theta,phi) # incident field is converted to cartesian coordinates
        for i in range(N):
            kin = kins[i]
            if N == 1:
                # Mesh is translated to have its origin at the center of sphere i
                xi = x - pos[0]
                yi = y - pos[1]
                zi = z - pos[2]
            else:
                # Mesh is translated to have its origin at the center of sphere i
                xi = x - pos[i,0]
                yi = y - pos[i,1]
                zi = z - pos[i,2]
            ri,thetai,phii = cart2sph(xi,yi,zi) # translated coordinates are converted to spherical coordinates
                
            rhoouti = kout*ri # new rhoout is found
            rhoini = kin*ri # and a new rhoin which is rho inside the spheres assuming they have the same materrial

            E_scati = np.zeros_like(Ematrix,dtype=np.complex128) # matrices for the electric fields of scatterer i
            E_ini = np.zeros_like(Ematrix,dtype=np.complex128)
            for l in range(1,self.lmax+1):
                for m in range(-l,l+1):    
                    p = 2*(indx(l,m)-1) # index to account for polarization, -1 due to l starting at 1

                    Ms,Ns = self.M_N(m,l,rhoouti,thetai,phii,Mie.h1l) # VSWF for scattered field with hankel functions
                    Min,Nin = self.M_N(m,l,rhoini,thetai,phii,Mie.jl) # VSWF for transmitted field with Bessel functions

                    E_scati += a_scat[0,p+lmodes*i]*Ms+a_scat[0,p+lmodes*i+1]*Ns # summing up contributions from all l and m. First index assumes only one omega
                    E_ini += a_in[0,p+lmodes*i]*Min+a_in[0,p+lmodes*i+1]*Nin

            E_scati_xyz = E_sph2cart(E_scati,thetai,phii) # converting to cartesian coordinates before adding fields since spherical unit vectors may differ
            E_ini_xyz = E_sph2cart(E_ini,thetai,phii)

            if N == 1: # field inside spheres set to zero outside spheres before adding them together
                E_ini_xyz[:,np.sqrt((x-pos[0])**2+(y-pos[1])**2+(z-pos[2])**2)>Rs] = 0
            else:
                E_ini_xyz[:,np.sqrt((x-pos[i,0])**2+(y-pos[i,1])**2+(z-pos[i,2])**2)>Rs[i]] = 0

            E_scat_xyz += E_scati_xyz # adding together scattered field for sphere i to total field
            E_in_xyz += E_ini_xyz # same with transmitted fields. These should not interact as they only exist inside each sphere
        
        E_out_xyz = E_scat_xyz + E_inc_xyz # total field outside the spheres

        if N == 1: # field outside the spheres is set to zero inside spheres
            E_out_xyz[:,np.sqrt((x-pos[0])**2+(y-pos[1])**2+(z-pos[2])**2)<Rs] = 0
        else:
            for i in range(N):
                E_out_xyz[:,np.sqrt((x-pos[i,0])**2+(y-pos[i,1])**2+(z-pos[i,2])**2)<Rs[i]] = 0
        
        E_xyz = E_out_xyz + E_in_xyz #fields inside and outside spheres are added together.
            
        return E_xyz