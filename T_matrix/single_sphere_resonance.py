import numpy as np
from scipy.special import lpmv,spherical_jn,spherical_yn,factorial
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline
import scipy.io
from pathlib import Path

from .Drude import *

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

def riccati(l,rho):
    jl = spherical_jn(l,rho)
    yl = spherical_yn(l,rho)
    djl = spherical_jn(l,rho,'true')
    dyl = spherical_yn(l,rho,'true')
    psi = rho*jl
    dpsi = jl+rho*djl
    xi = psi+1j*rho*yl
    dxi = jl+1j*yl+rho*(djl+1j*dyl)

    return psi,dpsi,xi,dxi

def resonance_func(omega_vec,l,a,epsinf,omega_p,gamma,epsout):
    hc=197.46358707 #eV*nm

    omega = omega_vec[0] + 1j * omega_vec[1]

    epsin=Drude(omega,epsinf,omega_p,gamma)
    k2=omega/hc*np.sqrt(epsout)
    x=k2*a
    m=np.sqrt(epsin)/np.sqrt(epsout)

    _,_,_,dxi= riccati(l,x)
    _,dpsim,_,_= riccati(l,m*x)

    result = (dxi/h1l(l,x)-dpsim/(m**2*spherical_jn(l,m*x)))

    h1 = h1l(l,x)
    jm = spherical_jn(l,m*x)
    psi,dpsi,xi,dxi= riccati(l,x)
    psim,dpsim,_,_= riccati(l,m*x)

    #result = (m**2*jm*dxi-h1*dpsim)
    return [result.real, result.imag]

def resonance(l,a,epsinf,omega_p,gamma,epsout):
    guess = [omega_p/3,-5*gamma]
    sol = fsolve( lambda omega: resonance_func(omega,l,a,epsinf,omega_p,gamma,epsout),guess)
    return sol[0]+1j*sol[1]

def dinter(omega):
        script_dir = Path(__file__).parent
        mat_path = script_dir / 'rs=4.mat'

        rs = 4
        omegap = rs**(-1.5)*47.1552
        
        data = scipy.io.loadmat(mat_path)
        omegadat = data['omegav']
        omegadat = omegadat[0,:]*omegap
        d = data['dperp']
        d = d[0,:]
        cs = CubicSpline(omegadat, d)
        return cs(omega)

def resonance_funcQ(omega_vec,l,a,epsinf,omega_p,gamma,epsout):
    hc=197.46358707 #eV*nm

    omega = omega_vec[0] + 1j * omega_vec[1]

    em = Drude(omega,epsinf,omega_p,gamma)

    ed = epsout

    kd = omega/hc*np.sqrt(ed)
    xd = kd*a
    km = omega/hc*np.sqrt(em)
    xm = km*a

    dorth = dinter(omega)

    dorthR = l*(l+1)*dorth/a

    jlm = spherical_jn(l,xm)
    h1ld = h1l(l,xd)
    _,_,_,dxid= riccati(l,xd)
    _,dpsim,_,_= riccati(l,xm)

    result = (dxid/h1ld
            -ed*dpsim/(em*jlm)
            +(em-ed)/em*dorthR)
    return [result.real, result.imag]

def resonanceQ(l,a,epsinf,omega_p,gamma,epsout):
    guess = [omega_p/3,-5*gamma]
    sol = fsolve( lambda omega: resonance_funcQ(omega,l,a,epsinf,omega_p,gamma,epsout),guess)
    return sol[0]+1j*sol[1]