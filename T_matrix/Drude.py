
def Drude(omega,epsinf,omegap,gamma):
    return epsinf-omegap**2/(omega**2+1j*gamma*omega)