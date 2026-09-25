import numpy as np
from numpy import sin, cos, tan, arctan, sqrt, pi
import matplotlib.pyplot as plt
from orbital_elements_gwecc import (rE,rtE, phitE, x_x1, get_k, get_l, get_beta, 
get_w, u_from_l_3PN, get_u_,hms_to_rad, dms_to_rad)
from constants import *
from antenna_pattern import antenna_pattern 
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.interpolate import CubicSpline
from ecc_utils import evolve_orbit

def orbit_evolve(tarr,t0,z0,M,eta,order,get_ngel=None):
    n0,e0,gamma0,l0=z0
    Mc=M*eta**(3/5)
    n,e,l,g=evolve_orbit(tarr, Mc, eta, n0, e0, l0, gamma0, t0)
    x1=(tsun * M * n)**(2./3)
    x=x_x1(x1,e,eta,order)
    u=np.array([u_from_l_3PN(l[ii],e[ii],x[ii],eta,order) for ii in range(len(l))])
    phi=l+g+get_w(x,e,eta,u,order)
    if get_ngel==None:
        return x, e, u,phi
    elif get_ngel==True:
        return n,e,g,l
        
def del_p(gwphi, cos_gwtheta, phi, theta,pdist):
    gwra = gwphi
    gwdec = np.arcsin(cos_gwtheta)

    psrra = phi
    psrdec = np.pi/2 - theta
    
    cosmu, Fp, Fx = antenna_pattern(gwra, gwdec, psrra, psrdec)
    
    return pdist/c*(1-cosmu)


def get_hp_hx(zarr,M,eta,inc,dist,order):
    x, e, u,phi=zarr
    r1=rE(x,e,eta,u,order)
    rphit=r1*phitE(x,e,eta,u,order)
    rt=rtE(x,e,eta,u,order)
    z=1/r1
    hp_arr=(-eta*(sin(inc)**2*(z-rphit**2-rt**2)+(1+cos(inc)**2)*((z+rphit**2-rt**2)*cos(2*phi)+2*rt*rphit*sin(2*phi))))
    hx_arr=(-2*eta*cos(inc)*((z+rphit**2-rt**2)*sin(2*phi)-2*rt*rphit*cos(2*phi)))
    dis=M*dsun
    sc=dist*1e9*pc/dis
    return hp_arr/sc, hx_arr/sc

def cal_sp_sx_n(tz_arr,t0,Amp,M,z,eta,i,order):
    x10 = (tsun * M * z[0])**(2./3)
    x=x_x1(x10,z[1],eta,order)
    z2=orbit_evolve(tz_arr,t0,z,M,eta,order)
    D_GW=M*dsun*x10*eta/(Amp*z[0]*1e9*pc)
    h=get_hp_hx(z2,M,eta,i,D_GW,order)
    s_arr = [cumtrapz(h[i], x = tz_arr, initial=0) for i in range(len(h))]
    return s_arr

def cal_sp_sx_A(t,t0,Amp,M,z,eta,i,order):
    n,et,gamma_0,l_0=z

    x10 = (tsun * M * n)**(2./3)
    x0=x_x1(x10,et,eta,order)
    x, e, u,phi=orbit_evolve(t,t0,z,M,eta,order)
    
    bb=get_beta(x0,et,eta,order)
    vmu=2*arctan(bb*sin(u)/(1-bb*cos(u)))
    v=u+vmu
    
    omg=phi-v
    
    et2=et*et
    w=1-et*cos(u)
    P=sqrt(1-et2)*(cos(2*u)-et*cos(u))/w
    Q=((et2-2)*cos(u)+et)*sin(u)/w
    R=et*sin(u)
    
    
    spA=((cos(i)**2+1)*(-P*sin(2*omg)+Q*cos(2*omg))+sin(i)**2*R)
    sxA=2*cos(i)*(P*cos(2*omg)+Q*sin(2*omg))
    
    return Amp*spA, Amp*sxA

#from enterprise.signals.signal_base import function as enterprise_function, PTA

def add_ecc_cgw(toas,
    theta,
    phi,
    cos_gwtheta,
    gwphi,
    psi,
    cos_inc,
    log10_n,
    q,
    D_L,# in Mpc
    e0,
    log10_Mc,
    tref,
    gamma_0,
    l_0,
    pdist,
    res='Both',
    interp_steps=1000
):
    order = 3
    n0 = 10**log10_n # mean motion
    Mc = 10**log10_Mc
    
    eta=q/(1+q)**2
    M=Mc/eta**(3/5)
    ts = toas - tref
    
    z0=[n0,e0,gamma_0,l_0]
    x10 = (tsun * M * n0)**(2./3)
    
    Amp=M*dsun*x10*eta/(D_L*n0*1e6*pc)
    # ti, tf, tzs in seconds, in source frame
    ti = min(ts)
    tf = max(ts)
    Tspan=tf-ti
    
    tz_arr = np.linspace(ti, tf, interp_steps)
    delta_t_arr = (tz_arr[1]-tz_arr[0])
    
    inc = np.arccos(cos_inc)

    gwra = gwphi
    gwdec = np.arcsin(cos_gwtheta)

    psrra = phi
    psrdec = np.pi/2 - theta


    cosmu, Fp, Fx = antenna_pattern(gwra, gwdec, psrra, psrdec)
    
    tP_arr=tz_arr-pdist/c*(1-cosmu)
    
    
    
    if res=='Both':
        spE,sxE=cal_sp_sx_n(tz_arr,0,Amp,M,z0,eta,inc,order)
        spP,sxP=cal_sp_sx_n(tP_arr,0,Amp,M,z0,eta,inc,order)
        sp=spE-spP
        sx=sxE-sxP
        
    if res=='Earth':
        sp,sx=cal_sp_sx_n(tz_arr,0,Amp,M,z0,eta,inc,order)

    if res=='Pulsar':
        spP,sxP=cal_sp_sx_n(tP_arr,0,Amp,M,z0,eta,inc,order)
        sp=-spP
        sx=-sxP

    c2psi = np.cos(2*psi)
    s2psi = np.sin(2*psi)
    Rpsi = np.array([[c2psi, -s2psi],
                     [s2psi, c2psi]])
    s_arr = np.dot([Fp,Fx], np.dot(Rpsi, [sp,sx]))

    s_spline = CubicSpline(tz_arr, s_arr)

    s = s_spline(ts)

    return s



