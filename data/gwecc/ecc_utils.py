import numpy as np
from numpy import pi, sin, cos, sqrt
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.special import hyp2f1
import scipy.constants as sc

from mikkola import get_u

GMsun = 1.327124400e20  # measured more precisely than Msun alone!
c = sc.speed_of_light
TSUN = GMsun / c**3     #[s]


def get_dedtau(tau, e):
    dedtau = (1 - e*e)**(3./2)/(e**(29./19) * (121*e*e + 304)**(1181./2299))
    return dedtau


try:
    data_tau_e = np.load("tau_e.npy")
    data_read = True
except:
    print("Warning!!! No input file.")
    data_read = False

if data_read:
    taus = data_tau_e[:,0]
    es = data_tau_e[:,1]
    
    fun_e_tau = interp1d(taus, es, bounds_error=False) #fill_value="extrapolate")
    fun_tau_e = interp1d(es, taus, bounds_error=False) #fill_value="extrapolate")
    
else:
	print("Calculating e(tau)!!!!")
	e0 = 2.5e-9
	tau0 = (e0 * 19**(145./242) /(2**(559./726) * 3**(19./48)))**(48./19)
	#print(tau0)
	#taus = np.linspace(tau0,500, 5000)
	sol = solve_ivp(get_dedtau, (tau0, 500), [e0], dense_output = False, rtol = 1e-12, atol = 1e-15)
	es = sol.y
	taus = sol.t
	es = np.reshape(sol.y,len(taus))
	tau_e = np.vstack((taus, es)).T
	np.save('tau_e.npy', tau_e)#, fmt='%1.10e')
	
	fun_e_tau = interp1d(taus,es)
	fun_tau_e = interp1d(es, taus)
	
	#plt.plot(taus,es)
	#plt.grid()
	#plt.show()
	
tau_max = taus[-1]
e_max = es[-1]
e_min = es[0]



def e_from_tau(tau):
	if np.max(tau) >= tau_max:
		#print("tau > tau_max")
		a = 2*np.sqrt(2)/(5 * 5**(63./2299) * 17**(1181./2299))
		b = 2./np.sqrt(1 - e_max) - a * tau_max
		
		return(1 - 4./(a*tau + b)**2)
	else:
		return(fun_e_tau(tau))


def tau_from_e(e):
	if e > e_max:
		a = 2*np.sqrt(2)/(5 * 5**(63./2299) * 17**(1181./2299))
		b = 2./np.sqrt(1 - e_max) - a * tau_max
		return((2./np.sqrt(1 - e) - b)/a)
	elif e < e_min:
		return((e * 19**(145./242) /(2**(559./726) * 3**(19./48)))**(48./19))
	else:
		return(fun_tau_e(e))

#print(e_from_tau([10,10,30,100]))


def n_from_e(e, n0, e0):
	e2 = e*e
	e02 = e0*e0
	n = ( n0 * (e0/e)**(18./19) * ((1 - e2)/(1 - e02))**1.5 
		 * ((304 + 121*e02)/(304 + 121*e2))**(1305./2299) )
	return(n)


def compute_alpha_coeff(A, n0, e0):
	e02 = e0*e0
	alpha = ( (3./A) * (1 - e02)**(5./2)/(n0**(5./3) * e0**(30./19) 
			 * (304 + 121*e02)**(2175./2299)) )
	return(alpha)

def compute_beta_coeff(A, m, n0, e0):
	e02 = e0*e0
	beta = ( (9./A) * (TSUN * m)**(2./3) * (1 - e02)**(3./2)/(n0 * e0**(18./19)
			* (304 + 121*e02)**(1305./2299)) )
	return(beta)
	

def compute_beta2_coeff(A, m, n0, e0):
	e02 = e0*e0
	beta2 = ( 3./(4*A) * (TSUN * m)**(4./3) * sqrt(1- e02) /(n0**(1./3) * e0**(6./19)
			* (304 + 121*e02)**(435./2299)) )
	return(beta2)


def lbar_from_e(e):
	coeff_l = (19**(2175./2299))/(30 * 2**(496./2299))
	lbar = coeff_l * e**(30./19) * hyp2f1(124./2299, 15./19, 34./19, -121.*e*e/304)
	return(lbar)


def gbar_from_e(e):
	coeff_g = (19**(1305./2299))/(36 * 2**(1677./2299))
	gbar = coeff_g * e**(18./19) * hyp2f1(994./2299,  9./19, 28./19, -121.*e*e/304)
	return(gbar)
	

def gbar2_from_e(e, eta):
	e2 = e*e
	coeff_g2 = 3 * 2**(1740./2299) * 19**(435./2299)
	gbar2 = ( e**(6./19)/336 * (4 *  (51 - 26*eta) * (304 + 121*e2)**(435./2299)
			+ coeff_g2 * (23 +  2*eta) * hyp2f1(3./19, 1864./2299, 22./19, -121*e2/304)) )
	
	return(gbar2)


def evolve_orbit(ts, mc, eta, n0, e0, l0, gamma0, tref):
    #m = (((1+q)**2)/q)**(3/5) * mc
    m = mc/eta**(3/5)
    e02 = e0*e0

    A = (TSUN * mc)**(5./3)/5
    P = (A/3.)* n0**(8./3)*e0**(48./19)*(304 + 121*e02)**(3480./2299)/(1 - e02)**4
	
    tau0 = tau_from_e(e0)
    taus = tau0 - P*(ts - tref)
    taus = taus.astype('float64')

    if np.any(taus < 0):
        if np.any(taus < 0):
            print(f"mc = {mc}, q = {q}, n0 = {n0}, e0 = {e0}, tref = {tref}.")
            print(f"tau0 = {tau0}")
            print(f"P = {P}")
            print(f"ts = {ts}")
            print("ts - tref =",ts-tref)
            print(f"taus = {taus}")
            raise ValueError("tau < 0 encountered!!")

    es = e_from_tau(taus)
    ns = n_from_e(es, n0, e0)
	
    alpha = compute_alpha_coeff(A, n0, e0)
    lbar0 = lbar_from_e(e0)
    lbars = lbar_from_e(es)
    ls = l0 + (lbar0 - lbars)*alpha
	
    beta = compute_beta_coeff(A, m, n0, e0)
    gbar0 = gbar_from_e(e0)
    gbar = gbar_from_e(es)
    gamma1 = gamma0 + (gbar0 - gbar)*beta
	
    beta2 = compute_beta2_coeff(A, m, n0, e0)
    gbar20 = gbar2_from_e(e0, eta)
    gbar2 = gbar2_from_e(es, eta)
    gamma2 = (gbar20 - gbar2)*beta2
	
    gammas = gamma1 + gamma2
    return(ns, es, ls, gammas)



def get_k(n, e, m, eta):
	xi = (TSUN * m * n)**(2./3)
	e2 = e*e
	OTS = sqrt(1 - e2)
	OTS2 = 1 - e2
	k = ( 3*xi/OTS2 + ((78 + e2*(51 - 26*eta) - 28*eta)* xi*xi)/(4.*OTS2*OTS2)
		+ ((18240 - 25376*eta + 896*eta**2 + e**4*(2496 - 1760*eta + 1040*eta**2) 
		+ (1920 + e2*(3840 - 1536*eta) - 768*eta)*OTS + 492*eta*pi**2  
		+ e2*(28128 - 27840*eta + 5120*eta**2 
		+ 123*eta*pi**2))*xi**3)/(128.*OTS**6) )
	return(k)


def get_PN_x(n, m, k):
	x = (TSUN * m * n * (1 + k))**(2./3)
	return(x)


def get_x_k_ephi(n, e, m, eta):
    k = get_k(n, e, m, eta)
    x = (TSUN * m * n * (1 + k))**(2./3)
    OTS = sqrt(1 - e*e)
	
    ep = e*(1 + x*(4 - eta) + + x*x *(4*(-12*(26 + 15*OTS) 
    + eta*(17 + 72*OTS + eta)) + e*e *(1152 + eta*(-659 + 41*eta)))/(96*(-1 + e*e)))
    return(x, k, ep)


def get_u_phi(x, k, ephi, e, l, gamma):
    u = get_u(l,e)
    L = l + gamma
	
    su = sin(u)
    cu = cos(u)
	
    if ephi[-1]>1.e-15:
        betaphi = (1 - sqrt(1 - ephi**2))/ephi
    else:
        betaphi = e/2. + e**3/8. + e**5/16
	
    v_u = 2 * np.arctan2(betaphi*su, 1 - betaphi*cu)
    v_l = v_u + e*su
	
    W = (1 + k) * v_l
    phi = L + W
        
    return(u, phi)
