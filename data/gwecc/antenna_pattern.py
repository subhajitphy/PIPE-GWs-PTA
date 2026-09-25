import numpy as np
from numpy import pi, sin, cos, sqrt, arctan2


# Based on Taylor et al. 2016, ApJ 817 70
# This version is already implemented in enterprise extensions
# There may be some issue with this version of the antenna pattern
# We are not using this verison of the antenna pattern for now
"""
def antenna_pattern(delta_gw, alpha_gw, delta_p, alpha_p):

    
    Arguments:
        delta_gw    is DEC of GW source
        alpha_gw    is RA  of GW source
        delta_p     is DEC of pulsar
        alpha_p     is RA  of pulsar

    Returns:
        coseta      Dot product of direction to GW source and direction to pulsar
        Fp          Antenna pattern F+
        Fx          Antenna pattern Fx 
    

    #theta_gw = pi/2-delta
    #phi_gw   = alpha

    s_th_gw = cos(delta_gw)
    c_th_gw = sin(delta_gw)
    s_phi_gw = sin(alpha_gw)
    c_phi_gw = cos(alpha_gw)
    
    s_th_p  = cos(delta_p)
    c_th_p  = sin(delta_p)
    s_phi_p  = sin(alpha_p)
    c_phi_p  = cos(alpha_p)

    # Equations 4-6 of https://arxiv.org/pdf/1204.4218.pdf
    omg_hat = [-s_th_gw*c_phi_gw, -s_th_gw*s_phi_gw, -c_th_gw]      # Propagation direction of GW
    mhat = [-     s_phi_gw,       c_phi_gw,  0]         			# Direction of increasing RA of GW source
    nhat = [-c_th_gw*c_phi_gw, -c_th_gw*s_phi_gw,  s_th_gw]      	# Direction of increasing DEC of GW source

    phat = [ s_th_p *c_phi_p,   s_th_p *s_phi_p,   c_th_p]       	# Direction to pulsar
    
    #print(omg_hat, mhat, nhat, phat)

    # Dot products
    mhat_phat = np.dot(mhat,phat)
    nhat_phat = np.dot(nhat,phat)
    omg_hat_phat = np.dot(omg_hat,phat)
    
    # Equations 9-10 of https://arxiv.org/pdf/1204.4218.pdf
    if(omg_hat_phat != -1):
        Fp   = 0.5 * (mhat_phat**2 - nhat_phat**2) / (1+omg_hat_phat)
        Fx   = mhat_phat * nhat_phat / (1+omg_hat_phat)
    else:
        print("Warning: The pulsar and GW source are collinear. Antenna patterns are undefined.")
        Fp=Fx=np.nan
    
    return -omg_hat_phat, Fp, Fx
"""


# Antenna pattern based on K. J. Lee et al 2011, MNRAS 414, 3251–3264, Appendix A
# Currently in use for this code

def antenna_pattern(alpha_gw, delta_gw, alpha_p, delta_p):

    """
    Arguments:
        alpha_gw    is RA  of GW source
        delta_gw    is DEC of GW source
        alpha_p     is RA  of pulsar
        delta_p     is DEC of pulsar

    Returns:
        coseta      Dot product of direction to GW source and direction to pulsar
        Fp          Antenna pattern F+
        Fx          Antenna pattern Fx 
    """
    
    n1 = cos(alpha_p)*cos(delta_p)
    n2 = sin(alpha_p)*cos(delta_p)
    n3 = sin(delta_p)
    
    cos_theta = cos(delta_gw)*cos(delta_p)*cos(alpha_gw-alpha_p) + sin(delta_gw)*sin(delta_p)
    
    e11p = (sin(alpha_gw))**2 - (cos(alpha_gw))**2 * (sin(delta_gw))**2
    e12p = -sin(alpha_gw)*cos(alpha_gw) * ((sin(delta_gw))**2 + 1)
    e13p = cos(alpha_gw)*sin(delta_gw)*cos(delta_gw)
    e21p = -sin(alpha_gw)*cos(alpha_gw) * ((sin(delta_gw))**2 + 1)
    e22p = (cos(alpha_gw))**2 - (sin(alpha_gw))**2 * (sin(delta_gw))**2
    e23p = sin(alpha_gw)*sin(delta_gw)*cos(delta_gw)
    e31p = cos(alpha_gw)*sin(delta_gw)*cos(delta_gw)
    e32p = sin(alpha_gw)*sin(delta_gw)*cos(delta_gw)
    e33p = -(cos(delta_gw))**2
    
    Fp = (n1*(n1*e11p+n2*e12p+n3*e13p)+
          n2*(n1*e21p+n2*e22p+n3*e23p)+
          n3*(n1*e31p+n2*e32p+n3*e33p))
    Fp = 0.5 * Fp/ (1-cos_theta)
    
    e11c = sin(2*alpha_gw) * sin(delta_gw)
    e12c = -cos(2*alpha_gw) * sin(delta_gw)
    e13c = -sin(alpha_gw) * cos(delta_gw)
    e21c = -cos(2*alpha_gw) * sin(delta_gw)
    e22c = -sin(2*alpha_gw) * sin(delta_gw)
    e23c = cos(alpha_gw) * cos(delta_gw)
    e31c = -sin(alpha_gw) * cos(delta_gw)
    e32c = cos(alpha_gw) * cos(delta_gw)
    e33c  = 0
    
    Fx = (n1*(n1*e11c+n2*e12c+n3*e13c)+
          n2*(n1*e21c+n2*e22c+n3*e23c)+
          n3*(n1*e31c+n2*e32c+n3*e33c))
    Fx = 0.5 * Fx/ (1-cos_theta)
    
    if (cos_theta ==1):
        print("Warning: The pulsar and GW source are collinear. Antenna patterns are undefined.")
        Fp=Fx=np.nan
    
    return cos_theta, Fp, Fx
