# import astropy.constants as ac

# #Constants to be used later
# GMsun = ac.GM_sun.value                                  #G*(Solar Mass)
# c = ac.c.value                                           #Speed of light in vacuum
GMsun=1.3271244e+20
c=299792458.0
dsun = GMsun/(c**2)                                      #Natural length scale
tsun = GMsun/(c**3)                                      #Natural time scale
#pc = ac.pc.value    #One parsec
pc=3.085677581491367e+16
yr = (365.25)*(24)*(60)*(60)                             #Seconds in one year