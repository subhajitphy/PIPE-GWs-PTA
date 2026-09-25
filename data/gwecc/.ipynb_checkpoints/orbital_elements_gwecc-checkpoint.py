import numpy as np
from numpy import sin, cos, tan, arctan, sqrt, pi as pi

def pn(term,order):
    sum1=0
    for i in range(order+1):
        sum1+=term[i]
    return sum1
    

def ephietE(x,et,eta,order):
	epet0=1
	epet1=(4 - eta)*x
	epet2=(((4*(-312 - 180*sqrt(1 - et**2) + 17*eta + 72*sqrt(1 - et**2)*eta + eta**2) + et**2*(1152 - 659*eta + 41*eta**2))*x**2)/(96*(-1 + et**2)))
	epet3=(-1/26880*((70*et**4*(-384*(35 + 32*sqrt(1 - et**2)) + (6912 + 11233*sqrt(1 - et**2))*eta + (192 - 1915*sqrt(1 - et**2))*eta**2 + 15*sqrt(1 - et**2)*eta**3) + 20*(-1344*(65 + 54*sqrt(1 - et**2)) 
		+ 4*(29960 + 33431*sqrt(1 - et**2))*eta + 56*(-240 + 247*sqrt(1 - et**2))*eta**2 - 861*(1 + sqrt(1 - et**2))*eta*pi**2) + et**2*(107520*(25 + 19*sqrt(1 - et**2)) - 32*(90020 + 38949*sqrt(1 - et**2))*eta 
		+ 420*(608 + 1319*sqrt(1 - et**2))*eta**2 - 18900*sqrt(1 - et**2)*eta**3 + 4305*(4 + sqrt(1 - et**2))*eta*pi**2))*x**3)/(1 - et**2)**(5/2))

	return pn([epet0,epet1,epet2,epet3],order)

def vE(x,et,eta,u,order):
    ehi=et*ephietE(x,et,eta,order)
    f1=2*arctan(sqrt((1+ephi)/(1-ephi))*tan(u/2))
    return f1
	
def rE(x,et,eta,u,order):
	w=1-et*cos(u)
	OTS=sqrt(1-et*et)

	rE_0=1
	rE_1=(1/6)*x*(((7*eta-6)*w-9*eta+24)*et**2+(18-7*eta)*w+9*eta-24)/(OTS**2*w)
	rE_2=(1/72)*x**2*(((35*eta**2-231*eta+72)*et**4+(150*eta-70*eta**2+468)*et**2+648-567*eta+35*eta**2)*w+(261*eta-27*eta**2)*et**4+(288-1026*eta+54*eta**2)*et**2
	-288-27*eta**2+765*eta-36*(2*eta-5)*(w-3)*OTS**3)/(w*OTS**4)
	rE_3=(1/181440)*x**3*((((-1254960*eta+302400*eta**2+453600)*w-498960*eta+362880*eta**2+1360800)*et**4+((5725440*eta-1239840*eta**2-38745*pi**2*eta)*w+116235*pi**2*eta-13003200*eta+1179360*eta**2
	+5443200)*et**2+(4989600+38745*pi**2*eta-6647760*eta+937440*eta**2)*w+13502160*eta-6804000-1542240*eta**2-116235*pi**2*eta)*OTS+((986580*eta+6860*eta**3-550620*eta**2-120960)*w+525420*eta**2-11340*eta**3-963900*eta)*et**6+((2358720-20580*eta**3
	-3458700*eta+2458260*eta**2)*w+34020*eta**3+2903040-9259596*eta+1946700*eta**2+116235*pi**2*eta)*et**4+((-20173860*eta+16148160+3539340*eta**2+20580*eta**3+116235*pi**2*eta)*w+232470*pi**2*eta+948780*eta-34020*eta**3-1296540*eta**2)*et**2
	+(-6860*eta**3+4717440+1220940*eta**2+464940*pi**2*eta-17875620*eta)*w-2903040-1175580*eta**2+11340*eta**3-348705*pi**2*eta+9274716*eta)/(w*OTS**6)
	fac=w/x

	return fac*pn([rE_0,rE_1,rE_2,rE_3],order)


def delta_L(x,et,eta,u,order):
    ephi=ephietE(x,et,eta,order)*et
    v=2*arctan(sqrt((1+ephi)/(1-ephi))*tan(u/2))
    dl2=((x**2*(12*(-5 + 2*eta)*(u - v) - et*(-15 + eta)*eta*sin(v)))/(8*sqrt(1 - et**2)))
    dl3=((x**3*(et*(67200 - 525*(112 + 27*et**2)*eta**2 + 35*(-8 + 65*et**2)*eta**3 + eta*(93468 - 315*et**2 - 4305*pi**2))*sin(v) 
		+ 35*(-((2880 - 10880*eta + 960*eta**2 + 96*et**2*(30 - 29*eta + 11*eta**2) + 123*eta*pi**2)*(u - v)) 
		+ 12*et**2*eta*(116 - 49*eta + 3*eta**2)*sin(2*v) + et**3*eta*(23 - 73*eta + 13*eta**2)*sin(3*v))))/(6720*(1 - et**2)**(3/2)))

   
    if order==0 or order==1:
    	return 0 
    if order==2:
    	return dl2
    if order==3:
    	return dl2+dl3

def rtE(x,et,eta,u,order):

	OTS=sqrt(1-et*et)
	w=1-et*cos(u)
	rtE_0=1
	rtE_1=((-7*eta + et**2*(-6 + 7*eta))*x)/(6*OTS**2)
	rtE_2=((-135*eta + 9*eta**2 + et**2*(405*eta - 27*eta**2) + et**6*(135*eta - 9*eta**2) + et**4*(-405*eta + 27*eta**2) 
		+ (-540 + 351*eta - 9*eta**2 + et**4*(-540 + 351*eta - 9*eta**2) + et**2*(1080 - 702*eta + 18*eta**2))*w + (-324 + 189*eta + 35*eta**2 + et**2*(-234 + 366*eta - 70*eta**2) + et**4*(72 - 231*eta + 35*eta**2))*w**3 
		- 36*(1 - et**2)*(-5 + 2*eta)*OTS*w**2*(3 + w))*x**2)/(72*OTS**4*w**3)
	rtE_3=((-22680*eta*(23 - 73*eta + 13*eta**2)*OTS**10 + 22680*eta*(-635 + 53*eta + 23*eta**2)*OTS**8*w + OTS**6*(-14515200 - 4490640*eta**2 
		- 196560*eta**3 + et**2*(1088640*eta - 1315440*eta**2 + 151200*eta**3) + eta*(32397408 + 232470*pi**2))*w**2 + OTS**4*(9072000 - 1451520*eta**2 - 30240*eta**3 + et**2*(-2721600 + 7892640*eta - 2494800*eta**2 
		+ 30240*eta**3) + eta*(6264432 - 464940*pi**2))*w**3 + OTS**3*(5443200 + 3084480*eta**2 + et**2*(2721600 - 997920*eta + 725760*eta**2) + eta*(-23738400 + 232470*pi**2))*w**4 + (-4717440 - 3591000*eta**2 
		- 13720*eta**3 + et**4*(1179360 - 6191640*eta + 4190760*eta**2 - 41160*eta**3) + et**6*(-241920 + 1973160*eta - 1101240*eta**2 + 13720*eta**3) + eta*(23806440 - 464940*pi**2) + et**2*(-11249280 - 6166440*eta**2 
		+ 41160*eta**3 + eta*(16034760 - 116235*pi**2)) + OTS*(1814400 + 1874880*eta**2 + et**4*(907200 - 2509920*eta + 604800*eta**2) + eta*(-10029600 + 77490*pi**2) + et**2*(-8164800 - 2479680*eta**2 + eta*(14716800 
		- 77490*pi**2))))*w**5)*x**3)/(362880*OTS**6*w**5)
	
	fac=et*sqrt(x)*sin(u)/w
	
	return fac*pn([rtE_0,rtE_1,rtE_2,rtE_3],order)
	
	


def phitE(x,et,eta,u,order):

	OTS=sqrt(1-et*et)
	w=1-et*cos(u)

	phitE_0=1
	phitE_1=((-4 + eta)*(-1 + et**2 + w)*x)/(OTS**2*w)
	phitE_2=((-6*eta*(3 + 2*eta)*OTS**6 + (108 + 63*eta + 33*eta**2)*OTS**4*w + OTS**2*(-240 - 31*eta - 29*eta**2 + et**2*(48 - 17*eta + 17*eta**2) 
		+ (180 - 72*eta)*OTS)*w**2 + (42 + 22*eta + 8*eta**2 + et**2*(-147 + 8*eta - 14*eta**2) + (-90 + 36*eta)*OTS)*w**3)*x**2)/(12*OTS**4*w**3)
	phitE_3=((5040*eta*(-3 + 8*eta + 2*eta**2)*OTS**10 + 1120*eta*(-349 
		- 186*eta + 6*eta**2)*OTS**8*w + 4*eta*OTS**6*(539788 + 20160*eta - 19600*eta**2 + et**2*(4200 - 5040*eta + 1120*eta**2) - 4305*pi**2)*w**2 + 140*OTS**4*(-4032 - 15688*eta + 1020*eta**2 + 724*eta**3 
		+ et**2*(1728 + 3304*eta - 612*eta**2 - 460*eta**3) + (8640 - 5616*eta + 864*eta**2)*OTS)*w**3 + 4*OTS**2*(127680 - 32900*eta**2 - 11060*eta**3 + et**4*(4620*eta + 3220*eta**2 - 4060*eta**3) + eta*(19372 
		+ 12915*pi**2) + et**2*(-252000 + 98560*eta**2 + 16800*eta**3 + (134400 - 119280*eta + 40320*eta**2)*OTS + eta*(-300528 + 4305*pi**2)) + OTS*(-235200 + eta*(-162400 + 4305*pi**2)))*w**4 + (-147840 + 8960*eta**2 
		+ 4480*eta**3 + et**4*(-221760 - 113680*eta + 94640*eta**2 + 13440*eta**3) + eta*(1127280 - 43050*pi**2) + OTS*(-67200 - 53760*eta**2 + eta*(674240 - 8610*pi**2)) + et**2*(-194880 - 112000*eta**2 - 11200*eta**3 
		+ (-739200 + 544320*eta - 127680*eta**2)*OTS + eta*(692928 + 12915*pi**2)))*w**5)*x**3)/(13440*OTS**6*w**5)
	fac=OTS*x**(3/2)/w**2

	return fac*pn([phitE_0,phitE_1,phitE_2,phitE_3],order)




def x_x1(x1,et,eta,order):


	x_x1_0=x1
	x_x1_1=(-2*x1**2)/(-1 + et**2)
	x_x1_2=(((72 + 51*et**2 - 28*eta - 26*et**2*eta)*x1**3)/(6*(-1 + et**2)**2))
	x_x1_3=(((-1920 - 1920*et**2 + 3840*et**4 - 16000*sqrt(1 - et**2) - 26496*et**2*sqrt(1 - et**2) - 2496*et**4*sqrt(1 - et**2) 
		+ 768*eta + 768*et**2*eta - 1536*et**4*eta + 24480*sqrt(1 - et**2)*eta + 27008*et**2*sqrt(1 - et**2)*eta + 1760*et**4*sqrt(1 - et**2)*eta - 896*sqrt(1 - et**2)*eta**2 
		- 5120*et**2*sqrt(1 - et**2)*eta**2 - 1040*et**4*sqrt(1 - et**2)*eta**2 - 492*sqrt(1 - et**2)*eta*pi**2 - 123*et**2*sqrt(1 - et**2)*eta*pi**2)*x1**4)/(192*sqrt(1 - et**2)*(-1 + et**2)**3))
	
	return pn([x_x1_0,x_x1_1,x_x1_2,x_x1_3],order)




def get_k(x,et,eta,order):
	order=order-1
	OTS=sqrt(1-et*et)
	k0=1
	k1=(1/12)*x*((51-26*eta)*et**2+54-28*eta)/OTS**2
	k2=(1/384)*x**2*(-384*(2*et**2+1)*(2*eta-5)*OTS+(1040*eta**2+2496-1760*eta)*et**4
		+(18336+5120*eta**2+(123*pi**2-22848)*eta)*et**2+6720+896*eta**2+(-20000+492*pi**2)*eta)/OTS**4
	fac=3*x/OTS**2
	return fac*pn([k0,k1,k2],order)

def get_l(x,et,eta,u,order):
	
	OTS=sqrt(1-et*et)
	bb=get_beta(x,et,eta,order)
	vmu=2*arctan(bb*sin(u)/(1-bb*cos(u)))
	w=et*cos(u)-1
	L_0=u-et*sin(u)
	L_2=(1/8)*x**2*((60-24*eta)*w*vmu+(-eta**2+15*eta)*et*sin(u)*OTS)/(OTS*w)
	L_3=(1/6720)*(35*w**3*(-2784*et**2*eta+1056*et**2*eta**2-10880*eta+960*eta**2+123*pi**2*eta+2880+2880*et**2)*vmu
	-(((11620*eta**2-1820*eta**3+1120*eta)*et**2-67200+1820*eta**3+11900*eta**2+(4305*pi**2+51152)*eta)*w**2+((141400*eta-36680*eta**2-280*eta**3)*et**2-141400*eta+36680*eta**2+280*eta**3)*w+(10220*eta**2-3220*eta
	-1820*eta**3)*et**4+(-20440*eta**2+3640*eta**3+6440*eta)*et**2+10220*eta**2-3220*eta-1820*eta**3)*OTS*et*sin(u))*x**3/(OTS**3*w**3)
	return pn([L_0,0,L_2,L_3],order)

def get_beta(x,et,eta,order):
	OTS=sqrt(1-et*et)
	beta_0=1
	beta_1=(8-2*eta+(4-eta)/OTS)*x
	beta_2=((89/48)*eta**2-(1043/48)*eta+40+((137/96)*eta**2+43-(2003/96)*eta)/OTS+((85/16)*eta+35/2-(7/16)*eta**2)/OTS**2+(9+(1/32)*eta**2+(69/32)*eta)/OTS**3)*x**2
	beta_3=(-(179/192)*eta**3
		-(8795/64)*eta+160+(5207/192)*eta**2+((3473/128)*eta**2-(343/384)*eta**3-(70337/384)*eta+258)/OTS+((617/32)*eta**2+175-(83777/3360)*eta-(71/96)*eta**3+(41/128)*pi**2*eta)/OTS**2+(-(953597/6720)*eta-(11/96)*eta**3
		+(1303/48)*eta**2+(369/256)*pi**2*eta+158)/OTS**3+(-(2583/64)*eta**2+116+(83/64)*eta**3-(245573/960)*eta+(205/128)*pi**2*eta)/OTS**4+((81/128)*eta**3-(158893/1920)*eta-(3577/128)*eta**2+(123/256)*pi**2*eta+46)/OTS**5)*x**3
	fac=(1-OTS)/et
	return fac*pn([beta_0,beta_1,beta_2,beta_3],order)

def get_w(x,et,eta,u,order):
	
	OTS=sqrt(1-et*et)
	bb=get_beta(x,et,eta,order)
	vmu=2*arctan(bb*sin(u)/(1-bb*cos(u)))
	w=et*cos(u)-1
	et_snu=et*sin(u)
	W_0=et*sin(u)+vmu
	W_1=3*(vmu+et*sin(u))*x/OTS**2
	W_2=(1/32)*(8*(((-12*eta+30)*et**2-30+12*eta)*OTS+(51-26*eta)*et**2+54-28*eta)*w**3*vmu+(-4*OTS**5*eta*(-1+3*eta)+8*(18*eta+1)*OTS**3*w+(((3*eta**2-eta)*et**2-148*eta-8+12*eta**2)*OTS
		+(-60*eta+4*eta**2)*et**4+(120*eta-8*eta**2)*et**2-60*eta+4*eta**2)*w**2+((-208*eta+408)*et**2-224*eta+432)*w**3)*et_snu)*x**2/(w**3*OTS**4)
	W_3=(1/26880)*(1680*OTS**9*et_snu*eta*(1-5*eta+5*eta**2)+4480*OTS**7*et_snu*eta*(28-36*eta+3*eta**2)*w
		+4*et_snu*OTS**4*w**2*(((840*eta+2940*eta**3-3220*eta**2)*et**2+13440-5460*eta**3-62300*eta**2+(-4305*pi**2+265248)*eta)*OTS-140*eta*(23-73*eta+13*eta**2)*OTS**4)+2*et_snu*OTS**2*w**3*(((-1260*eta**3-140700*eta**2+206640*eta+6720)*et**2+16800
		-5040*eta**3+105840*eta**2+(-327808-12915*pi**2)*eta)*OTS+560*eta*(-505+eta**2+131*eta)*OTS**4)+et_snu*w**4*(((-1050*eta+4270*eta**2-3990*eta**3)*et**4+(-40320-36540*eta**3+594300*eta**2+(4305*pi**2-1030488)*eta)*et**2
		-87360+10080*eta**3+207200*eta**2+(43050*pi**2-532496)*eta)*OTS+4*((11620*eta**2-1820*eta**3+1120*eta)*et**2-67200+1820*eta**3+14420*eta**2+(4305*pi**2+13352)*eta)*OTS**4)+210*et_snu*w**5*(-384*(2*et**2+1)*(2*eta-5)*OTS
		+(1040*eta**2+2496-1760*eta)*et**4+(18336+5120*eta**2+(123*pi**2-22848)*eta)*et**2+6720+896*eta**2+(-20000+492*pi**2)*eta)+70*w**5*vmu*(((5760+2112*eta**2-5568*eta)*et**4+(20160-192*eta**2+(246*pi**2-24256)*eta)*et**2
		-8640-1920*eta**2+(22912-246*pi**2)*eta)*OTS+(7488+3120*eta**2-5280*eta)*et**4+(55008+15360*eta**2+(369*pi**2-68544)*eta)*et**2+20160+2688*eta**2+(-60000+1476*pi**2)*eta))*x**3/(w**5*OTS**6)
	
	return pn([W_0,W_1,W_2,W_3],order)

# from scipy.optimize import minimize, fsolve

# def get_u_(l,et):
    
#     res = fsolve(lambda u: u-et*sin(u)-l, 0)[0]
#     return res

from mikkola import get_u as get_u_



def u_from_l_3PN(l,e,x,eta,order):    
    if order==0:
        return get_u_(l,e)
    else:
        for i in range(4):
            u0 = get_u_(l,e)
            dl0=delta_L(x,e,eta,u0,order)
            l1 = l-dl0
            l=l1
        return u0



def hms_to_rad(hh, mm, ss):
    sgn = np.sign(hh)
    return sgn * (sgn * hh + mm / 60 + ss / 3600) * np.pi / 12


def dms_to_rad(dd, mm, ss):
    sgn = np.sign(dd)
    return sgn * (sgn * dd + mm / 60 + ss / 3600) * np.pi / 180
