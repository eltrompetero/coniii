# Equations of 2-spin Ising model.

# 25/10/2018
from numpy import zeros, exp

def calc_observables(params):
	"""
	Give each set of parameters concatenated into one array.
	"""
	Cout = zeros((3))
	H = params[0:2]
	J = params[2:3]
	Z = +exp(+H[0]+H[1]+J[0])+exp(+H[0]-H[1]-J[0])+exp(-H[0]+H[1]-J[0])+exp(-H[0]-H[1]+J[0])
	Cout[0] = (+exp(+H[0]+H[1]+J[0])+exp(+H[0]-H[1]-J[0])+exp(-H[0]+H[1]-J[0])*-1+exp(-H[0]-H[1]+J[0])*-1)/Z
	Cout[1] = (+exp(+H[0]+H[1]+J[0])+exp(+H[0]-H[1]-J[0])*-1+exp(-H[0]+H[1]-J[0])+exp(-H[0]-H[1]+J[0])*-1)/Z
	Cout[2] = (+exp(+H[0]+H[1]+J[0])+exp(+H[0]-H[1]-J[0])*-1+exp(-H[0]+H[1]-J[0])*-1+exp(-H[0]-H[1]+J[0]))/Z

	return(Cout)

def p(params):
	"""
	Give each set of parameters concatenated into one array.
	"""
	Cout = zeros((3))
	H = params[0:2]
	J = params[2:3]
	H = params[0:2]
	J = params[2:3]
	Pout = zeros((4))
	Z = +exp(+H[0]+H[1]+J[0])+exp(+H[0]-H[1]-J[0])+exp(-H[0]+H[1]-J[0])+exp(-H[0]-H[1]+J[0])
	Pout[0] = +exp(+H[0]+H[1]+J[0])/Z
	Pout[1] = +exp(+H[0]-H[1]-J[0])/Z
	Pout[2] = +exp(-H[0]+H[1]-J[0])/Z
	Pout[3] = +exp(-H[0]-H[1]+J[0])/Z

	Pout = Pout[::-1]
	return(Pout)
