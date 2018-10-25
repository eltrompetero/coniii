# Equations of 3-spin Ising model.

# 25/10/2018
from numpy import zeros, exp

def calc_observables(params):
	"""
	Give each set of parameters concatenated into one array.
	"""
	Cout = zeros((6))
	H = params[0:3]
	J = params[3:6]
	Z = 	+exp(+0)+exp(+H[2]+0)+exp(+H[1]+0)+exp(+H[1]+H[2]+J[2])+exp(+H[0]+0)+exp(+H[0]+H[2]+J[1])+exp(+H[0]+\
H[1]+J[0])+exp(+H[0]+H[1]+H[2]+J[0]+J[1]+J[2])
	Cout[0] = (+exp(+H[0]+0)+exp(+H[0]+H[2]+J[1])+exp(+H[0]+H[1]+J[0])+exp(+H[0]+H[1]+H[2]+J[0]+J[1]+J[2]))/Z
	Cout[1] = (+exp(+H[1]+0)+exp(+H[1]+H[2]+J[2])+exp(+H[0]+H[1]+J[0])+exp(+H[0]+H[1]+H[2]+J[0]+J[1]+J[2]))/Z
	Cout[2] = (+exp(+H[2]+0)+exp(+H[1]+H[2]+J[2])+exp(+H[0]+H[2]+J[1])+exp(+H[0]+H[1]+H[2]+J[0]+J[1]+J[2]))/Z
	Cout[3] = (+exp(+H[0]+H[1]+J[0])+exp(+H[0]+H[1]+H[2]+J[0]+J[1]+J[2]))/Z
	Cout[4] = (+exp(+H[0]+H[2]+J[1])+exp(+H[0]+H[1]+H[2]+J[0]+J[1]+J[2]))/Z
	Cout[5] = (+exp(+H[1]+H[2]+J[2])+exp(+H[0]+H[1]+H[2]+J[0]+J[1]+J[2]))/Z

	return(Cout)

def p(params):
	"""
	Give each set of parameters concatenated into one array.
	"""
	Cout = zeros((6))
	H = params[0:3]
	J = params[3:6]
	H = params[0:3]
	J = params[3:6]
	Pout = zeros((8))
	Z = 	+exp(+0)+exp(+H[2]+0)+exp(+H[1]+0)+exp(+H[1]+H[2]+J[2])+exp(+H[0]+0)+exp(+H[0]+H[2]+J[1])+exp(+H[0]+\
H[1]+J[0])+exp(+H[0]+H[1]+H[2]+J[0]+J[1]+J[2])
	Pout[0] = +exp(+0)/Z
	Pout[1] = +exp(+H[2]+0)/Z
	Pout[2] = +exp(+H[1]+0)/Z
	Pout[3] = +exp(+H[1]+H[2]+J[2])/Z
	Pout[4] = +exp(+H[0]+0)/Z
	Pout[5] = +exp(+H[0]+H[2]+J[1])/Z
	Pout[6] = +exp(+H[0]+H[1]+J[0])/Z
	Pout[7] = +exp(+H[0]+H[1]+H[2]+J[0]+J[1]+J[2])/Z

	return(Pout)
