# Code written by Adrienne Propp in May 2018

# Based on Monodomain-FentonKarma.py written by Ricardo Ruiz Baier,
#	 modified to reflect the Bueno-Orovio model, with TNNP parameters
# Parameter names follow those in Bueno-Orovio (2008) though
#	 final solutions are in the form (v,r1,r2,r3) to match project report

from dolfin import *
import time

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
list_linear_solver_methods()

fileO = XDMFFile("out/monodomain-BuenoOrovio-amp3.xdmf");
t = 0.0; dt = 0.3; T = 500.0; frequencySave = 50;
fileO.parameters['rewrite_function_mesh'] = False # This makes each variable separate rather than adding components to the same solution
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True



# ********* Mesh and finite dimensional spaces ********* #
nps = 100; L = 10 # is L wall thickness?
mesh =  RectangleMesh(Point(0,0),Point(L,L),nps,nps,"crossed")

Element = FiniteElement("CG", mesh.ufl_cell(), 1)
Mh = FunctionSpace(mesh, Element)
#Mh = FunctionSpace(mesh, "Lagrange", 1) # This shouldn't make a difference & doesn't seem to
Nh = FunctionSpace(mesh,MixedElement([Element,Element,Element,Element]))

Sol = Function(Nh)
(v,r1,r2,r3) = TrialFunctions(Nh) # v is transmemb pot, r are ionic quantities
(w,s1,s2,s3) = TestFunctions(Nh)


# ********* Model coefficients and parameters ********* #
diffScale = Constant(1.0e-3) # Time scale to ms from s
D0 = 1.171*diffScale

# TNNP
#M_uo = Constant(0.0)
#M_uu = Constant(1.58)
#M_tetav = Constant(0.3)
#M_tetaw  = Constant(0.015)
#M_tetavm = Constant(0.015)
#M_tetao = Constant(0.006)
#M_tauv1 = Constant(60.0)
#M_tauv2 = Constant(1150.0)
#M_tauvp = Constant(1.4506)
#M_tauw1 = Constant(70.0)
#M_tauw2 = Constant(20.0)
#M_kw = Constant(65.0)
#M_uw = Constant(0.03)
#M_tauwp = Constant(280.0)
#M_taufi = Constant(0.11)
#M_tauo1 = Constant(6.0)
#M_tauo2 = Constant(6.0)
#M_tauso1 = Constant(43.0)
#M_tauso2 = Constant(0.2)
#M_kso = Constant(2.0)
#M_uso = Constant(0.65)
#M_taus1 = Constant(2.7342)
#M_taus2 = Constant(3.0)
#M_ks = Constant(2.0994)
#M_us = Constant(0.9087)
#M_tausi = Constant(2.8723)
#M_tauwinf = Constant(0.07)
#M_winfstar = Constant(0.94)

# Epicardial parameters
M_uo = Constant(0.0)
M_uu = Constant(1.55)
M_tetav = Constant(0.3)
M_tetaw  = Constant(0.13)
M_tetavm = Constant(0.006)
M_tetao = Constant(0.006)
M_tauv1 = Constant(60.0)
M_tauv2 = Constant(1150.0)
M_tauvp = Constant(1.4506)
M_tauw1 = Constant(60.0)
M_tauw2 = Constant(15.0)
M_kw = Constant(65.0)
M_uw = Constant(0.03)
M_tauwp = Constant(200.0)
M_taufi = Constant(0.11)
M_tauo1 = Constant(400.0)
M_tauo2 = Constant(6.0)
M_tauso1 = Constant(30.0181)
M_tauso2 = Constant(0.9957)
M_kso = Constant(2.0458)
M_uso = Constant(0.65)
M_taus1 = Constant(2.7342)
M_taus2 = Constant(16.0)
M_ks = Constant(2.0994)
M_us = Constant(0.9087)
M_tausi = Constant(1.8875)
M_tauwinf = Constant(0.07)
M_winfstar = Constant(0.94)


# ********* Initial  conditions, forcing terms ******* #
# Initial conditions from paper
vold = interpolate(Constant(0.0),Mh) # u in paper
r1old = interpolate(Constant(1.0),Mh) # v
r2old = interpolate(Constant(1.0),Mh) # w
r3old = interpolate(Constant(0.0),Mh) # s

# Not sure about the rest of this - how do we determine Iext?
stim_t1   = 1.0
stim_t2   = 330.0
stim_dur1 = 3.0
stim_dur2 = 3.0
stim_amp  = 3.0

#waveS1 = Expression("amp*(x[0] < 0.2)", amp=stim_amp, L=L, degree=3)
#waveS2 = Expression("amp*(x[0]>= 3.2 && x[0] <= 3.4 && x[1] <= 2.6)",\
                  #  amp = stim_amp, degree=3) # this is a strip
waveS1 = Expression("amp*(x[0]<=0.01*L)", amp=stim_amp, L=L, degree=2)
waveS2 = Expression("amp*(x[1] < 0.5*L && x[0] < 0.5*L)", amp = stim_amp, L=L, degree=2) # this is a square


def Istim(t):
    if (stim_t1 <= t and t <= stim_t1 + stim_dur1):
        return waveS1
    if (stim_t2 <= t and t <= stim_t2 + stim_dur2):
        return waveS2
    else:
        return Constant(0.0)


# ***** Linearized weak forms ******** # 
Left = v/dt*w*dx \
	 + inner(D0*grad(v),grad(w))*dx \
	 + r1/dt*s1*dx + r2/dt*s2*dx + r3/dt*s3*dx

LHS = assemble(Left) # acquire tensor form
solver = LUSolver(LHS)
solver.parameters["reuse_factorization"] = True


# ************* Time loop ************ #
start = time.clock(); inc = 0

while (t <= T):
	print "t=", t

	# Heaviside functions
	Hv = conditional(ge(vold,M_tetav), 1.0, 0.0)
	Hw = conditional(ge(vold,M_tetaw), 1.0, 0.0)
	#Hs = conditional(ge(vold,M_us), 1.0, 0.0)
	Hvm = conditional(ge(vold,M_tetavm), 1.0, 0.0)
	Ho = conditional(ge(vold,M_tetao), 1.0, 0.0)

	# Extra parameters defined from given parameters
	tau_vm = (1.0 - Hvm)*M_tauv1 + Hvm*M_tauv2
	tau_wm = M_tauw1 + (M_tauw2 - M_tauw1)*(1.0 + tanh(M_kw*(vold - M_uw)))/2.0
	tau_so = M_tauso1 + (M_tauso2 - M_tauso1)*(1.0 + tanh(M_kso*(vold - M_uso)))/2.0
	tau_s = (1.0 - Hw)*M_taus1 + Hw*M_taus2
	tau_o = (1.0 - Ho)*M_tauo1 + Ho*M_tauo2
	#vinf = conditional(ge(vold,M_tetavm), 0.0, 1.0)
	vinf = conditional(ge(M_tetavm,vold), 1.0, 0.0)
	winf = (1.0 - Ho)*(1.0 - vold/M_tauwinf) + Ho*M_winfstar

	# Currents
	Jfi = -r1old*Hv*(vold - M_tetav)*(M_uu - vold)/M_taufi
	Jso = (vold - M_uo)*(1.0 - Hw)/tau_o + Hw/tau_so
	Jsi = -Hw*r2old*r3old/M_tausi
	Iion = Jfi + Jso + Jsi # Check sign - I think this is right but disagrees with report writeup

	# Reaction terms, m(v,r)
	ReactR1 = (1.0 - Hv)*(vinf - r1old)/tau_vm - Hv*r1old/M_tauvp
	ReactR2 = (1.0 - Hw)*(winf - r2old)/tau_wm - Hw*r2old/M_tauwp
	ReactR3 = ((1.0 + tanh(M_ks*(vold - M_us)))/2.0 - r3old)/tau_s

	# Put it all together in variational form
	Right = vold*w/dt*dx + r1old*s1/dt*dx + r2old*s2/dt*dx + r3old*s3/dt*dx \
		+ (-Iion + Istim(t))*w*dx \
		+ ReactR1*s1*dx + ReactR2*s2*dx + ReactR3*s3*dx
		# The sign of Iion should definitely be negative - this is correct

	RHS = assemble(Right)
   #solve(LHS == RHS, Sol, solver_parameters={'linear_solver':'bicgstab'})
	solver.solve(Sol.vector(), RHS)
	v,r1,r2,r3 = Sol.split()

	if (inc % frequencySave == 0):
		v.rename("v","v"); fileO.write(v,t)
		r1.rename("r1","r1"); fileO.write(r1,t)
		r2.rename("r2","r2"); fileO.write(r2,t)
		r3.rename("r3","r3"); fileO.write(r3,t)

	# Updating solutions
	assign(vold,v); assign(r1old,r1); assign(r2old,r2);assign(r3old,r3)
	t += dt; inc += 1

# ************* End **************** #    
print "Time elapsed: ", time.clock()-start
