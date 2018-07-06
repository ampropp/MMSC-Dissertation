# Code written by Adrienne Propp in June 2018
from dolfin import *
from fenics import *
import time
import numpy as np
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2
list_linear_solver_methods()

# Mesh
t = 0.0; dt = 0.3; T = 600.0; freqSave = 50; freqMech = 5;
nps = 100; L = 12 # is L wall thickness?
mesh =  RectangleMesh(Point(0,0),Point(L,L),nps,nps,"crossed")

# File
fileO = XDMFFile(mesh.mpi_comm(), "out/coupled_slab.xdmf");
fileO.parameters['rewrite_function_mesh'] = False # This makes each variable separate rather than adding components to the same solution
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True

# Define custom functions
def subplus(u) :
    return conditional(ge(u, 0.0), u, 0.0)


# ********* Finite dimensional spaces ********* #
P0 = FiniteElement("DG", mesh.ufl_cell(), 0) # Pressure
P1v = VectorElement("CG", mesh.ufl_cell(), 1) # Vector element for u in H1
P0t = TensorElement("DG", mesh.ufl_cell(), 0, symmetry= True) # For stress
TensorOut = TensorFunctionSpace(mesh, "DG", 0) # Tensor FE space
Hh = FunctionSpace(mesh,MixedElement([P0t,P1v,P0]))
Ta = Function(Mh) # Increasing over time, piecewise linear?
print " **************** Mech Dof = ", Hh.dim()


Element = FiniteElement("CG", mesh.ufl_cell(), 1)
Mh = FunctionSpace(mesh, "CG", 1)
Nh = FunctionSpace(mesh,MixedElement([Element,Element,Element,Element]))

# Fiber directions
f0   = Constant((1,0))
s0   = Constant((0,-1))
n0   = cross(f0,s0)

# Test & trial functions
Pup     = Function(Hh)
dPup    = TrialFunction(Hh)
(Pi,u,p)    = split(Pup)
(PiT,uT,pT) = TestFunctions(Hh)

Sol = Function(Nh)
(v,r1,r2,r3) = TrialFunctions(Nh) # v is transmemb pot, r are ionic quantities
(w,s1,s2,s3) = TestFunctions(Nh)


# ********* Mechanical parameters ********* #
a    = Constant(2.3621) # KPa
b    = Constant(10.810)
a_f  = Constant(1.16037) # KPa
b_f  = Constant(14.154)
a_s  = Constant(3.7245)
b_s  = Constant(5.1645)
a_fs = Constant(4.0108) # Pa, already scaled
b_fs = Constant(11.300)
eta  = Constant(0.001)
Ta0 = Constant(1.75)

# ******** Mechanical entities ************* #
ndim = u.geometric_dimension()
print " **************** Geometric Dim = ", ndim
I = Identity(ndim); F = I + grad(u); F = variable(F)
C = F.T*F; B = F*F.T; J = det(F); invF = inv(F)
I1 = tr(C); I8_fs = inner(f0, C*s0) 
I4_f = inner(f0, C*f0); I4_s = inner(s0, C*s0)
k = Constant(0.3)

# Passive Cauchy stress tensor (active is calculated in time loop)
CPassive = a*exp(b*subplus(I1-ndim))*B \
           +2*a_f*(I4_f-1)*exp(b_f*subplus(I4_f-1)**2)*outer(F*f0,F*f0) \
           +2*a_s*(I4_s-1)*exp(b_s*subplus(I4_s-1)**2)*outer(F*s0,F*s0) \
           +a_fs*I8_fs*exp(b_fs*subplus(I8_fs)**2)*(outer(F*f0,F*s0) \
                                                    + outer(F*s0,F*f0)) # where does this last term come from?


# ********* Electrophysiological coefficients and parameters ********* #
diffScale = Constant(1.0e-3) # Time scale to ms from s
D0 = 1.171*diffScale
D1 = 0.5*diffScale
D2 = 0.5*diffScale

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
Pi_old = Identity(Pi.geometric_dimension()) # pi


# ********* Stimulus ******* #
stim_t1   = 1.0
stim_t2   = 365.0
stim_dur1 = 3.0
stim_dur2 = 3.0
stim_amp  = 8.0
waveS1 = Expression("amp*(x[0]<=0.01*L)", amp=stim_amp, L=L, degree=2)
waveS2 = Expression("amp*(x[1] < 0.5*L && x[0] < 0.5*L)", amp = stim_amp, L=L, degree=2) # this is a square
def Istim(t):
    if (stim_t1 <= t and t <= stim_t1 + stim_dur1):
        return waveS1
    if (stim_t2 <= t and t <= stim_t2 + stim_dur2):
        return waveS2
    else:
        return Constant(0.0)

Ta = Expression("Ta0*pow(sin(DOLFIN_PI*t),2)", Ta0=Ta0, t=0.0, degree=3)

CActive = Ta / I4_f * outer(F*f0,F*f0) + k * Ta *I4_f / I4_s * outer(F*s0,F*s0) \
+ k * Ta * I4_f / I8_fs * (outer(F*f0,F*s0) + outer(F*s0,F*f0))

TotalCauchy = CPassive + CActive
calG =  J * TotalCauchy * invF.T 

FF = inner(Pi - calG + p*J*I, PiT) * dx \
    + inner(Pi, grad(uT)*invF.T) * dx \
    + dot(J*eta*invF.T*u,uT) * ds \
    + pT*(J-1)*dx

JJ = derivative(FF, Pup, dPup)



# ************* Time loop ************ #
start = time.clock(); inc = 0

while (t <= T):
	print "t=", t

	# Calculate D - determine if ref or current config
	#D = D0 + D1*v + D0*inner(f0,f0) # current config - not sure about inner()
  D = (D0 + D1*vold)*J*inv(C) + D0*J*outer(invF*f0,invF.T*f0) + D2*Pi_old # ref config - not sure about outer()

	# Linearized weak form - LHS
  Left = v/dt*w*dx \
	 + inner(D0*grad(v),grad(w))*dx \
	 + r1/dt*s1*dx + r2/dt*s2*dx + r3/dt*s3*dx
	LHS = assemble(Left) # acquire tensor form PROBLEM when using D not D0
	solver = LUSolver(LHS) # PROBLEM
	solver.parameters["reuse_factorization"] = True

	# Solve monodomain equations
	# Heaviside functions
	Hv = conditional(ge(vold,M_tetav), 1.0, 0.0)
	Hw = conditional(ge(vold,M_tetaw), 1.0, 0.0)
	Hvm = conditional(ge(vold,M_tetavm), 1.0, 0.0)
	Ho = conditional(ge(vold,M_tetao), 1.0, 0.0)
  
	# Extra parameters defined from given parameters
	tau_vm = (1.0 - Hvm)*M_tauv1 + Hvm*M_tauv2
	tau_wm = M_tauw1 + (M_tauw2 - M_tauw1)*(1.0 + tanh(M_kw*(vold - M_uw)))/2.0
	tau_so = M_tauso1 + (M_tauso2 - M_tauso1)*(1.0 + tanh(M_kso*(vold - M_uso)))/2.0
	tau_s = (1.0 - Hw)*M_taus1 + Hw*M_taus2
	tau_o = (1.0 - Ho)*M_tauo1 + Ho*M_tauo2
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


	if (inc % freqMech == 0):
		# Calculate Ta - will depend later on concentration
		Ta.t = t*1.0e3 # This is rescaled!


 
 		solve(FF == 0, Pup, J=JJ, \
        solver_parameters={'newton_solver':{'linear_solver':'lu',\
                                              'absolute_tolerance':5.0e-4,\
												'relative_tolerance':5.0e-4,\
												'maximum_iterations':10}})
    	# This is a projection step with the Kirchhoff stress
    	Pi,u,p  = Pup.split()
    	Pi_out = project(Pi,TensorOut)
    	assign(Pi_old,Pi_out)

    	if (inc % freqSave == 0):
			v.rename("v","v"); fileO.write(v,t)
			r1.rename("r1","r1"); fileO.write(r1,t)
			r2.rename("r2","r2"); fileO.write(r2,t)
			r3.rename("r3","r3"); fileO.write(r3,t)
			Pi_out.rename("Pi","Pi"); fileO.write(Pi_out,t)
			u.rename("u","u"); fileO.write(u,t)
			p.rename("p","p"); fileO.write(p,t)


	    # Updating solutions
	assign(vold,v); assign(r1old,r1); assign(r2old,r2); assign(r3old,r3); 
	t += dt; inc += 1

# ************* End **************** #    
print "Time elapsed: ", time.clock()-start
