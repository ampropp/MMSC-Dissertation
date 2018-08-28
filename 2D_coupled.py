##########################################################################################
## Written and modified by Ricardo Ruiz Baier and Adrienne Propp		        ##
##									        	##
## Code to simulate cardiac electromechanics on 2D slab of tissue        		##
## Uses an active stress formulation, coupled with Bueno-Orovio et al. (2008) model	##
## of human ventricular action potential						##
##											##
## This code is written for the hyperelastic case, but can be modified to the		##
## viscoelastic case by uncommenting the code in the relevant section.			##
##########################################################################################

# ********* Dependencies ********* #
# Note: this code uses FEniCS to solve using the finite element method
from dolfin import *
from mshr import *
import time


# ********* Setup ********* #
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True


# ********* Problem-specific parameters ********* #
L = 12.0; nps = 100
dt = 0.3; Tfinal = 600.0 # change as necessary; time in ms
frequencyMech = 5.0; Dt = dt*frequencyMech # mechanics will be solved on coarser timestep than electrophysiology


# ********* Create mesh ********* #
domain = Rectangle(Point(0,0),Point(L,L))
mesh = generage_mesh(domain,nps)
he = FacetArea(mesh)

# ********* Define file to save results ********* #
fileO = XDMFFile(mesh.mpi_comm(), "out/2D_coupled.xdmf")
fileO.parameters['rewrite_function_mesh']=False # mesh is consistent throughout our time interval
fileO.parameters["functions_share_mesh"] = True # each function at a given time shares the same mesh
fileO.parameters["flush_output"] = True # dataset can be saved after each timestep
# Note: these parameters help reduce filesize


# ********* User-defined functions ********* #
def subplus(u) :
    return conditional(ge(u, 0.0), u, 0.0)


# ********* Define elements ********* #
P0 = FiniteElement("DG", mesh.ufl_cell(), 0) # Pressure
P1 = FiniteElement("CG", mesh.ufl_cell(), 1) # Electric variables and active tension (Lagrange FE)
P1v = VectorElement("CG", mesh.ufl_cell(), 1) # Displacement (vector)
P0T = TensorElement("DG", mesh.ufl_cell(), 0, symmetry= True) # Stress (tensor)


# ********* Define function spaces ********* #
Mh = FunctionSpace(mesh, P1)
Vh = FunctionSpace(mesh, P1v)
Hh = FunctionSpace(mesh, MixedElement([P0T,P1v,P0,P1])) # Mechanical system
TensorOut = TensorFunctionSpace(mesh, "DG", 0) # Tensor FE space
Nh = FunctionSpace(mesh, MixedElement([P1,P1,P1,P1])) # Electrical system

print " **************** Mech Dof = ", Hh.dim()


# ********* Electrical functions ********* #
Sol = Function(Nh)
(v,r1,r2,r3) = TrialFunctions(Nh) # v is transmemb potential (voltage), r are ionic quantities
(w,s1,s2,s3) = TestFunctions(Nh)
# Note: in Bueno-Orovio et al. (2008), these are defined as (u,v,w,s)
# Note: r3 is a proxy for calcium


# ********* Mechanical functions ********* #
PupTa     = Function(Hh)
dPupTa    = TrialFunction(Hh)
(Pi,u,p,Ta) = split(PupTa) # (stress, displacement, pressure, active tension)
(PiT,uT,pT,Tb) = TestFunctions(Hh)

ndim = u.geometric_dimension(); print " **************** Geometric Dim = ", ndim


# ********* Fiber directions ********* #
f0   = Constant((1,0)) # fiber direction
s0   = Constant((0,-1)) # sheetlet normal direction
n0   = cross(f0,s0)


# ********* Initial values ********* #
vold = Function(Mh)
r1old = Function(Mh)
r2old = Function(Mh)
r3old = Function(Mh)
vold = interpolate(Constant(0.0),Mh) # initial values taken from Bueno-Orovio et al. (2008)
r1old = interpolate(Constant(1.0),Mh)
r2old = interpolate(Constant(1.0),Mh)
r3old = interpolate(Constant(0.0),Mh)


# ********* Pressure stabilisation ********* #
uold = interpolate(Constant((0.0,0.0)),Vh)
uuold = interpolate(Constant((0.0,0.0)),Vh)
stabP = Constant(0.05)


# ********* Electrical parameters ********* #
diffScale = Constant(1.0e-3) # Rescale time from s to ms

# Diffusion parameters
D0 = 1.171*diffScale # taken from Bueno-Orovio et al. (2008)
D1 = 9.0*diffScale*1.0e-1 # D1 and D2 are scaled such that D0 is dominant
D2 = 1.0*diffScale*1.0e-2

# Parameters taken from epicardial parameter set in Bueno-Orovio et al. (2008)
M_uo = Constant(0.0); M_uu = Constant(1.55)
M_tetav = Constant(0.3); M_tetaw  = Constant(0.13)
M_tetavm = Constant(0.006); M_tetao = Constant(0.006)
M_tauv1 = Constant(60.0); M_tauv2 = Constant(1150.0)
M_tauvp = Constant(1.4506)
M_tauw1 = Constant(60.0); M_tauw2 = Constant(15.0)
M_kw = Constant(65.0); M_uw = Constant(0.03)
M_tauwp = Constant(200.0); M_taufi = Constant(0.11)
M_tauo1 = Constant(400.0); M_tauo2 = Constant(6.0)
M_tauso1 = Constant(30.0181); M_tauso2 = Constant(0.9957)
M_kso = Constant(2.0458); M_uso = Constant(0.65)
M_taus1 = Constant(2.7342); M_taus2 = Constant(16.0)
M_ks = Constant(2.0994); M_us = Constant(0.9087)
M_tausi = Constant(1.8875); M_tauwinf = Constant(0.07)
M_winfstar = Constant(0.94)


# ********* Mechanical parameters ********* #
a    = Constant(0.23621) # KPa, rescaled down one order of mag
b    = Constant(10.810)
a_f  = Constant(0.116037) # KPa, rescaled down one order of mag
b_f  = Constant(14.154)
a_s  = Constant(0.37245) # Rescaled down one order of mag
b_s  = Constant(5.1645)
a_fs = Constant(4.0108) # Pa, already scaled
b_fs = Constant(11.300)
eta  = Constant(0.001) # determines stiffness of tissue


# ********* Electrical Stimuli ********* #
stim_t1   = 1.0
stim_t2   = 330.0
stim_dur1 = 3.0
stim_dur2 = 3.0
stim_amp  = 3.0
waveS1 = Expression("amp*(x[0]<=0.01*L)", amp=stim_amp, L=L, degree=2) # planar wave
waveS2 = Expression("amp*(x[1] < 0.5*L && x[0] < 0.5*L)", amp = stim_amp, L=L, degree=2) # square wave in lower left quadrant

# Define external stimulus  to be added to reaction-diffusion equation
def Istim(t):
    if (stim_t1 <= t and t <= stim_t1 + stim_dur1):
        return waveS1
    if (stim_t2 <= t and t <= stim_t2 + stim_dur2):
        return waveS2
    else:
        return Constant(0.0)


# ******** Mechanical entities ************* #
I = Identity(ndim); F = I + grad(u); F = variable(F) # F is deformation gradient
C = F.T*F; B = F*F.T; J = det(F); invF = inv(F) # C and B are Cauchy-Green deformation tensors
J = variable(J); C = variable(C)

# Define invariants
I1 = tr(C) # isometric invariant
I4_f = inner(f0, C*f0); I4_s = inner(s0, C*s0); I8_fs = inner(f0, C*s0) # anisotropic pseudoinvariants

k = Constant(0.3) # transverse fiber stress as a fraction of axial tension

# Define passive component of Cauchy stress tensor
CPassive = a*exp(b*subplus(I1-ndim))*B \
           +2*a_f*(I4_f-1)*exp(b_f*subplus(I4_f-1)**2)*outer(F*f0,F*f0) \
           +2*a_s*(I4_s-1)*exp(b_s*subplus(I4_s-1)**2)*outer(F*s0,F*s0) \
           +a_fs*I8_fs*exp(b_fs*subplus(I8_fs)**2)*(outer(F*f0,F*s0) \
                                                    + outer(F*s0,F*f0)) 

# Define active component of Cauchy stress tensor
CActive = Ta / I4_f * outer(F*f0,F*f0) + k * Ta *I4_f / I4_s * outer(F*s0,F*s0) 
  # + k * Ta * I4_f / I8_fs * (outer(F*f0,F*s0) + outer(F*s0,F*f0))

TotalCauchy = CPassive + CActive # total Cauchy stress tensor

###################
# VISCOELASTICITY #
###################
## Uncomment the following code to modify the model from hyperelastic to viscoelastic
#gam = Constant(22.6) # Ps*s from Krister thesis
#beta = Constant(0.01) # s
#Fdot = grad(u - uold)/Dt
#gradv = Fdot*invF
#d = 0.5*(gradv + gradv.T)
#Bdot = gradv*B + B*gradv.T
#Cdot = 2.0*F.T*d*F
#Idot = tr(Cdot)
#CVisco = gam*exp(beta*Idot)*Bdot
#TotalCauchy = TotalCauchy + CVisco

calG = J * TotalCauchy


# ******** Coupling ************* #
Pi_out = Function(TensorOut)

def DTens(v,u,PI):
	self_diff = (D0 + D1*v)*J*inv(C)
	aniso = D0*J*outer(invF*f0,invF.T*f0)
	SAD = D2*invF*PI*invF.T
	return self_diff + aniso + SAD

alpha = Constant(0.01) # determines relationship between calcium (here, r3)  and active tension Ta
calcium = Function(Mh)
calcium = r3old


# ******** Weak forms ************* #
Left = v/dt*w*dx \
  + inner(DTens(vold,uold,Pi)*grad(v),grad(w))*dx \
  + r1/dt*s1*dx + r2/dt*s2*dx + r3/dt*s3*dx

FF = 1.0 / pow(Dt,2.0) * dot(u-2.0*uold+uuold,uT) * dx \
     + inner(Pi - calG + p*J*I, PiT) * dx \
     + inner(Pi, grad(uT)*invF.T) * dx \
     + dot(J*eta*invF.T*u,uT) * ds \
     + pT*(J-1.0)*dx \
     + stabP*avg(he)*jump(p)*jump(pT)*dS \
     + Ta * Tb * dx \
     - alpha * Tb * calcium * dx \
     + 10.0*D0*dot(grad(Ta), grad(Tb)) * dx

JJ = derivative(FF, PupTa, dPupTa)


# ********* Time loop ************* #
t = 0.0; tMech = stim_t2 + 5.0 # begin mechanics after spiral has formed
inc = 0; frequencySave = 20

while (t <= Tfinal):
	print "t=", t

	## Bueno-Orovio model
	Hv = conditional(ge(vold,M_tetav), 1.0, 0.0)
	Hw = conditional(ge(vold,M_tetaw), 1.0, 0.0)
	Hvm = conditional(ge(vold,M_tetavm), 1.0, 0.0)
	Ho = conditional(ge(vold,M_tetao), 1.0, 0.0)
	tau_vm = (1.0 - Hvm)*M_tauv1 + Hvm*M_tauv2
	tau_wm = M_tauw1 + (M_tauw2 - M_tauw1)*(1.0 + tanh(M_kw*(vold - M_uw)))/2.0
	tau_so = M_tauso1 + (M_tauso2 - M_tauso1)*(1.0 + tanh(M_kso*(vold - M_uso)))/2.0
	tau_s = (1.0 - Hw)*M_taus1 + Hw*M_taus2
	tau_o = (1.0 - Ho)*M_tauo1 + Ho*M_tauo2
	vinf = conditional(ge(M_tetavm,vold), 1.0, 0.0)
	winf = (1.0 - Ho)*(1.0 - vold/M_tauwinf) + Ho*M_winfstar

	# Ionic currents
	Jfi = -r1old*Hv*(vold - M_tetav)*(M_uu - vold)/M_taufi
	Jso = (vold - M_uo)*(1.0 - Hw)/tau_o + Hw/tau_so
	Jsi = -Hw*r2old*r3old/M_tausi
	Iion = Jfi + Jso + Jsi

	# Right hand sides for gating variables
	ReactR1 = (1.0 - Hv)*(vinf - r1old)/tau_vm - Hv*r1old/M_tauvp
	ReactR2 = (1.0 - Hw)*(winf - r2old)/tau_wm - Hw*r2old/M_tauwp
	ReactR3 = ((1.0 + tanh(M_ks*(vold - M_us)))/2.0 - r3old)/tau_s

	# Total right hand side for full electric system
	Right = vold*w/dt*dx + r1old*s1/dt*dx + r2old*s2/dt*dx + r3old*s3/dt*dx \
	+ (-Iion + Istim(t))*w*dx \
	+ ReactR1*s1*dx + ReactR2*s2*dx + ReactR3*s3*dx

	solve(Left == Right, Sol)
	v,r1,r2,r3 = Sol.split()


	# Solve mechanical equations
	if (inc % frequencyMech == 0) and (t >= tMech):
		solve(FF == 0, PupTa, J=JJ, \
          solver_parameters={'newton_solver':{'linear_solver':'lu',\
                                              'absolute_tolerance':5.0e-7,\
                                              'relative_tolerance':5.0e-7,\
                                              'maximum_iterations':30}})
		Pi,u,p,Ta = PupTa.split()
		Pi_out = project(Pi,TensorOut)
		assign(uold,u); assign(uuold,uold)

	# Save solutions
	if (inc % frequencySave == 0):
		v.rename("v","v"); fileO.write(v,t)
		r1.rename("r1","r1"); fileO.write(r1,t)
		r2.rename("r2","r2"); fileO.write(r2,t)
		r3.rename("r3","r3"); fileO.write(r3,t)
		if (t > tMech):
			Pi_out.rename("Pi","Pi"); fileO.write(Pi_out,t)
			u.rename("u","u"); fileO.write(u,t)
			p.rename("p","p"); fileO.write(p,t)
			Ta.rename("Ta","Ta"); fileO.write(Ta,t)

	assign(vold,v); assign(r1old,r1); assign(r2old,r2);assign(r3old,r3)
 	assign(Pi_out,Pi_out)

	t += dt; inc += 1


# end



