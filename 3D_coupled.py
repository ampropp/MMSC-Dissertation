
##########################################################################################
## Written and modified by Ricardo Ruiz Baier and Adrienne Propp                        ##
##                                                                                      ##
## Code to simulate cardiac electromechanics on an idealized 3D ventricular gemoetry    ##
## Uses an active strain formulation, coupled with Bueno-Orovio et al. (2008) model     ##
## of human ventricular action potential                                                ##
##                                                                                      ##
## This code is written for the hyperelastic case, but can be modified to the           ##
## viscoelastic case by uncommenting the code in the relevant section.                  ##
##########################################################################################

# ********* Dependencies ********* #
# Note: this code uses FEniCS to solve using the finite element method
from fenics import *
from mshr import *
import numpy
import time


# ********* Setup ********* #
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2


# ********* Problem-specific parameters ********* #
t = 0.0; dt = 0.25; Tfinal = 600  # change as necessary; time in ms
frequencyMech = 10.0; Dt = dt*frequencyMech # mechanics will be solved on coarser timestep than electrophysiology


# ********* Load mesh ********* #
mesh = Mesh("meshes/bigEllipCoarse.xml")
bdry = MeshFunction("size_t", mesh, "meshes/bigEllipCoarse_facet_region.xml")
ds = Measure("ds", subdomain_data=bdry) #epi:91, base:92, endo:93
nn = FacetNormal(mesh)
he = FacetArea(mesh)


# ********* Define file to save results ********* #
fileO = XDMFFile(mesh.mpi_comm(), "out/3D_coupled.xdmf")
fileO.parameters['rewrite_function_mesh']=False # mesh is consistent throughout our time interval
fileO.parameters["functions_share_mesh"] = True # each function at a given time shares the same mesh
fileO.parameters["flush_output"] = True # dataset can be saved after each timestep
# Note: these parameters help reduce filesize


# ********* User-defined functions ********* #
def varAngle(s):
    return ((thetaEpi - thetaEndo) * s + thetaEndo)/ 180.0 * pi

def normalizeAndProj_vec(u):
    return project(u/sqrt(dot(u,u)),FiberSpace)

def normalize_vec(u):
    return u/sqrt(dot(u,u))

def subplus(u) :
    return conditional(ge(u, 0.0), u, 0.0)


# ********* Define elements ********* #
P1 = FiniteElement("CG", mesh.ufl_cell(), 1) # Pressure, ionic quantities
P1v = VectorElement("CG", mesh.ufl_cell(), 1) #
P2v = VectorElement("CG", mesh.ufl_cell(), 2) # Displacement (vector)


# ********* Define function spaces ********* #
Mh = FunctionSpace(mesh, P1)
Vh = FunctionSpace(mesh,P1v)
Hh = FunctionSpace(mesh,P2v*P1) # Mechanical system
Nh = FunctionSpace(mesh, MixedElement([P1,P1,P1,P1])) # Electrical system
TensorOut = TensorFunctionSpace(mesh, "DG", 0) # Tensor FE space
FiberSpace = VectorFunctionSpace(mesh,"CG",1) # Fiber FE space

print " **************** Mech Dof = ", Hh.dim()


# ********* Electrical functions ********* #
Sol = Function(Nh)
(v,r1,r2,r3) = TrialFunctions(Nh) # v is transmemb potential (voltage), r are ionic quantities
(w,s1,s2,s3) = TestFunctions(Nh)
# Note: in Bueno-Orovio et al. (2008), these are defined as (u,v,w,s)
# Note: r3 is a proxy for calcium


# ********* Mechanical functions ********* #
up = Function(Hh)
dup = TrialFunction(Hh)
(u,p) = split(up) # (displacement, pressure)
(uT,pT) = TestFunctions(Hh)

ndim = u.geometric_dimension(); print " **************** Geometric Dim = ", ndim


# ******* Generate fibers and sheets (PRIMAL) **********
f0   = Function(FiberSpace)
s0   = Function(FiberSpace)
n0   = Function(FiberSpace)
phih = Function(Mh)

# Calculate fiber position within ventricular wall by solving a mixed Poisson problem
phif = TrialFunction(Mh)
psif = TestFunction(Mh)
AAf = dot(grad(phif),grad(psif))*dx
ggf = Constant(0.0)
BBf = ggf * psif * dx
bcphiEn = DirichletBC(Mh, Constant(0.0), bdry, 93)
bcphiEp = DirichletBC(Mh, Constant(1.0), bdry, 91)
bcphi = [bcphiEp,bcphiEn]

solve(AAf == BBf, phih, bcphi, \
      solver_parameters={'linear_solver':'lu'})

# Calculate sheetlet directions
s0 = normalizeAndProj_vec(grad(phih))

k0 = Constant((0.0,1.0,0.0))
kp = normalize_vec(k0 - dot(k0,s0) * s0)
f0flat = cross(s0, kp)

# Limits of fiber rotation
thetaEpi = Constant(-50.0)
thetaEndo = Constant(60.0)

# Rotate fibers according to their position within ventricular wall
f0 = normalizeAndProj_vec(f0flat*cos(varAngle(phih)) \
                          + cross(s0,f0flat)*sin(varAngle(phih)) \
                          + s0 * dot(s0,f0flat)*(1.0-cos(varAngle(phih))))

n0 = cross(f0,s0)


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
uold = interpolate(Constant((0.0,0.0,0.0)),Vh)
uuold = interpolate(Constant((0.0,0.0,0.0)),Vh)
stabP = Constant(0.05)


# ********* Electrical parameters ********* #
diffScale = Constant(1.0e-3)  # Rescale time from s to ms

#Diffusion parameters
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
a    = Constant(2.3621) # KPa
b    = Constant(10.810)
a_f  = Constant(1.16037) # KPa
b_f  = Constant(14.154)
a_s  = Constant(3.7245)
b_s  = Constant(5.1645)
a_fs = Constant(4.0108) # Pa, already scaled
b_fs = Constant(11.300)

mu = Constant(4.0)


# ********* Electrical Stimuli ********* #
stim_t1   = 1.0
stim_t2   = 335.0
stim_dur1 = 3.0
stim_dur2 = 3.0
stim_amp  = 3.0
waveS1 = Expression("amp*(x[1] < -4.1)", amp = stim_amp, degree=2) # planar wave
waveS2 = Expression("amp*(x[0] < 0.0 && x[1] < -3.75 && x[2] < 0.25)", amp = stim_amp, degree=2) # square wave in lower left octant

# Define external stimulus  to be added to reaction-diffusion equation
def Istim(t):
    if (stim_t1 <= t and t <= stim_t1 + stim_dur1):
        return waveS1
    if (stim_t2 <= t and t <= stim_t2 + stim_dur2):
        return waveS2
    else:
        return Constant(0.0)


# ********* Robin boundaries ********* #
robinbot = Constant(0.06)
robintop = Constant(0.1)

ytop = Constant(2.5)
ybot = Constant(-4.3)
totl  = Constant(ytop - ybot)

# Define spatially-dependent stiffness coefficient
robin = Expression("robinbot*(ytop-x[1])/totl + robintop*(x[1]-ybot)/totl", \
                   totl = totl, ytop=ytop, ybot = ybot, robintop=robintop, \
                   robinbot=robinbot, degree=3)

inten  = Constant(-0.15)
p0     = Constant(0.05)

gm = Expression("inten*pow(sin(DOLFIN_PI*t*1.0e-3),2)", inten=inten, t=0.0, degree=3)
pEndo = Expression("p0*pow(sin(DOLFIN_PI*t*1.0e-3),2)", p0=p0, t=0.0, degree=3)

bcuBa = DirichletBC(Hh.sub(0).sub(1), Constant(0.0), bdry, 92)
bcMec = bcuBa


# ******** Mechanical entities ************* #
I = Identity(ndim); F = I + grad(u); F = variable(F) # F is deformation gradient
C = F.T*F; B = F*F.T; J = det(F); invF = inv(F)  # C and B are Cauchy-Green deformation tensors
J = variable(J); C = variable(C)

# Define invariants
I1 = tr(C) # isometric invariant
I8_fs = inner(f0, C*s0); I4_f = inner(f0, C*f0); I4_s = inner(s0, C*s0) # anisotropic pseudoinvariants

k = Constant(0.3) # transverse fiber stress as a fraction of axial tension
propS = Constant(1.0)

def P(gamma,u):
    gammaN = propS*(1.0-phih)*gamma+ phih*(pow(1.0-gamma,-0.5)-1.0)
    # or simply gammaN = propS * gamma
    gammaS = pow((1.0+gamma)*(1.0+gammaN),-1.0)-1.0
    F_a = I+gamma*outer(f0,f0)+gammaS*outer(s0,s0)+gammaN*outer(n0,n0) # active strain component
    invFa = inv(F_a)
    F_e = variable(F * invFa); C_e = F_e.T*F_e; B_e = F_e*F_e.T # passive strain component
    I1e = tr(B_e); I8_fse = inner(f0, C_e*s0) # passive invariants
    I4_fe = inner(f0, C_e*f0); I4_se = inner(s0, C_e*s0) # passive invariants
    sHOActiveStrain = a*exp(b*subplus(I1e-3))*B_e \
                      +2*a_f*(I4_fe-1)*exp(b_f*subplus(I4_fe-1)**2)*outer(F*f0,F*f0) \
                      +2*a_s*(I4_se-1)*exp(b_s*subplus(I4_se-1)**2)*outer(F*s0,F*s0) \
                      + a_fs*I8_fse*exp(b_fs*subplus(I8_fse)**2)*(outer(F*f0,F*s0) \
                                                                  + outer(F*s0,F*f0))
    Cauchy = sHOActiveStrain - p*I # Cauchy stress tensor
    ###################
    # VISCOELASTICITY #
    ###################
    ## Uncomment the following code to modify the model from hyperelastic to viscoelastic
    #Fdot = grad(u-uold)/Dt
    #gradv = Fdot*inv(F)
    #d = 0.5*(gradv + gradv.T)
    #Bdot = gradv*B + B*gradv.T
    #Cdot = 2.0*F.T*d*F
    #Idot = tr(Cdot)
    #Visco =  gam*exp(beta*Idot)*Bdot
    #Cauchy = Cauchy + Visco
	
    return J * Cauchy * invF.T # pull-back to reference configuration


gamma = Function(Mh)
gamma = gm


# ********* Coupling functions ********* #
Pi = Function(TensorOut)
Pi = P(gamma,u)*F.T

def DTens(v,u,PI):
    self_diff = (D0 + D1*v)*J*inv(C)
    aniso = D0*J*outer(invF*f0,invF.T*f0)
    SAD = D2*invF*PI*invF.T
    return self_diff + aniso + SAD


# ******** Weak forms ************* #
Left = v/dt*w*dx \
   + inner(DTens(vold,uold,Pi)*grad(v),grad(w))*dx \
   + r1/dt*s1*dx + r2/dt*s2*dx + r3/dt*s3*dx

FF = inner(P(gamma,u), grad(uT)) * dx + inner(pT,J-1)*dx \
     - dot(J*pEndo*invF.T*nn,uT) * ds(93) \
     + dot(J*robin*invF.T*u,uT) * ds(91)

JJ = derivative(FF, up, dup)


# ********* Time loop ************* #
t = 0.0; tMech = stim_t1 + 10.0 # begin mechanics after spiral has formed
inc = 0; frequencySave = 10
while (t <= Tfinal):
	print "t=", t

	pEndo.t = t; gm.t = t; gamma = project(gm,Mh)

	LHS = assemble(Left)

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

	RHS = assemble(Right)
	solve(LHS, Sol.vector(), RHS)
	(v,r1,r2,r3) = Sol.split()


        # Solve mechanical equations
	if (inc % frequencyMech == 0) and (t >= tMech):
		solve(FF == 0, up, J=JJ, bcs = bcMec, \
          solver_parameters={'newton_solver':{'linear_solver':'lu',\
                                              'absolute_tolerance':5.0e-4,\
                                              'relative_tolerance':5.0e-4,\
                                              'maximum_iterations':50}})
		u,p = up.split()

	# Save solutions
	if (inc % frequencySave == 0):
		v.rename("v","v"); fileO.write(v,t)
		r1.rename("r1","r1"); fileO.write(r1,t)
		r2.rename("r2","r2"); fileO.write(r2,t)
		r3.rename("r3","r3"); fileO.write(r3,t)
		f0.rename("f0","f0"); fileO.write(f0,t)
		s0.rename("s0","s0"); fileO.write(s0,t)
		if (t >= tMech + frequencyMech):
			u.rename("u","u"); fileO.write(u,t)
			p.rename("p","p"); fileO.write(p,t)
			gamma.rename("gam","gam"); fileO.write(gamma,t)
			Pi.rename("Pi","Pi"); fileO.write(Pi,t)
	assign(vold,v); assign(r1old,r1); assign(r2old,r2);assign(r3old,r3)

	t += dt; inc += 1

# end





