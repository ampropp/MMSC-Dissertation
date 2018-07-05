# Active strain approach, from Ricardo - 3D

from fenics import *
import numpy
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2

def varAngle(s):
    return ((thetaEpi - thetaEndo) * s + thetaEndo)/ 180.0 * pi

def normalizeAndProj_vec(u):
    return project(u/sqrt(dot(u,u)),FiberSpace)

def normalize_vec(u):
    return u/sqrt(dot(u,u))

def subplus(u) :
    return conditional(ge(u, 0.0), u, 0.0)
 
mesh = Mesh("meshes/ellipsoid.xml")
bdry = MeshFunction("size_t", mesh, "meshes/ellipsoid_facet_region.xml") 
#mesh = Mesh("meshes/bigEllipCoarse.xml")
#bdry = MeshFunction("size_t", mesh, "meshes/bigEllipCoarse_facet_region.xml")
ds = Measure("ds", subdomain_data=bdry) #epi:91, base:92, endo:93
nn = FacetNormal(mesh)

fileO = XDMFFile(mesh.mpi_comm(), "out/Ventricle-Coarse.xdmf")
fileO.parameters['rewrite_function_mesh']=False
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True

P1 = FiniteElement("CG", mesh.ufl_cell(), 1) # Pressure
P2v = VectorElement("CG", mesh.ufl_cell(), 2) # Elements for u?
FiberSpace = VectorFunctionSpace(mesh,"CG",1)
    
Mh = FunctionSpace(mesh, "CG", 1)
Hh = FunctionSpace(mesh,P2v*P1)

print " **************** Mech Dof = ", Hh.dim()
up      = Function(Hh)
dup     = TrialFunction(Hh)
(u,p)   = split(up)
(uT,pT) = TestFunctions(Hh)

f0   = Function(FiberSpace)
s0   = Function(FiberSpace)
n0   = Function(FiberSpace)
phih = Function(Mh)


# ******* Generate fibers and sheets  PRIMAL!! **********
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

s0 = normalizeAndProj_vec(grad(phih))

k0 = Constant((0.0,1.0,0.0))
kp = normalize_vec(k0 - dot(k0,s0) * s0)
f0flat = cross(s0, kp)

thetaEpi = Constant(-50.0)
thetaEndo = Constant(60.0)
f0 = normalizeAndProj_vec(f0flat*cos(varAngle(phih)) \
                          + cross(s0,f0flat)*sin(varAngle(phih)) \
                          + s0 * dot(s0,f0flat)*(1.0-cos(varAngle(phih))))

n0 = cross(f0,s0)


# ********* Mechanical parameters ********* #
a    = Constant(2.3621) # KPa
b    = Constant(10.810)
a_f  = Constant(1.16037) # KPa
b_f  = Constant(14.154)
a_s  = Constant(3.7245)
b_s  = Constant(5.1645)
a_fs = Constant(4.0108) # Pa, already scaled
b_fs = Constant(11.300)

mu   = Constant(4.0)

robinbot = Constant(0.06)
robintop = Constant(0.1)

ytop = Constant(2.5)
ybot = Constant(-4.3)
totl  = Constant(ytop - ybot)

robin = Expression("robinbot*(ytop-x[1])/totl + robintop*(x[1]-ybot)/totl", \
                   totl = totl, ytop=ytop, ybot = ybot, robintop=robintop, \
                   robinbot=robinbot, degree=3)

inten  = Constant(-0.15)
p0     = Constant(0.05)
#pow(sin(0.5*x[1]),2)*
gm = Expression("inten*pow(sin(DOLFIN_PI*t),2)", inten=inten, t=0.0, degree=3)

pEndo = Expression("p0*pow(sin(DOLFIN_PI*t),2)", p0=p0, t=0.0, degree=3)

bcuBa = DirichletBC(Hh.sub(0).sub(1), Constant(0.0), bdry, 92)
bcMec = bcuBa

# ******** Mechanical entities ************* #
I = Identity(3); F = I + grad(u); F = variable(F)
C = F.T*F; B = F*F.T; J = det(F); invF = inv(F)

I1 = tr(C); I8_fs = inner(f0, C*s0) 
I4_f = inner(f0, C*f0); I4_s = inner(s0, C*s0)

propS = Constant(3.0)

def P(gamma,u):

    gammaN = propS*(1.0-phih)*gamma+ phih*(pow(1.0-gamma,-0.5)-1.0)
    # or simply gammaN = propS * gamma
    gammaS = pow((1.0+gamma)*(1.0+gammaN),-1.0)-1.0
    F_a = I+gamma*outer(f0,f0)+gammaS*outer(s0,s0)+gammaN*outer(n0,n0) 
    invFa = inv(F_a)
    F_e = variable(F * invFa); C_e = F_e.T*F_e; B_e = F_e*F_e.T
    I1e = tr(B_e); I8_fse = inner(f0, C_e*s0) 
      # why is it tr(B_e)? actually gives the same as tr(C_e)
    I4_fe = inner(f0, C_e*f0); I4_se = inner(s0, C_e*s0)
    
    #sNHActiveStrain = mu * B_e; Cauchy = sNHActiveStrain - p*I
    
    sHOActiveStrain = a*exp(b*subplus(I1e-3))*B_e \
                      +2*a_f*(I4_fe-1)*exp(b_f*subplus(I4_fe-1)**2)*outer(F*f0,F*f0) \
                      +2*a_s*(I4_se-1)*exp(b_s*subplus(I4_se-1)**2)*outer(F*s0,F*s0) \
                      + a_fs*I8_fse*exp(b_fs*subplus(I8_fse)**2)*(outer(F*f0,F*s0) \
                                                                  + outer(F*s0,F*f0))

    Cauchy = sHOActiveStrain - p*I
    return J * Cauchy * invF.T 


gamma = Function(Mh)
gamma = gm

FF = inner(P(gamma,u), grad(uT)) * dx + inner(pT,J-1)*dx \
     - dot(J*pEndo*invF.T*nn,uT) * ds(93) \
     + dot(J*robin*invF.T*u,uT) * ds(91)

JJ = derivative(FF, up, dup)
    
# ********* Time loop ************* #
t = 0.0; dt = 0.1; Tfinal =1.5
while (t <= Tfinal):

    print "t=", t

    pEndo.t = t; gm.t = t; gamma = project(gm,Mh) 
   
    
    solve(FF == 0, up, J=JJ, bcs = bcMec, \
          solver_parameters={'newton_solver':{'linear_solver':'lu',\
                                              'absolute_tolerance':5.0e-4,\
                                              'relative_tolerance':5.0e-4}})
    
    u,p=up.split()

    gamma.rename("gam","gam"); fileO.write(gamma,t)
    u.rename("u","u"); fileO.write(u,t)
    p.rename("p","p"); fileO.write(p,t)
    f0.rename("f0","f0"); fileO.write(f0,t)
    s0.rename("s0","s0"); fileO.write(s0,t)
    t +=dt; 
