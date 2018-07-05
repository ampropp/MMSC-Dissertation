# Modified from PassiveMech_Mixed.py very minimally by Adrienne Propp in May 2018
# Modified because some commands from original were outdated
# Eliminated some old, commented-out code
# Active stress approach, 2D

from fenics import *
#import numpy

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
#parameters["form_compiler"]["quadrature_degree"] = 2

def subplus(u) :
    return conditional(ge(u, 0.0), u, 0.0)


L=2.0; nps = 40;
mesh = RectangleMesh(Point(0,0),Point(L,L),nps,nps)
#nn = FacetNormal(mesh)

fileO = XDMFFile(mesh.mpi_comm(), "out/2DHolzapfelThreeFields.xdmf")
fileO.parameters['rewrite_function_mesh']=False
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True

P0 = FiniteElement("DG", mesh.ufl_cell(), 0) # Pressure
P1v = VectorElement("CG", mesh.ufl_cell(), 1) # Vector element for u in H1
P0t = TensorElement("DG", mesh.ufl_cell(), 0, symmetry= True) # For stress

Mh = FunctionSpace(mesh, "CG", 1)

#Output of tensor without symmetry
TensorOut = TensorFunctionSpace(mesh, "DG", 0) # Tensor FE space

Hh = FunctionSpace(mesh,MixedElement([P0t,P1v,P0]))


Ta = Function(Mh) # Increasing over time, piecewise linear?

print " **************** Mech Dof = ", Hh.dim()

Pup     = Function(Hh)
dPup    = TrialFunction(Hh)
(Pi,u,p)    = split(Pup)
(PiT,uT,pT) = TestFunctions(Hh)

f0   = Constant((1,0))
s0   = Constant((0,-1))
n0   = cross(f0,s0)

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
Ta = Expression("Ta0*pow(sin(DOLFIN_PI*t),2)", Ta0=Ta0, t=0.0, degree=3)

# ******** Mechanical entities ************* #
ndim = u.geometric_dimension()
print " **************** Geometric Dim = ", ndim

I = Identity(ndim); F = I + grad(u); F = variable(F)
C = F.T*F; B = F*F.T; J = det(F); invF = inv(F)

I1 = tr(C); I8_fs = inner(f0, C*s0) 
I4_f = inner(f0, C*f0); I4_s = inner(s0, C*s0)

k = Constant(0.3)

# Passive Cauchy stress tensor 
CPassive = a*exp(b*subplus(I1-ndim))*B \
           +2*a_f*(I4_f-1)*exp(b_f*subplus(I4_f-1)**2)*outer(F*f0,F*f0) \
           +2*a_s*(I4_s-1)*exp(b_s*subplus(I4_s-1)**2)*outer(F*s0,F*s0) \
           +a_fs*I8_fs*exp(b_fs*subplus(I8_fs)**2)*(outer(F*f0,F*s0) \
                                                    + outer(F*s0,F*f0)) # where does this last term come from?
            # also note I added the 2 in the fourth line
            # also what about factor of Jinv that should be with CPassive

CActive = Ta / I4_f * outer(F*f0,F*f0) + k * Ta *I4_f / I4_s * outer(F*s0,F*s0) 
+ k * Ta * I4_f / I8_fs * (outer(F*f0,F*s0) + outer(F*s0,F*f0))
    
TotalCauchy = CPassive + CActive
calG =  J * TotalCauchy * invF.T 

FF = inner(Pi - calG + p*J*I, PiT) * dx \
     + inner(Pi, grad(uT)*invF.T) * dx \
     + dot(J*eta*invF.T*u,uT) * ds \
     + pT*(J-1)*dx

JJ = derivative(FF, Pup, dPup)
    
# ********* Time loop ************* #
t = 0.0; dt = 0.1; Tfinal =3.0
while (t <= Tfinal):

    print "t=", t

    Ta.t = t
   
    
    solve(FF == 0, Pup, J=JJ, \
          solver_parameters={'newton_solver':{'linear_solver':'lu',\
                                              'absolute_tolerance':5.0e-4,\
                                              'relative_tolerance':5.0e-4}})
    # This is a projection step with the Kirchhoff stress
    Pi,u,p=Pup.split()
    Pi_out = project(Pi,TensorOut)

    Pi_out.rename("Pi","Pi"); fileO.write(Pi_out,t)
    u.rename("u","u"); fileO.write(u,t)
    p.rename("p","p"); fileO.write(p,t)
    t +=dt; 
