# Transverse-Route Geometric Flow for Binary Quadratic Optimization  
## A First-Order Manifold Dynamics with Exact Collective-Flip Curvature

**Authors:** [to be inserted]

### Abstract

We introduce a continuous-time geometric formulation for quadratic unconstrained binary optimization (QUBO) and Ising ground-state search in which each binary variable is temporarily embedded on a continuous manifold containing multiple routes between its two eventual Ising states. Unlike simulated bifurcation, the dynamics contain no momentum variables, inertial terms, nonlinear oscillators, or Hamiltonian time evolution. Unlike conventional XY relaxations, the transverse degrees of freedom do not simply constitute additional components of the Ising interaction. Instead, the original Ising interaction acts along a distinguished longitudinal coordinate, while auxiliary interaction channels act only away from the binary manifold and vanish identically on every binary configuration.

For the minimal planar construction, each spin is represented by an angle \(\theta_i\), with longitudinal coordinate \(x_i=\cos\theta_i\) and transverse coordinate \(y_i=\sin\theta_i\). We introduce the route coordinate \(r_i=x_i y_i\), and define an auxiliary interaction \(r^\top J r/2\) using the same Ising coupling matrix \(J\). A higher-order route coordinate \(p_i=y_i^3\) can additionally distinguish the two transverse routes in regions where \(r_i\) vanishes, without modifying the local quadratic stability of binary configurations. The resulting system evolves by first-order gradient flow on the \(N\)-torus.

The principal analytical result is an exact connection between transverse curvature and discrete collective spin flips. Around an arbitrary binary state \(s\), the transverse Hessian at zero confinement is

\[
M(s)
=
SJS-\operatorname{diag}\!\left[S(Js+h)\right],
\qquad
S=\operatorname{diag}(s).
\]

For any subset \(C\) of spins, represented by its binary indicator \(q_C\), the exact energy change produced by flipping all spins in \(C\) satisfies

\[
q_C^\top M(s)q_C
=
\frac{1}{2}
\left[
E(s^{(C)})-E(s)
\right].
\]

Consequently, every strictly suboptimal binary configuration is a saddle of the unconstrained geometric energy: if any collective spin flip lowers the original Ising energy, the corresponding transverse direction has negative curvature. A time-dependent transverse-confinement parameter can subsequently deform the continuous manifold toward the binary submanifold, yielding a dissipative continuation algorithm.

All pairwise interactions reduce to multiplication by the original coupling matrix \(J\). For multiple solver trajectories, the longitudinal and route coordinates can be concatenated and updated with a single dense GEMM or sparse SpMM operation, making the method naturally compatible with massively parallel GPU execution. We derive the governing equations, establish binary faithfulness and Lyapunov properties, analyze the local stability spectrum, and specify a benchmark protocol for evaluating whether the additional route geometry improves practical escape from metastable QUBO states.

**Keywords:** QUBO, Ising optimization, continuous relaxation, dynamical systems, manifold optimization, GPU computing, spin glass, geometric flow, combinatorial optimization.

---

# 1. Introduction

Quadratic unconstrained binary optimization is the problem of minimizing a quadratic function over binary variables. A large class of discrete optimization problems can be mapped to QUBO or, equivalently, to an Ising Hamiltonian with polynomial overhead [1]. This common representation has motivated a broad family of physics-inspired optimization methods, including simulated annealing, quantum annealing, coherent Ising machines, oscillator Ising machines, mean-field methods, digital annealers, and simulated bifurcation. The broader landscape of Ising machines and their computational principles has been reviewed extensively [2]. citeturn332191search0turn332191search1

Simulated annealing searches the discrete configuration space through stochastic local transitions controlled by a temperature schedule [3]. Continuous mean-field approaches instead replace binary variables with continuous magnetizations and progressively drive those variables toward binary values [4]. Other continuous approaches introduce explicit penalty functionals whose minima approach the Boolean hypercube as a control parameter is varied [5]. citeturn332191search5turn459052search5turn821917search3

Dynamical Ising machines extend this idea by associating computational variables with physical or pseudo-physical dynamical states. Simulated bifurcation, for example, numerically integrates classical nonlinear Hamiltonian systems possessing position- and momentum-like variables and exploits adiabatic bifurcation and chaotic dynamics to search the Ising landscape [6]. Its simultaneous state updates are particularly attractive for parallel digital hardware. citeturn459052search0

Continuous angular and vector-spin representations constitute another established direction. XY and oscillator models allow Ising variables to explore continuous phases before binary reconstruction, and continuous vector relaxations are related to classical convex and semidefinite relaxations of discrete optimization. Recent work has explicitly demonstrated that additional spin dimensions and subsequent XY-to-Ising or higher-dimensional-to-binary collapse can improve optimization dynamics [7–9]. citeturn459052search1turn328433search0turn328433search7

The present work begins from a more specific geometric question.

Consider a binary variable currently associated with \(s_i=+1\). In a one-dimensional continuous relaxation, changing its state requires traversing essentially one scalar direction,

\[
+1\longrightarrow 0\longrightarrow -1.
\]

If that transition encounters a high effective barrier, the only available mechanisms are typically a sufficiently large deterministic force, inertial overshoot, stochastic noise, or a collective change of the surrounding variables.

Instead, suppose the binary variable is embedded in a larger manifold containing several geometrically distinct trajectories between the same two terminal states. These trajectories are useful only if they are dynamically inequivalent. Merely embedding a scalar variable on a circle does not accomplish this: if the objective depends only on \(\cos\theta_i\), the upper and lower semicircles are energetically equivalent and reduce mathematically to a scalar relaxation.

We therefore seek a construction satisfying four requirements.

First, the two eventual binary states must exactly reproduce the original Ising objective. Second, transverse coordinates must affect interactions during the continuous evolution so that distinct geometric routes correspond to distinct effective interaction histories. Third, those transverse interactions should be derived directly from the original \(J\) and \(h\), rather than requiring an independently designed optimization problem. Fourth, the complete dynamics should consist primarily of matrix multiplication and elementwise operations, making synchronous GPU execution natural.

The central construction developed below satisfies these conditions and yields an unexpected additional property: the local transverse curvature around a binary state can be made to encode the exact energy changes of collective Ising spin flips.

This produces a different computational picture from a conventional bifurcation solver. Rather than giving each spin one deforming double-well potential, the collective configuration continuously reshapes a family of local angular potentials. Other variables therefore influence a given variable in two ways: directly through the conventional Ising field and indirectly by changing which transverse trajectory is locally favorable.

The contribution of this paper is specifically this route-dependent energy construction and its associated curvature identity. The generic concepts of vector spins, continuous spin dimensions, XY-to-Ising transitions, and dimensional collapse are established prior art and are not claimed here as new principles. citeturn459052search1turn328433search0turn328433search7

---

# 2. Binary quadratic optimization

## 2.1 QUBO representation

Consider

\[
\min_{z\in\{0,1\}^N}
C(z)
=
z^\top Qz,
\]

where \(Q\) can be taken symmetric without loss of generality.

Introduce Ising variables

\[
s_i=2z_i-1,
\qquad
z_i=\frac{1+s_i}{2}.
\]

Then

\[
C(z)
=
\frac14
(s+\mathbf 1)^\top
Q
(s+\mathbf 1),
\]

which can be written, up to an additive constant, as

\[
E(s)
=
\frac12s^\top Js+h^\top s,
\qquad
s_i\in\{-1,+1\}.
\]

We work directly with this Ising representation.

The coupling matrix is assumed symmetric,

\[
J=J^\top.
\]

Its diagonal can be set to zero because

\[
J_{ii}s_i^2=J_{ii}
\]

is constant on the Ising domain.

Thus throughout the remainder of the paper,

\[
\boxed{
E(s)
=
\frac12s^\top Js+h^\top s,
\qquad
J=J^\top,
\qquad
J_{ii}=0.
}
\]

---

# 3. Geometric state space

## 3.1 From an Ising spin to a circle

Replace each binary variable by an angular coordinate

\[
\theta_i\in S^1.
\]

Define

\[
x_i=\cos\theta_i,
\qquad
y_i=\sin\theta_i.
\]

Therefore

\[
x_i^2+y_i^2=1.
\]

The longitudinal coordinate \(x_i\) is the continuous Ising coordinate. The two binary states are

\[
\theta_i=0
\quad\Longleftrightarrow\quad
(x_i,y_i)=(+1,0),
\]

and

\[
\theta_i=\pi
\quad\Longleftrightarrow\quad
(x_i,y_i)=(-1,0).
\]

The complete solver state is therefore

\[
\theta
\in
(S^1)^N,
\]

the \(N\)-torus.

No mechanical interpretation of \(\theta_i\) is required. It is an intrinsic coordinate of the optimization manifold, not the physical phase of an oscillator.

---

## 3.2 Why projection alone is insufficient

The simplest continuous extension would be

\[
E_x(\theta)
=
\frac12x^\top Jx+h^\top x.
\]

However, this energy depends only on

\[
x_i=\cos\theta_i.
\]

Consequently,

\[
E_x(+\theta_i)=E_x(-\theta_i).
\]

The upper and lower semicircles are therefore exactly equivalent.

Indeed, any gradient dynamics derived exclusively from \(x_i\) can be rewritten entirely in terms of the scalar variables \(x_i\in[-1,1]\). The nominally two-dimensional representation introduces no genuinely new optimization degree of freedom.

The transverse coordinate must therefore enter the interaction law.

---

# 4. Route-dependent interaction coordinates

## 4.1 Endpoint-oriented route coordinate

Define

\[
\boxed{
r_i=x_i y_i.
}
\]

Equivalently,

\[
r_i
=
\frac12\sin 2\theta_i.
\]

This coordinate has three useful properties.

First,

\[
r_i=0
\]

at both binary states.

Second, near either binary endpoint it behaves as an endpoint-oriented transverse coordinate. If the current Ising state is \(s_i=\pm1\), then locally

\[
x_i
=
s_i\sqrt{1-y_i^2}
=
s_i
\left(
1-\frac12y_i^2
\right)
+
O(y_i^4),
\]

and therefore

\[
r_i
=
s_i y_i+O(y_i^3).
\]

Third, \(r_i\) changes sign when the transverse direction \(y_i\) changes sign. Consequently, pairwise interactions involving \(r_i r_j\) can distinguish different collective routes through the manifold.

---

## 4.2 Higher-order route separation

The coordinate \(r_i=x_i y_i\) necessarily vanishes whenever \(x_i=0\). Thus, at the precise longitudinal midpoint,

\[
(x_i,y_i)=(0,\pm1),
\]

the \(r\)-channel alone cannot distinguish the two routes.

This is not required for the local curvature result developed below, but it can create an undesirable route-blind region far from the binary manifold.

We therefore introduce an optional higher-order route coordinate

\[
\boxed{
p_i=y_i^3.
}
\]

This term also vanishes at both binary endpoints, but

\[
p_i=+1
\]

at the upper midpoint and

\[
p_i=-1
\]

at the lower midpoint.

Near a binary state,

\[
p_i=O(y_i^3),
\]

so interactions involving \(p\) contribute only at sixth order to the energy and do not alter the quadratic transverse stability derived later.

Thus the \(r\)-channel controls local collective-flip curvature, whereas the \(p\)-channel can preserve route distinction deeper in the transverse manifold.

---

# 5. Geometric energy functional

We define the continuous energy

\[
\boxed{
\mathcal F_{\kappa,\gamma}(\theta)
=
\frac12x^\top Jx
+
h^\top x
+
\frac12r^\top Jr
+
\frac{\gamma}{2}p^\top Jp
+
\frac{\kappa}{2}y^\top y.
}
\tag{1}
\]

Here

\[
x=\cos\theta,
\qquad
y=\sin\theta,
\qquad
r=x\odot y,
\qquad
p=y^{\odot3},
\]

and \(\odot\) denotes elementwise multiplication.

The parameter

\[
\gamma\ge0
\]

controls the higher-order route-separation channel.

The parameter

\[
\kappa
\]

controls transverse confinement.

For

\[
\kappa>0,
\]

states with \(y_i\neq0\) are penalized, favoring the binary axis.

For

\[
\kappa<0,
\]

transverse displacement is energetically encouraged.

The solver can therefore use a continuation schedule

\[
\kappa(t):
\qquad
\kappa_0
\rightarrow
\kappa_f,
\]

typically with

\[
\kappa_0\le0,
\qquad
\kappa_f>0.
\]

The early system has substantial transverse freedom. The late system increasingly confines every variable toward

\[
y_i=0,
\]

and hence toward

\[
x_i=\pm1.
\]

No change in manifold topology is mathematically required. The topology remains \((S^1)^N\); what changes is the energy geometry on that manifold.

---

# 6. Exact preservation of the Ising problem

### Proposition 1 — Binary faithfulness

For every binary state

\[
s\in\{-1,+1\}^N,
\]

let \(\theta_s\) denote the corresponding point of the manifold with

\[
x=s,
\qquad
y=0.
\]

Then

\[
\boxed{
\mathcal F_{\kappa,\gamma}(\theta_s)=E(s).
}
\]

### Proof

At a binary state,

\[
y=0.
\]

Therefore

\[
r=x\odot y=0
\]

and

\[
p=y^{\odot3}=0.
\]

The confinement term also vanishes. Equation (1) consequently reduces to

\[
\mathcal F_{\kappa,\gamma}
=
\frac12s^\top Js+h^\top s
=
E(s).
\]

\(\square\)

Thus neither the route interaction nor the deformation schedule changes the energy ordering of any two binary configurations.

This property distinguishes the auxiliary transverse terms from a conventional approximation of the objective: they alter only the continuous paths between binary configurations.

---

# 7. Collective state-dependent local potentials

Equation (1) has a useful local interpretation.

Because \(J_{ii}=0\), define the three collective fields

\[
A_i
=
h_i+\sum_jJ_{ij}x_j,
\]

\[
B_i
=
\sum_jJ_{ij}r_j,
\]

and

\[
C_i
=
\sum_jJ_{ij}p_j.
\]

For fixed states of all other variables, the energy dependence on \(\theta_i\) is, up to an additive constant,

\[
\boxed{
V_i(\theta_i)
=
A_i\cos\theta_i
+
B_i\sin\theta_i\cos\theta_i
+
\gamma C_i\sin^3\theta_i
+
\frac{\kappa}{2}\sin^2\theta_i.
}
\tag{2}
\]

This expression provides the physical intuition behind the algorithm.

The field \(A_i\) is the conventional longitudinal Ising field. It directly pushes the variable toward one Ising orientation or the other.

The field \(B_i\) is generated by the transverse route states of the other variables. It changes the shape and angular asymmetry of the local potential.

The field \(C_i\) provides a higher-order route-dependent deformation that remains active near the transverse midpoint.

Consequently, each variable experiences not one predetermined deforming double well but a continuously evolving local potential

\[
V_i(\theta_i;t)
\]

whose shape is itself generated by the collective configuration.

The surrounding variables therefore exert both a direct force and an indirect landscape deformation.

---

# 8. Pure dissipative geometric dynamics

No momentum variable is introduced.

No oscillator equation is assumed.

No second-order equation of motion is required.

The system follows first-order gradient flow,

\[
\boxed{
\dot\theta_i
=
-\mu
\frac{\partial\mathcal F}{\partial\theta_i},
}
\tag{3}
\]

where

\[
\mu>0
\]

is a mobility coefficient.

Define

\[
A=Jx+h,
\qquad
B=Jr,
\qquad
C=Jp.
\]

Using

\[
\frac{dx_i}{d\theta_i}=-y_i,
\]

\[
\frac{dr_i}{d\theta_i}
=
x_i^2-y_i^2,
\]

and

\[
\frac{dp_i}{d\theta_i}
=
3x_i y_i^2,
\]

we obtain

\[
\boxed{
\dot\theta
=
\mu
\left[
y\odot A
-
(x^{\odot2}-y^{\odot2})\odot B
-
3\gamma
x\odot y^{\odot2}\odot C
-
\kappa x\odot y
\right].
}
\tag{4}
\]

Equation (4) is the proposed solver dynamics.

The evolution is entirely geometric: a dissipative flow of points on a product of circles under a time-dependent energy landscape.

---

# 9. Lyapunov property

### Proposition 2 — Monotonic energy decay at fixed deformation

For fixed \(\kappa\) and \(\gamma\),

\[
\boxed{
\frac{d\mathcal F}{dt}
=
-\mu
\|\nabla_\theta\mathcal F\|_2^2
\le0.
}
\]

### Proof

From Eq. (3),

\[
\dot\theta
=
-\mu\nabla_\theta\mathcal F.
\]

Therefore

\[
\frac{d\mathcal F}{dt}
=
\nabla_\theta\mathcal F^\top\dot\theta
=
-\mu
\nabla_\theta\mathcal F^\top
\nabla_\theta\mathcal F
\le0.
\]

\(\square\)

For a time-dependent confinement schedule,

\[
\mathcal F=\mathcal F(\theta,\kappa(t)),
\]

and therefore

\[
\frac{d\mathcal F}{dt}
=
-\mu
\|\nabla_\theta\mathcal F\|^2
+
\frac{\dot\kappa}{2}
\|y\|^2.
\]

Thus the system is dissipative relative to the instantaneous landscape, while the external deformation schedule can inject or remove energy by changing that landscape.

This is a continuation process rather than Hamiltonian motion.

---

# 10. Local geometry around an Ising configuration

The central property of the construction appears when Eq. (1) is expanded around a binary configuration.

Let

\[
s\in\{-1,+1\}^N
\]

be an arbitrary Ising state and define

\[
S=\operatorname{diag}(s).
\]

Define the Ising local field

\[
H=Js+h
\]

and the signed local field

\[
\ell
=
S H
=
s\odot(Js+h).
\]

Near the binary state, use \(y\) as the local transverse coordinate. Since

\[
x_i
=
s_i\sqrt{1-y_i^2},
\]

we have

\[
x_i
=
s_i
-
\frac12s_i y_i^2
+
O(y_i^4).
\]

Furthermore,

\[
r_i
=
x_i y_i
=
s_i y_i
+
O(y_i^3),
\]

whereas

\[
p_i=O(y_i^3).
\]

Substituting into Eq. (1) yields

\[
\boxed{
\mathcal F(s,y)
=
E(s)
+
\frac12
y^\top
M_\kappa(s)
y
+
O(\|y\|^4),
}
\tag{5}
\]

with

\[
\boxed{
M_\kappa(s)
=
SJS
-
\operatorname{diag}(\ell)
+
\kappa I.
}
\tag{6}
\]

The higher-order \(p\)-channel does not appear in Eq. (6).

Thus the transverse stability of a binary configuration is completely determined, to quadratic order, by the original Ising couplings and fields.

---

# 11. Exact connection to collective spin flips

Consider a subset

\[
C\subseteq\{1,\ldots,N\}.
\]

Let

\[
q_C\in\{0,1\}^N
\]

be its indicator,

\[
(q_C)_i
=
\begin{cases}
1,&i\in C,\\
0,&i\notin C.
\end{cases}
\]

Flipping every spin in \(C\) produces

\[
s^{(C)}
=
s-2Sq_C.
\]

The exact Ising energy change is

\[
\Delta E_C
=
E(s^{(C)})-E(s).
\]

Expanding exactly,

\[
\Delta E_C
=
2
\left[
q_C^\top SJSq_C
-
\ell^\top q_C
\right].
\tag{7}
\]

At zero confinement,

\[
M_0(s)
=
SJS-\operatorname{diag}(\ell).
\]

Because

\[
q_i^2=q_i
\]

for an indicator vector,

\[
q_C^\top
\operatorname{diag}(\ell)
q_C
=
\ell^\top q_C.
\]

Combining with Eq. (7) gives the central identity.

### Theorem 1 — Collective-flip curvature identity

For every binary state \(s\) and every spin subset \(C\),

\[
\boxed{
q_C^\top M_0(s)q_C
=
\frac12
\left[
E(s^{(C)})-E(s)
\right].
}
\tag{8}
\]

The second-order geometric curvature in the transverse direction associated with a coordinated subset excursion is therefore exactly proportional to the discrete energy change of flipping that subset.

This is not an approximation to the discrete flip cost.

It is an exact identity.

---

# 12. Suboptimal binary states are necessarily saddles

Theorem 1 immediately produces a stronger result.

### Theorem 2 — Absence of strictly suboptimal binary local minima at zero confinement

Let \(s\) be a binary configuration that is not globally optimal. Then \(s\) is not a local minimum of \(\mathcal F_{0,\gamma}\) on the continuous manifold.

More specifically, \(M_0(s)\) possesses a negative-curvature direction.

### Proof

Because \(s\) is not globally optimal, there exists a binary state \(s'\) such that

\[
E(s')<E(s).
\]

Let \(C\) be the set of spins on which \(s'\) differs from \(s\). Then

\[
s'=s^{(C)}.
\]

Therefore

\[
\Delta E_C
=
E(s')-E(s)
<0.
\]

From Theorem 1,

\[
q_C^\top M_0(s)q_C
=
\frac12\Delta E_C
<0.
\]

Hence \(M_0(s)\) is not positive semidefinite.

The binary point is therefore a saddle of the continuous energy.

\(\square\)

This result provides the main theoretical motivation for the transverse construction.

A conventional discrete local search can be trapped at a configuration for which every single-spin flip increases the objective.

The present continuous landscape does not restrict its local geometry to single-spin moves. If some coordinated subset of spins yields a lower Ising state, that subset generates a negative-curvature transverse direction.

Thus a discrete local minimum need not be a continuous local minimum.

This does **not** imply global convergence. The descending transverse trajectory may terminate at a nonbinary stationary point, and the continuous energy can possess nonbinary local minima and saddles. The theorem only establishes that strictly suboptimal binary configurations cannot be stable local minima of the zero-confinement continuous energy.

---

# 13. Stability spectrum and geometric annealing

For finite confinement,

\[
M_\kappa(s)
=
M_0(s)+\kappa I.
\]

Therefore every eigenvalue is shifted uniformly,

\[
\lambda_j(M_\kappa)
=
\lambda_j(M_0)+\kappa.
\]

Define the binary stability threshold

\[
\boxed{
\kappa_c(s)
=
-\lambda_{\min}[M_0(s)].
}
\tag{9}
\]

The binary state is linearly stable to transverse perturbations when

\[
\kappa>\kappa_c(s).
\]

It is unstable when

\[
\kappa<\kappa_c(s).
\]

Consequently, increasing \(\kappa\) continuously changes which binary configurations are dynamically capable of trapping the flow.

This suggests a geometric continuation schedule.

At early times, choose

\[
\kappa\lesssim0.
\]

Binary configurations are weakly confined or actively destabilized, and the state can explore transverse routes.

As \(\kappa\) increases, transverse excursions become progressively more expensive.

Eventually,

\[
\kappa\gg0,
\]

and the system collapses toward the binary axis.

This produces the conceptual sequence

\[
\text{transverse exploration}
\rightarrow
\text{selective stabilization}
\rightarrow
\text{binary confinement}.
\]

No mechanical bifurcation or inertial overshoot is required.

---

# 14. Lower bound on the stabilization of a suboptimal state

The collective-flip identity provides a quantitative relation between objective improvement and the confinement required to stabilize an inferior configuration.

For any improving subset \(C\),

\[
\Delta E_C<0.
\]

The Rayleigh quotient gives

\[
\lambda_{\min}[M_0(s)]
\le
\frac{
q_C^\top M_0(s)q_C
}{
q_C^\top q_C
}.
\]

Using Theorem 1,

\[
\lambda_{\min}[M_0(s)]
\le
\frac{
\Delta E_C
}{
2|C|
}.
\]

Therefore

\[
\boxed{
\kappa_c(s)
\ge
\frac{
E(s)-E(s^{(C)})
}{
2|C|
}.
}
\tag{10}
\]

A configuration possessing a strongly improving collective move cannot become linearly stable until the transverse confinement exceeds at least the corresponding average improvement scale.

This gives \(\kappa\) a concrete interpretation: it competes against the energetic value of collective escape directions.

---

# 15. Route geometry beyond infinitesimal perturbations

Theorems 1 and 2 concern the local geometry around binary configurations. Optimization performance, however, depends on finite trajectories.

The two transverse signs

\[
y_i>0
\]

and

\[
y_i<0
\]

represent two geometrically distinct routes between \(x_i=+1\) and \(x_i=-1\).

The ordinary projection energy cannot distinguish them.

The route fields can.

For the \(r\)-channel,

\[
r_i=x_i y_i,
\]

so

\[
r_i(x_i,-y_i)
=
-r_i(x_i,y_i).
\]

Consequently, for fixed neighboring route states,

\[
B_i r_i
\]

assigns different local energies to the two transverse directions.

The collective state therefore determines not only whether the Ising coordinate is pushed toward \(+1\) or \(-1\), but also which angular approach toward that transition is preferred.

The higher-order coordinate

\[
p_i=y_i^3
\]

extends this route distinction to the longitudinal midpoint \(x_i=0\), where the \(r\)-channel vanishes.

Because the \(p\)-channel is higher order near \(y=0\), it modifies global route geometry without changing Theorem 1.

The parameter \(\gamma\) therefore separates two roles:

\[
r=x\odot y
\]

encodes correct local collective-flip curvature, while

\[
p=y^{\odot3}
\]

controls nonlocal route separation.

Whether the latter materially improves optimization is an empirical question and should be tested by ablation rather than assumed.

---

# 16. Relationship to simulated bifurcation

Simulated bifurcation numerically evolves nonlinear Hamiltonian systems with canonical dynamical variables and exploits adiabatic bifurcation and chaotic dynamics [6]. citeturn459052search0

The present dynamics are structurally different.

There is no canonical momentum \(p_i\).

There is no equation of the form

\[
\dot x_i
=
\frac{\partial H}{\partial p_i},
\qquad
\dot p_i
=
-\frac{\partial H}{\partial x_i}.
\]

There is no inertial mechanism.

There is no nonlinear oscillator whose two bifurcation branches represent Ising states.

Instead,

\[
\dot\theta
=
-\mu\nabla_\theta\mathcal F.
\]

The dynamics are purely dissipative.

The superficial similarity lies only in the use of a time-dependent landscape that ultimately favors two binary sectors. In simulated bifurcation, escape can result from Hamiltonian motion, momentum, interaction forces, and chaotic evolution. In the present model, escape arises from negative-curvature directions and route-dependent first-order flow on an enlarged state space.

---

# 17. Relationship to XY and vector-spin relaxations

For standard planar XY spins,

\[
v_i
=
(\cos\theta_i,\sin\theta_i),
\]

the pairwise interaction generally has the form

\[
J_{ij}
v_i^\top v_j
=
J_{ij}
\cos(\theta_i-\theta_j),
\]

or equivalently,

\[
J_{ij}
(x_i x_j+y_i y_j).
\]

Thus longitudinal and transverse components participate directly in the same vector-spin interaction.

Anisotropy can subsequently force the spins toward an Ising axis. Such XY/vector-spin relaxations and their relation to Ising computation have been studied previously [7], and recent work explicitly exploits high-dimensional spin dynamics or XY-to-Ising transitions to improve escape from local minima [8,9]. citeturn459052search1turn328433search0turn328433search7

The present construction has a different structure:

\[
E_{\rm Ising}(x)
=
\frac12x^\top Jx+h^\top x
\]

is retained as a distinguished longitudinal objective.

The transverse interactions instead act through feature coordinates such as

\[
r=x\odot y
\]

and

\[
p=y^{\odot3},
\]

which vanish on the entire binary manifold.

The auxiliary interactions therefore alter transition geometry without contributing to the final objective.

Most importantly, the specific \(r=x\odot y\) construction produces the exact collective-flip curvature identity in Theorem 1.

The contribution is consequently not the generic use of continuous angular dimensions but the structure imposed on their interaction with the original Ising Hamiltonian.

---

# 18. Relationship to scalar continuous relaxations

Mean-field annealing and Ginzburg–Landau-type continuous Boolean optimization replace discrete variables by scalar continuous variables and progressively encourage binary values [4,5]. citeturn459052search5turn821917search3

Such methods operate on a continuous scalar domain such as

\[
x_i\in[-1,1]
\]

or

\[
x_i\in\mathbb R
\]

with a binary penalty.

The present system instead retains an explicit transverse degree of freedom.

Two states having the same longitudinal value

\[
x_i
\]

but opposite

\[
y_i
\]

can experience different route fields.

Consequently, the transverse state cannot in general be quotiented out into a one-dimensional scalar evolution.

This route dependence is the essential additional degree of freedom.

---

# 19. Relationship to oscillator Ising machines

Oscillator Ising machines typically encode binary values in oscillator phases and use mechanisms such as synchronization or second-harmonic injection locking to favor phase states separated by \(\pi\). Such dynamics are well established and can suffer from premature phase freezing when the binarization term becomes too strong [10]. citeturn472308search0turn472308search3

Although Eq. (4) is expressed using angles, \(\theta_i\) is not an oscillator phase in the physical sense.

There is no intrinsic frequency,

\[
\omega_i,
\]

no periodic driving,

no injection locking,

and no synchronization equation.

The angular coordinate is simply a convenient parameterization of \(S^1\).

---

# 20. GPU implementation

The computational structure is unusually simple.

For one trajectory, every iteration requires

\[
x=\cos\theta,
\]

\[
y=\sin\theta,
\]

\[
r=x\odot y,
\]

\[
p=y^{\odot3},
\]

followed by

\[
A=Jx+h,
\]

\[
B=Jr,
\]

\[
C=Jp.
\]

All pairwise interactions therefore involve the same matrix \(J\).

The three vectors can be concatenated,

\[
Z
=
\begin{bmatrix}
x&r&p
\end{bmatrix},
\]

and all interaction fields computed simultaneously,

\[
\boxed{
G=JZ.
}
\tag{11}
\]

For dense \(J\), Eq. (11) is a dense matrix–matrix multiplication.

For sparse \(J\), it is a sparse matrix–dense matrix multiplication.

The remainder of Eq. (4) consists exclusively of elementwise arithmetic.

---

# 21. Parallel trajectories

Suppose \(R\) independent solver trajectories are evolved simultaneously.

Let

\[
\Theta
\in
\mathbb R^{N\times R}.
\]

Compute

\[
X=\cos\Theta,
\qquad
Y=\sin\Theta,
\]

\[
R_1=X\odot Y,
\qquad
R_2=Y^{\odot3}.
\]

Concatenate

\[
Z
=
[
X,\,
R_1,\,
R_2
]
\in
\mathbb R^{N\times3R}.
\]

Then

\[
\boxed{
G=JZ
}
\tag{12}
\]

computes the complete pairwise interaction workload for all \(R\) trajectories in one GEMM or SpMM operation.

This is preferable to executing \(3R\) independent matrix–vector multiplications because it exposes more parallel work and generally improves accelerator utilization.

For \(\gamma=0\), only the \(x\) and \(r\) channels are required and the matrix has \(2R\) columns.

---

# 22. Computational complexity

Let

\[
M=\operatorname{nnz}(J)
\]

for a sparse Ising graph.

The pairwise cost per step and per trajectory is

\[
O(M)
\]

for each active interaction channel.

The minimal \(x+r\) model therefore requires approximately

\[
2M
\]

weighted edge operations per trajectory per time step.

The route-complete \(x+r+p\) model requires approximately

\[
3M.
\]

All remaining work is

\[
O(N)
\]

per trajectory.

For \(R\) parallel trajectories,

\[
T_{\rm sparse}
=
O(MR)
\]

with a small channel-dependent prefactor.

For dense \(J\),

\[
T_{\rm dense}
=
O(N^2R).
\]

The memory requirement is

\[
O(M+NR)
\]

for sparse problems, excluding modest temporary channel storage.

Importantly, there is no data-dependent sequential spin-selection loop and no accept/reject branching.

Every variable is updated synchronously.

---

# 23. Numerical integration

The most direct discretization of Eq. (4) is explicit Euler,

\[
\theta^{(n+1)}
=
\theta^{(n)}
+
\Delta t\,
\dot\theta^{(n)}.
\]

This maximizes computational simplicity.

A second-order explicit method such as Heun or midpoint integration may provide better stability for stiff late-stage confinement while preserving GPU parallelism.

A natural scaling is to normalize the problem using an interaction scale such as

\[
\Lambda
=
\max_i
\left(
|h_i|
+
\sum_j|J_{ij}|
\right)
\]

and express

\[
J\rightarrow J/\Lambda,
\qquad
h\rightarrow h/\Lambda.
\]

The mobility and time step can then be chosen in dimensionless units.

Adaptive per-variable integration is intentionally avoided in the initial implementation because it introduces branching and complicates GPU synchronization.

---

# 24. Confinement schedule

A minimal schedule is

\[
\kappa(t)
=
\kappa_0
+
(\kappa_f-\kappa_0)
\left(
\frac{t}{T}
\right)^\nu,
\qquad
0\le t\le T.
\tag{13}
\]

The exponent \(\nu\) controls how long the dynamics remain in the transversely open regime.

A particularly interesting choice is

\[
\kappa_0<0,
\]

because negative confinement explicitly destabilizes the binary axis early in the optimization.

The schedule then crosses

\[
\kappa=0,
\]

where Theorem 1 has its direct interpretation, before becoming positive and eventually forcing binarization.

The optimal schedule should not be assumed to be monotonic in future work. Recent XY-to-Ising studies report benefits from temporarily restoring continuous spin freedom after partial binarization, suggesting that re-opening the transverse manifold may sometimes be useful. citeturn328433search7

The first implementation, however, should use a monotonic schedule to isolate the effect of the proposed route interactions.

---

# 25. Candidate extraction

The instantaneous binary candidate is

\[
\boxed{
s_i(t)=\operatorname{sign}[x_i(t)].
}
\]

The exact discrete objective

\[
E[s(t)]
\]

can be evaluated periodically.

The solver should retain

\[
s_{\rm best}
=
\arg\min_t
E[s(t)]
\]

rather than assuming that the final rounded configuration is necessarily the best encountered candidate.

For GPU execution, exact discrete evaluation need not be performed every integration step. It can be performed at fixed intervals or whenever a sufficiently large fraction of signs changes.

---

# 26. Minimal algorithm

For \(R\) trajectories, the complete solver is:

\[
\Theta
\leftarrow
\text{random initial angles}.
\]

At every integration step,

\[
X=\cos\Theta,
\]

\[
Y=\sin\Theta,
\]

\[
R_1=X\odot Y,
\]

\[
R_2=Y^{\odot3}.
\]

Construct

\[
Z=[X,R_1,R_2].
\]

Compute

\[
G=JZ.
\]

Split \(G\) into

\[
JX,\quad JR_1,\quad JR_2.
\]

Broadcast \(h\) into the longitudinal field,

\[
A=JX+h.
\]

Then update

\[
\dot\Theta
=
\mu
\Big[
Y\odot A
-
(X^{\odot2}-Y^{\odot2})\odot JR_1
-
3\gamma X\odot Y^{\odot2}\odot JR_2
-
\kappa XY
\Big].
\]

Finally,

\[
\Theta
\leftarrow
\Theta+\Delta t\,\dot\Theta.
\]

The central computational kernel is therefore Eq. (12).

---

# 27. Interpretation as a dynamically selected family of potentials

Equation (2) allows another useful interpretation.

At any instant, variable \(i\) moves in a one-dimensional angular potential whose coefficients are

\[
(A_i,B_i,C_i).
\]

These coefficients are not externally prescribed.

They are generated by all other variables.

Thus the collective state selects one member from a continuous family

\[
V_i(\theta;A,B,C,\kappa).
\]

As other variables move,

\[
(A_i,B_i,C_i)
\]

change continuously.

The local potential therefore deforms continuously during the trajectory.

This can be viewed intuitively as follows.

The longitudinal Ising interaction directly pushes or pulls each variable toward one binary sector.

The transverse interactions simultaneously change which route through the enlarged state space is favorable.

A variable can therefore be driven toward a different binary state not only because the longitudinal force becomes large enough, but because the surrounding configuration opens a lower-energy transverse corridor.

This interpretation resembles a continuously changing family of bifurcation-like landscapes, but the mathematics itself contains no bifurcation oscillator and no momentum.

---

# 28. What the theoretical result does and does not establish

Theorem 2 is strong but limited.

It establishes that at

\[
\kappa=0,
\]

a strictly suboptimal binary state cannot be a local minimum of the continuous energy.

It does **not** establish that gradient flow reaches the global Ising optimum.

Several failure modes remain possible.

The flow can reach a nonbinary stationary point.

A descending transverse direction can lead into a continuous basin that does not connect monotonically to the lower discrete state that generated the initial curvature.

A global optimum can possess negative continuous directions that do not correspond to binary subset flips.

A confinement schedule can stabilize inferior binary configurations before the trajectory reaches the optimum.

Finite-step numerical integration can also alter the ideal continuous dynamics.

Thus the present theory identifies a mechanism for eliminating a specific class of traps—strictly suboptimal binary minima—but does not solve the NP-hard global optimization problem analytically.

---

# 29. Experimental hypotheses

The principal empirical hypothesis is that converting improving collective spin flips into continuous negative-curvature directions should reduce trapping in discrete metastable states.

A second hypothesis is that the route-separation channel \(p=y^3\) should improve the probability that an initially favorable transverse excursion develops into a complete basin-to-basin transition rather than terminating in an intermediate continuous minimum.

A third hypothesis is that the matrix-dominated synchronous update structure should make the method particularly attractive when many trajectories are executed simultaneously on GPU hardware.

These hypotheses must be tested independently. Superior numerical performance should not be inferred from the theoretical saddle property alone.

---

# 30. Benchmark methodology

A credible evaluation should include structurally different problem families rather than only one random graph ensemble. MQLib provides thousands of Max-Cut and QUBO instances and was explicitly designed for systematic heuristic evaluation; its associated study emphasizes that solver ranking is highly instance-dependent [11]. citeturn821917search0

Wishart-planted Ising instances are particularly useful because the ground state is known while landscape ruggedness can be tuned over an easy–hard–easy regime [12]. citeturn332191search3

The experimental suite should therefore contain at minimum dense Sherrington–Kirkpatrick-type spin glasses, sparse random regular and Erdős–Rényi Max-Cut problems, Wishart-planted instances, and heterogeneous MQLib instances.

The method should be compared at equal wall-clock or equal computational-budget conditions with simulated annealing, a scalar continuous or mean-field solver, simulated bifurcation where an appropriate implementation is available, and a conventional XY-to-Ising relaxation.

For known-optimum instances, the primary statistical metrics should be ground-state success probability and time-to-solution.

For unknown-optimum instances, the appropriate metrics are best-known gap, objective improvement versus time, solution distribution over repeated runs, and GPU throughput.

---

# 31. Required ablation studies

The central mechanism can be falsified cleanly.

Removing the route interaction,

\[
r^\top Jr\rightarrow0,
\]

reduces the method toward a longitudinal scalar-like relaxation on the circle.

Setting

\[
\gamma=0
\]

retains the exact collective-flip curvature but removes the higher-order midpoint route-separation mechanism.

Comparing these two systems determines whether the theoretically motivated local curvature alone is useful.

The full model,

\[
\gamma>0,
\]

tests whether maintaining route asymmetry away from the binary manifold materially improves successful transitions.

A fixed \(\kappa\) should additionally be compared against the deformation schedule to determine whether continuation is necessary or whether the static manifold energy already provides the benefit.

---

# 32. Diagnostic measurements

Performance statistics alone are insufficient to establish mechanism.

For selected benchmark instances, trajectories should record

\[
x_i(t),
\qquad
y_i(t),
\qquad
r_i(t),
\]

the discrete candidate

\[
s_i(t)=\operatorname{sign}x_i(t),
\]

and the fields

\[
A_i(t),
\qquad
B_i(t).
\]

Near metastable binary configurations, one can estimate the minimum eigenvalues of

\[
M_\kappa(s)
\]

or use Lanczos iterations to identify dominant negative-curvature modes.

These modes can be compared with actual groups of spins that subsequently change sign.

A particularly direct test of the theory is to compare a negative eigenvector or low-Rayleigh-quotient transverse direction with the subset of variables involved in a later collective escape.

The experiment should determine whether the theoretically available collective direction is actually exploited by the nonlinear dynamics.

---

# 33. Scaling experiments

GPU performance should be measured separately from solution quality.

For sparse problems, vary both

\[
N
\]

and

\[
M=\operatorname{nnz}(J).
\]

For dense problems, vary \(N\).

For each size, vary the number of simultaneous trajectories \(R\).

The expected hardware transition is from under-utilized matrix–vector-like operation at small \(R\) toward increasingly efficient GEMM or SpMM execution at moderate \(R\).

Relevant measurements include effective edge updates per second, matrix-operation throughput, memory bandwidth utilization, kernel-launch overhead, total solver trajectories per second, and time-to-best-solution.

No algorithmic speedup should be claimed solely from high arithmetic throughput; final comparisons must use equal solution-quality targets.

---

# 34. Numerical precision

The dynamical state is continuous and does not require exact arithmetic.

FP32 is therefore a natural first choice for trajectory evolution.

For very large or poorly scaled \(J\), BF16 or FP16 may introduce excessive distortion in accumulated interaction fields unless suitable scaling is used.

Mixed precision is a plausible architecture:

\[
JZ
\]

may use accelerator tensor operations at reduced precision while objective evaluation and best-solution bookkeeping use FP32 or FP64 accumulation.

The influence of numerical precision on both trajectory diversity and final objective quality should be measured rather than assumed.

---

# 35. Generalization to higher-dimensional route manifolds

The planar model contains two transverse directions corresponding to the two semicircles.

The framework can be generalized.

Let each spin have state

\[
q_i\in\mathcal M
\]

with two distinguished terminal points

\[
q_i^+,
\qquad
q_i^-.
\]

Let

\[
x_i(q_i)
\]

be the longitudinal Ising coordinate satisfying

\[
x_i(q_i^\pm)=\pm1.
\]

Introduce route features

\[
\phi_a(q_i),
\qquad
a=1,\ldots,K,
\]

that satisfy

\[
\phi_a(q_i^\pm)=0.
\]

A general auxiliary interaction is then

\[
\mathcal F_{\rm route}
=
\frac12
\sum_{a=1}^K
w_a
\phi_a^\top
J
\phi_a.
\]

Every route feature disappears on the binary manifold, guaranteeing binary faithfulness.

The planar construction corresponds to

\[
\phi_1=x\odot y,
\]

and optionally

\[
\phi_2=y^{\odot3}.
\]

Higher-dimensional manifolds can provide additional route classes, but they should only be introduced if the planar model demonstrates experimentally that route-dependent interactions contribute useful optimization behavior.

---

# 36. Design principle for additional route features

Any proposed route coordinate should satisfy three distinct conditions.

It should vanish on both binary endpoints so that the final objective remains unchanged.

Its low-order behavior near the endpoints should be analytically controlled so that the desired local curvature is preserved.

Finally, if the purpose of the coordinate is route differentiation, it should distinguish geometrically different transition corridors away from the binary manifold.

The route feature should therefore be designed from its effect on optimization geometry, not from visual analogy to a particular physical system.

This principle avoids the danger of introducing additional dimensions that are mathematically redundant.

---

# 37. Discussion

The proposed construction can be viewed as a continuous relaxation in which the additional degrees of freedom do not merely soften the binary constraint. They modify which coordinated transitions are locally downhill.

This distinction is important.

A scalar relaxation primarily asks how far a variable should move along a single binary coordinate.

The transverse route system additionally asks how groups of variables should leave their current Ising sectors.

The matrix

\[
SJS
\]

encodes pairwise compatibility of those collective transverse excursions, while

\[
\operatorname{diag}[S(Js+h)]
\]

encodes the local stability of the current assignment.

Their difference produces exactly the collective subset-flip energy when evaluated on indicator directions.

This result provides a direct bridge between discrete combinatorial structure and continuous differential geometry.

The confinement parameter then controls when these transverse escape directions cease to be available.

The resulting solver can therefore be interpreted as a dynamically deformed landscape in which the original Ising objective governs the terminal states, while temporary interaction channels govern the paths between them.

---

# 38. Distinction from simply adding dimensions

Additional dimensions do not automatically improve optimization.

If the energy depends exclusively on the longitudinal projection,

\[
E=E(x),
\]

all states with equal \(x\) are equivalent.

The extra coordinates are then redundant.

Likewise, adding arbitrary transverse interactions can create additional complexity without encoding useful information about the original optimization problem.

The present construction avoids both extremes.

The transverse interaction is derived from \(J\) itself.

Moreover, its leading-order effect has an exact discrete interpretation.

This is the principal reason to investigate the dynamics empirically.

---

# 39. Potential limitations

The main theoretical limitation is that binary-saddle elimination is weaker than global convergence.

Continuous stationary points not corresponding to binary configurations may still dominate the dynamics.

The quality of the continuation schedule may be important.

The higher-order route interaction may help or harm depending on instance structure.

For dense QUBOs, the \(O(N^2)\) interaction cost remains fundamental despite GPU acceleration.

For sparse problems, performance will depend strongly on graph structure and sparse-matrix efficiency.

Finally, the continuous dynamics introduce hyperparameters absent from the original QUBO, including \(\Delta t\), \(\kappa(t)\), \(\mu\), and potentially \(\gamma\).

Any practical advantage must survive fair tuning of competing algorithms.

---

# 40. Prior-work positioning

The general idea of physics-inspired Ising optimization is broad and mature [2]. Simulated bifurcation uses classical nonlinear Hamiltonian dynamics [6]; continuous Boolean and mean-field methods provide scalar relaxations [4,5]; XY/vector-spin systems use continuous angular degrees of freedom [7]; and recent optical work explicitly employs additional dimensions and dimensional collapse to improve Ising optimization [8,9]. citeturn332191search1turn459052search0turn459052search5turn821917search3turn459052search1turn328433search0turn328433search7

Accordingly, the scientifically distinguishing object examined here is not continuous spins or dimensional collapse themselves.

It is the specific decomposition into a longitudinal Ising interaction and auxiliary route interactions that vanish identically on the binary manifold, together with the endpoint-oriented route coordinate

\[
r=x\odot y
\]

and the resulting exact collective-flip curvature identity.

A literature search can reduce but cannot eliminate the possibility that mathematically equivalent constructions exist under different terminology. Any formal claim of novelty or freedom to operate requires a separate exhaustive prior-art and patent analysis.

---

# 41. Conclusion

We have formulated a first-order geometric dynamical system for QUBO and Ising optimization in which binary variables are embedded on a continuous route manifold.

The original Ising objective acts on the longitudinal coordinates,

\[
x_i=\cos\theta_i,
\]

while auxiliary interactions operate on route coordinates that vanish at every binary state.

The minimal endpoint-oriented coordinate

\[
r_i=x_i y_i
\]

produces the transverse Hessian

\[
M_0(s)
=
SJS
-
\operatorname{diag}[S(Js+h)].
\]

For every subset of spins \(C\),

\[
q_C^\top M_0(s)q_C
=
\frac12
\left[
E(s^{(C)})-E(s)
\right].
\]

This exact identity implies that every strictly suboptimal binary configuration is a saddle of the zero-confinement continuous landscape.

Thus the continuous system possesses descending transverse directions corresponding to collective discrete improvements even when ordinary local spin flips are unfavorable.

A confinement parameter

\[
\kappa(t)
\]

subsequently suppresses transverse motion and returns the system to the binary manifold.

The resulting optimization mechanism requires neither momentum nor oscillators. It is a dissipative geometric flow.

Computationally, all pairwise interactions are expressed through repeated application of the original coupling matrix \(J\). Multiple route channels and multiple independent solver trajectories can be concatenated into a single matrix multiplication,

\[
G=JZ,
\]

followed by elementwise updates.

The method is therefore naturally compatible with dense GPU GEMM and sparse GPU SpMM execution.

The principal unresolved question is empirical: whether the transverse directions guaranteed by the local theory develop into useful finite escape trajectories often enough to outperform simpler continuous relaxations and established Ising heuristics.

That question is sharply testable.

If the answer is positive, the framework would suggest a broader class of binary optimizers based not on stochastic barrier crossing, momentum-driven bifurcation, or oscillator synchronization, but on **problem-derived geometric escape routes whose interactions disappear exactly when the discrete solution manifold is reached**.

---

# References

[1] A. Lucas, “Ising formulations of many NP problems,” *Frontiers in Physics* **2**, 5 (2014), doi:10.3389/fphy.2014.00005. citeturn332191search0

[2] N. Mohseni, P. L. McMahon, and T. Byrnes, “Ising machines as hardware solvers of combinatorial optimization problems,” *Nature Reviews Physics* **4**, 363–379 (2022), doi:10.1038/s42254-022-00440-8. citeturn332191search1

[3] S. Kirkpatrick, C. D. Gelatt Jr., and M. P. Vecchi, “Optimization by Simulated Annealing,” *Science* **220**, 671–680 (1983), doi:10.1126/science.220.4598.671. citeturn332191search5

[4] M. T. Veszeli and G. Vattay, “Mean field approximation for solving QUBO problems,” *PLOS ONE* **17**, e0273709 (2022), doi:10.1371/journal.pone.0273709. citeturn459052search5

[5] Y.-S. Niu and R. Glowinski, “Discrete Dynamical System Approaches for Boolean Polynomial Optimization,” *Journal of Scientific Computing* **92**, 46 (2022), doi:10.1007/s10915-022-01882-z. citeturn821917search3

[6] H. Goto, K. Tatsumura, and A. R. Dixon, “Combinatorial optimization by simulating adiabatic bifurcations in nonlinear Hamiltonian systems,” *Science Advances* **5**, eaav2372 (2019), doi:10.1126/sciadv.aav2372. citeturn459052search0

[7] M. Erementchouk, A. Shukla, and P. Mazumder, “On computational capabilities of Ising machines based on nonlinear oscillators,” *Physica D* **437**, 133334 (2022), doi:10.1016/j.physd.2022.133334. citeturn459052search1

[8] S. Chiavazzo, M. Calvanese Strinati, C. Conti, and D. Pierangeli, “Ising Machine by Dimensional Collapse of Nonlinear Polarization Oscillators,” *Physical Review Letters* **135**, 063801 (2025), doi:10.1103/qs29-2xqc. citeturn328433search0

[9] K. Kim and Y. Yamamoto, “Accelerating a coherent Ising machine by XY-Ising spin transition,” *Scientific Reports* **16**, 10396 (2026). citeturn328433search7

[10] M. Farasat, E. M. H. E. B. Ekanayake, and N. Shukla, “Spin freezing in oscillator Ising machines: When second-harmonic injection impedes computation,” *Physical Review Applied* **25**, 034038 (2026). citeturn472308search3

[11] I. Dunning, S. Gupta, and J. Silberholz, “What Works Best When? A Systematic Evaluation of Heuristics for Max-Cut and QUBO,” *INFORMS Journal on Computing* **30**, 608–624 (2018), doi:10.1287/ijoc.2017.0798. citeturn821917search0

[12] F. Hamze, J. Raymond, C. A. Pattison, K. Biswas, and H. G. Katzgraber, “Wishart planted ensemble: A tunably rugged pairwise Ising model with a first-order phase transition,” *Physical Review E* **101**, 052102 (2020), doi:10.1103/PhysRevE.101.052102. citeturn332191search3