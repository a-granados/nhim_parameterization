# nhim_parameterization
Distributed computation of Normally Hyperbolic Manfiolds via parameterization method
<img src="https://github.com/a-granados/nhim_parameterization/blob/master/movie.gif">

This is a general purpose collection of tools aiming to compute Normally Hyperbolic Manfiold (NHIMs) by beams of the so-called Parameterization method developed by Cabré, de la Llave and Fontich (see <a href="http://www.sciencedirect.com/science/article/pii/S0022039604005170"> this paper </a>).
It is here implemented following the Newton-like method described in Chapter 5 of the <a href="http://www.springer.com/gp/book/9783319296609"> book </a> by Haro, Canadell, Figueras, Luque and Mondelo.
Taking advantage of grid formulation, the software is parallelized with OpenMP and makes use of all available processors.

The advantage of this Newton method is that it allows to compouted
- the parameterization of the Normally Hyperbolic Manifold
- its dinner dynamics
- its normal bundle.

The latter is a good approximation of the local stable and unstable manifolds and, when iterated, it provides good approximations of the global **stable and unstable manifolds**. The software iterates the normal bundle and computes proper re-sampling to guarantee that points along the leaves always reach a certain density. Points generating the manifolds are printed in files in a proper format to be used with matlab surf (see plot_manifolds.m). An example can be seen in <a href="http://people.compute.dtu.dk/algr/nhim_animation.html"> this animation </a>.



The parameterizations and inner dynamics are given by as functions evaluated at grid points. We use GSL 2-dimensional libraries to interpolate them and the user can choose between linear and cubic splines.

Initially, these tools were programmed to compute a NHIM in an energy harvesting system based on two coupled piezoelectric oscillators under a periodic forcing. There results can be found in <a href="http://arxiv.org/abs/1609.03215"> this paper</a>. Therefore, although the software is general, the current version has the following **limitations**:
- It is implemented for maps
- It is limited to two-dimensional NHIMs for maps
- the dimension of the stable and unstable leafs is limited to two and one, respectively.
