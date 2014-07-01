akx -- A matrix powers specializer built on the ctree SEJITS framework
-------

Matrix powers is the basis for the Krylov subspace which spans {x, Ax, A^2x, ..., A^kx}

For more information on ctree see [ctree on github](http://github.com/ucb-sejits/ctree>).

Examples
=============

<a name='simple'/>
### A simple matrix powers example
```python
import scipy.io.mmio
matrix = scipy.io.mmio.mmread('.mtx matrix').tocsr()

b_m = 4
b_n = 4
b_transpose = 0
browptr_comp = 4
bcolidx_comp = 4
basis = 0

c_akx = Akx(b_m, b_n, b_transpose, browptr_comp, bcolidx_comp, basis, matrix)

k = 3
vecs = numpy.ones((1 + k, matrix.shape[0]))

basis = c_akx(vecs)
```