akx -- A matrix powers specializer built on the ctree SEJITS framework
=============

Matrix powers is a basis spanning the Krylov subspace. {x, Ax, A^2x, ..., A^kx}

For more information on ctree see [ctree on github](http://github.com/ucb-sejits/ctree>)


Requirements
=============
* ctree (http://github.com/ucb-sejits/ctree>)
* numpy and scipy (http://www.scipy.org/)
* PaToH (http://bmi.osu.edu/~umit/software.html#patoh)
* Intel MKL (https://software.intel.com/en-us/non-commercial-software-development)


Examples
=============

<a name='simple'/>
### A simple matrix powers example
```python
import akx
import numpy
import scipy.io.mmio



# Replace mymatrix.mtx with the name of the matrix file to be used
matrix = scipy.io.mmio.mmread("mymatrix.mtx").tocsr()

# Number of steps per iteration
k = 2

# Symmetric optimization
sym = False

# Generated an akxobj associated with a given matrix and parameters
akxobj = akx.tune(matrix, k, sym)

# Total number of steps (Also highest power of matrix in Krylov subspace)
m = 4

# Generate a matrix of ones (all rows other than the first are overwritten)
vecs = numpy.ones((m, matrix.shape[0]))

# Actually perform the operation
akxobj.powers(vecs)



# Store the results into a file called Vecs
f = open("Vecs", "w")
print >>f, "Krylov vectors:"
print >>f, "".join("x_%d\t" % n for n in xrange(m + 1))
for element in vecs.transpose():
	print >>f, "".join("%.3g\t" % n for n in element)
```


Benchmark
=============

The performance of the specializer was benckmarked using conjugate gradient as an example
The following were run on a 4 core, 8 thread Intel processor running Ubuntu 12.04

A symmetric, positive definite matrix with 259,789 rows/columns and 4,242,673 entries was used in the benckmark (http://www.cise.ufl.edu/research/sparse/matrices/Um/offshore.html)

| Tool        | Calculation Time (s)|
|:-----------:|--------------------:|
| Specializer | 13.68               |
| Scipy       | 14.35               |