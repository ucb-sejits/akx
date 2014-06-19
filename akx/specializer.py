"""
Parses the python AST below, transforms it to C, JITs it, and runs it.
"""

import logging

logging.basicConfig(level=20)

import numpy as np

from ctree.frontend import get_ast
from ctree.c.nodes import *
from ctree.c.types import *
from ctree.templates.nodes import *
from ctree.dotgen import to_dot
from ctree.transformations import *
from ctree.jit import LazySpecializedFunction
from ctree.types import get_ctree_type

# ---------------------------------------------------------------------------
# Specializer code


class AkxGenerator(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        """
        Analyze arguments and return a 'subconfig', a hashable object
        that classifies them. Arguments with identical subconfigs
        might be processed by the same generated code.
        """

        return {
            'b_m': args[0],
            'b_n': args[1],
            'b_transpose': args[2],
            'browptr_comp': args[3],
            'brcolidx_comp': args[4],
            'basis': args[5],
        }


    def transform(self, py_ast, program_config):
        """
        Convert the Python AST to a C AST according to the directions
        given in program_config.
        """
        arg_config, tuner_config = program_config
        #variants_tuple = arg_config['variants_tuple']

        b_m = arg_config['b_m']
        b_n = arg_config['b_n']
        b_transpose = arg_config['b_transpose']
        browptr_comp = arg_config['browptr_comp']
        brcolidx_comp = arg_config['brcolidx_comp']
        basis = arg_config['basis']


        tree = CFile("generated", [
            StringTemplate("""
            //#include <Python.h>
            //#include <numpy/arrayobject.h>

            // C headers
            #include <stdlib.h> // for NULL
            #include <stdio.h>  // for fprintf

            #ifdef __SSE3__ // will be defined when compiling, but not when checking dependencies
            #include <pmmintrin.h> // for SSE
            #endif

            #include "akx.h"

            #ifdef _OPENMP
            #include <omp.h>
            #else
            #include <pthread.h> // for pthreads stuff
            pthread_barrier_t barrier;
            #endif
            """),
            StringTemplate("""
            //load_y tests
            """),
            self.load_y(y = 'y', ib = 'ib', b_m = 4, b_n = 6, b_transpose = 0),
            self.load_y(y = 'y', ib = 'ib', b_m = 4, b_n = 6, b_transpose = 1),
            self.load_y(y = 'y', ib = 'ib', b_m = 5, b_n = 6, b_transpose = 1),
            StringTemplate("""
            //load_y_zero tests
            """),
            self.load_y_zero(y = 'y', b_m = 4, b_n = 6, b_transpose = 0),
            self.load_y_zero(y = 'y', b_m = 4, b_n = 6, b_transpose = 1),
            self.load_y_zero(y = 'y', b_m = 5, b_n = 6, b_transpose = 1),
            StringTemplate("""
            //store_y tests
            """),
            self.store_y(y = 'y', ib = 'ib', b_m = 4, b_n = 6, b_transpose = 0),
            self.store_y(y = 'y', ib = 'ib', b_m = 4, b_n = 6, b_transpose = 1),
            self.store_y(y = 'y', ib = 'ib', b_m = 5, b_n = 6, b_transpose = 1),
            StringTemplate("""
            //load_x tests
            """),
            self.load_x(x = 'x', jb = 'jb', b_m = 4, b_n = 6, b_transpose = 0),
            self.load_x(x = 'x', jb = 'jb', b_m = 4, b_n = 6, b_transpose = 1),
            self.load_x(x = 'x', jb = 'jb', b_m = 5, b_n = 6, b_transpose = 1),
            StringTemplate("""
            //do_tile tests
            """),
            self.do_tile(y = 'y', x = 'x', b_m = 5, b_n = 4, b_transpose = 0),
            self.do_tile(y = 'y', x = 'x', b_m = 5, b_n = 3, b_transpose = 0),
            self.do_tile(y = 'y', x = 'x', b_m = 4, b_n = 4, b_transpose = 1),
            self.do_tile(y = 'y', x = 'x', b_m = 5, b_n = 4, b_transpose = 1),
        ])

        tree = PyBasicConversions().visit(tree)

        entry_point_typesig = FuncType().as_ctype()
        return Project([tree]), entry_point_typesig

    def load_y(self, y, ib, b_m, b_n, b_transpose):
        node_list = []
        if (not b_transpose) and (b_n % 2 == 0):
            for i in xrange(b_m):
                node_list.append(StringTemplate("""\
                    __m128d $y$i = _mm_load_sd(&y[$ib*$b_m + $i]);
                    """,{
                        'y': StringTemplate(y),
                        'i': Constant(i),
                        'ib': StringTemplate(ib),
                        'b_m': Constant(b_m)
                        }
                    )
                )
        elif b_transpose and (b_m % 2 == 0):
            for i in xrange(0, b_m, 2):
                node_list.append(StringTemplate("""\
                    __m128d $y$i = _mm_load_pd(&y[$ib*$b_m + $i]);
                    """,{
                        'y': StringTemplate(y),
                        'i': Constant(i),
                        'ib': StringTemplate(ib),
                        'b_m': Constant(b_m)
                        }
                    )
                )
        else:
            for i in xrange(b_m):
                node_list.append(StringTemplate("""\
                    double $y$i = y[$ib*$b_m + $i];
                    """,{
                        'y': StringTemplate(y),
                        'i': Constant(i),
                        'ib': StringTemplate(ib),
                        'b_m': Constant(b_m)
                        }
                    )
                )
        return node_list

    def load_y_zero(self, y, b_m, b_n, b_transpose):
        node_list = []
        if (not b_transpose) and (b_n % 2 == 0):
            for i in xrange(b_m):
                node_list.append(StringTemplate("""\
                    __m128d $y$i = = _mm_setzero_pd();
                    """,{
                        'y': StringTemplate(y),
                        'i': Constant(i)
                        }
                    )
                )
        elif b_transpose and (b_m % 2 == 0):
            for i in xrange(0, b_m, 2):
                node_list.append(StringTemplate("""\
                    __m128d $y$i = = _mm_setzero_pd();
                    """,{
                        'y': StringTemplate(y),
                        'i': Constant(i)
                        }
                    )
                )
        else:
            for i in xrange(b_m):
                node_list.append(StringTemplate("""\
                    double $y$i = 0.0;
                    """,{
                        'y': StringTemplate(y),
                        'i': Constant(i)
                        }
                    )
                )
        return node_list

    def store_y(self, y, ib, b_m, b_n, b_transpose):
        node_list = []
        if (not b_transpose) and (b_n % 2 == 0):
            for i in xrange(b_m):
                node_list.append(StringTemplate("""\
                    _mm_store_sd(&y[$ib*$b_m + $i], _mm_hadd_pd($y$i, $y$i));
                    """,{
                        'y': StringTemplate(y),
                        'i': Constant(i),
                        'ib': StringTemplate(ib),
                        'b_m': Constant(b_m)
                        }
                    )
                )
        elif b_transpose and (b_m % 2 == 0):
            for i in xrange(0, b_m, 2):
                node_list.append(StringTemplate("""\
                    _mm_store_pd(&y[$ib*$b_m + $i], $y$i);
                    """,{
                        'y': StringTemplate(y),
                        'i': Constant(i),
                        'ib': StringTemplate(ib),
                        'b_m': Constant(b_m)
                        }
                    )
                )
        else:
            for i in xrange(b_m):
                node_list.append(StringTemplate("""\
                    y[$ib*$b_m + $i] = $y$i;
                    """,{
                        'y': StringTemplate(y),
                        'i': Constant(i),
                        'ib': StringTemplate(ib),
                        'b_m': Constant(b_m)
                        }
                    )
                )
        return node_list

    def load_x(self, x, jb, b_m, b_n, b_transpose):
        node_list = []
        if (not b_transpose) and (b_n % 2 == 0):
            for j in xrange(0, b_n, 2):
                node_list.append(StringTemplate("""\
                    __m128d $x$j = _mm_load_pd(&x[$jb*$b_n + $j]);
                    """,{
                        'x': StringTemplate(x),
                        'j': Constant(j),
                        'jb': StringTemplate(jb),
                        'b_n': Constant(b_n)
                        }
                    )
                )
        elif b_transpose and (b_m % 2 == 0):
            for j in xrange(b_n):
                node_list.append(StringTemplate("""\
                    __m128d $x$j = _mm_load1_pd(&x[$jb*$b_n + $j]);
                    """,{
                        'x': StringTemplate(x),
                        'j': Constant(j),
                        'jb': StringTemplate(jb),
                        'b_n': Constant(b_n)
                        }
                    )
                )
        else:
            for j in xrange(b_n):
                node_list.append(StringTemplate("""\
                    double $x$j = x[$jb*$b_n + $j];
                    """,{
                        'x': StringTemplate(x),
                        'j': Constant(j),
                        'jb': StringTemplate(jb),
                        'b_n': Constant(b_n)
                        }
                    )
                )
        return node_list

    def do_tile(self, y, x, b_m, b_n, b_transpose):
        node_list = []
        if not b_transpose:
            if b_n % 2 == 0:
                for i in xrange(b_m):
                    for j in xrange(0, b_n, 2):
                        node_list.append(StringTemplate("""\
                            $y$i = _mm_add_pd($y$i, _mm_mul_pd($x$j, _mm_load_pd(&A->bvalues[jb*$b_mTIMESb_n + $iTIMESb_nPLUSj])));
                            """,{
                                'y': StringTemplate(y),
                                'i': Constant(i),
                                'x': StringTemplate(x),
                                'j': Constant(j),
                                'b_mTIMESb_n': Constant(b_m * b_n),
                                'iTIMESb_nPLUSj': Constant(i * b_n + j)
                                }
                            )
                        )
            else:
                for i in xrange(b_m):
                    for j in xrange(b_n):
                        node_list.append(StringTemplate("""\
                            $y$i += A->bvalues[jb*$b_mTIMESb_n + $iTIMESb_nPLUSj] * $x$j;
                            """,{
                                'y': StringTemplate(y),
                                'i': Constant(i),
                                'x': StringTemplate(x),
                                'j': Constant(j),
                                'b_mTIMESb_n': Constant(b_m * b_n),
                                'iTIMESb_nPLUSj': Constant(i * b_n + j)
                                }
                            )
                        )
        else:
            if b_m % 2 == 0:
                for j in xrange(b_n):
                    for i in xrange(0, b_m, 2):
                        node_list.append(StringTemplate("""\
                            $y$i = _mm_add_pd($y$i, _mm_mul_pd($x$j, _mm_load_pd(&A->bvalues[jb*$b_mTIMESb_n + $jTIMESb_mPLUSi])));
                            """,{
                                'y': StringTemplate(y),
                                'i': Constant(i),
                                'x': StringTemplate(x),
                                'j': Constant(j),
                                'b_mTIMESb_n': Constant(b_m * b_n),
                                'jTIMESb_mPLUSi': Constant(j * b_m + i)
                                }
                            )
                        )
            else:
                for j in xrange(b_n):
                    for i in xrange(b_m):
                       node_list.append(StringTemplate("""\
                            $y$i += A->bvalues[jb*$b_mTIMESb_n + $jTIMESb_mPLUSi] * $x$j;
                            """,{
                                'y': StringTemplate(y),
                                'i': Constant(i),
                                'x': StringTemplate(x),
                                'j': Constant(j),
                                'b_mTIMESb_n': Constant(b_m * b_n),
                                'jTIMESb_mPLUSi': Constant(j * b_m + i)
                                }
                            )
                        )
        return node_list




class Akx(object):
    """
    A class for generating jit akx c source code.
    """

    def __init__(self):
        """Instantiate translator."""
        self.c_apply_all = AkxGenerator(get_ast(self.nothing), "do_nothing")

    def __call__(self, b_m, b_n, b_transpose, browptr_comp, brcolidx_comp, basis):
        """Apply the operator to the arguments via a generated function."""
        return self.c_apply_all(b_m, b_n, b_transpose, browptr_comp, brcolidx_comp, basis)

    def nothing(self):
        pass


# ---------------------------------------------------------------------------
# User code


def main():

    c_akx = Akx()
    c_akx(2, 4, 0, 8, 10, 0)

if __name__ == '__main__':
    main()
