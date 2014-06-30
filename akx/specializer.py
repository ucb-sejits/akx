import logging

logging.basicConfig(level=20)

import numpy as np

from ctree.c.nodes import *
from ctree.templates.nodes import *
from ctree.transformations import *
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctypes import CFUNCTYPE

# ---------------------------------------------------------------------------
# Specializer code

class AkxFunction(ConcreteSpecializedFunction):

    def finalize(self, tree, entry_name, entry_type):
        self._c_function = self._compile(entry_name, tree, entry_type)
        return self

    def __call__(self):
        self._c_function()


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
            'bcolidx_comp': args[4],
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
        bcolidx_comp = arg_config['bcolidx_comp']
        basis = arg_config['basis']


        tree = CFile("generated", [
            FileTemplate('../templates/prologue'),
            self.bcsr_spmv('', b_m, b_n, b_transpose, browptr_comp, bcolidx_comp, basis),
            self.bcsr_spmv_rowlist('', b_m, b_n, b_transpose, browptr_comp, bcolidx_comp, basis),
            self.bcsr_spmv_stanzas('', b_m, b_n, b_transpose, browptr_comp, bcolidx_comp, basis),
            self.bcsr_spmv('_symmetric', b_m, b_n, b_transpose, browptr_comp, bcolidx_comp, basis),
            self.bcsr_spmv_rowlist('_symmetric', b_m, b_n, b_transpose, browptr_comp, bcolidx_comp, basis),
            self.bcsr_spmv_stanzas('_symmetric', b_m, b_n, b_transpose, browptr_comp, bcolidx_comp, basis),
            self.epilogue_dispatch(b_m, b_n, b_transpose, browptr_comp, bcolidx_comp, basis)
        ])

        entry_type = CFUNCTYPE(None)

        fn = AkxFunction()

        return fn.finalize(Project([tree]), "nothing", entry_type)

    def load_y(self, y, ib, b_m, b_n, b_transpose):
        node_list = []
        if (not b_transpose) and (b_n % 2 == 0):
            for i in xrange(b_m):
                node_list.append(StringTemplate("""\
                    __m128d $y$i = _mm_load_sd(&y[$ib*$b_m + $i]);
                    """, {
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
                    """, {
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
                    """, {
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
                    __m128d $y$i = _mm_setzero_pd();
                    """, {
                        'y': StringTemplate(y),
                        'i': Constant(i)
                        }
                    )
                )
        elif b_transpose and (b_m % 2 == 0):
            for i in xrange(0, b_m, 2):
                node_list.append(StringTemplate("""\
                    __m128d $y$i = _mm_setzero_pd();
                    """, {
                        'y': StringTemplate(y),
                        'i': Constant(i)
                        }
                    )
                )
        else:
            for i in xrange(b_m):
                node_list.append(StringTemplate("""\
                    double $y$i = 0.0;
                    """, {
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
                    """, {
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
                    """, {
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
                    """, {
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
                    """, {
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
                    """, {
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
                    """, {
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
                            """, {
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
                            """, {
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
                            """, {
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
                            """, {
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

    def do_tilerow(self, format, b_m, b_n, b_transpose, basis):
        node_list = []
        if format == '':
            node_list.append(self.load_y_zero("y", b_m, b_n, b_transpose))
            node_list.append(
                For(Assign(SymbolRef('jb'), ArrayRef(SymbolRef('browptr'), SymbolRef('ib'))),
                    Lt(SymbolRef('jb'), ArrayRef(SymbolRef('browptr'), Add(SymbolRef('ib'), Constant(1)))),
                    PreInc(SymbolRef('jb')),
                    [Assign(StringTemplate('index_t j'), ArrayRef(SymbolRef('bcolidx'), SymbolRef('jb'))),
                    self.load_x("x", "j", b_m, b_n, b_transpose),
                    self.do_tile("y", "x", b_m, b_n, b_transpose)]
                )
            )
            node_list.append(self.store_y("y", "ib", b_m, b_n, b_transpose))
        else:
            node_list.append(self.load_y("yi", "ib", b_m, b_n, b_transpose))
            node_list.append(self.load_x("xi", "ib", b_m, b_n, not b_transpose))
            node_list.append(
                For(Assign(SymbolRef('jb'), ArrayRef(SymbolRef('browptr'), SymbolRef('ib'))),
                    Lt(SymbolRef('jb'), ArrayRef(SymbolRef('browptr'), Add(SymbolRef('ib'), Constant(1)))),
                    PreInc(SymbolRef('jb')),
                    [Assign(StringTemplate('index_t j'), ArrayRef(SymbolRef('bcolidx'), SymbolRef('jb'))),
                     self.load_x("xj", "j", b_m, b_n, b_transpose),
                     self.do_tile("yi", "xj", b_m, b_n, b_transpose),
                     If(And(Gt(SymbolRef('j'), SymbolRef('ib')), Lt(SymbolRef('j'), SymbolRef('mb'))),
                        [self.load_y("yj", "j", b_m, b_n, not b_transpose),
                         self.do_tile("yj", "xi", b_m, b_n, not b_transpose),
                         self.store_y("yj", "j", b_m, b_n, not b_transpose)]
                     )]
                )
            )
            node_list.append(self.store_y("yi", "ib", b_m, b_n, b_transpose))
        if basis == 1:
            for i in xrange(b_m):
                node_list.append(StringTemplate("""\
                    y[ib*$b_m + $i] -= x[ib*$b_m + $i] * coeff;
                    """, {
                        'b_m': Constant(b_m),
                        'i': Constant(i),
                        }
                    )
                )

        return node_list

    def init(self, browptr_comp, bcolidx_comp):
        node_list = []
        if browptr_comp == 0:
            node_list.append(Assign(StringTemplate('index_t *__restrict__ browptr'), StringTemplate('A->browptr')))
        else:
            node_list.append(Assign(StringTemplate('uint16_t *__restrict__ browptr'), StringTemplate('A->browptr16')))
        if bcolidx_comp == 0:
            node_list.append(Assign(StringTemplate('index_t *__restrict__ bcolidx'), StringTemplate('A->bcolidx')))
        else:
            node_list.append(Assign(StringTemplate('uint16_t *__restrict__ bcolidx'), StringTemplate('A->bcolidx16')))

        return node_list

    def bcsr_spmv(self, format, b_m, b_n, b_transpose, browptr_comp, bcolidx_comp, basis):
        params = '\nconst struct bcsr_t *__restrict__ A,\nconst value_t *__restrict__ x,\nvalue_t *__restrict__ y,'
        if basis == 1:
            params += '\nvalue_t coeff,'
        params += '\nindex_t mb'

        return FunctionDecl(None, 'bcsr_spmv' + format, [StringTemplate(params)], [
            StringTemplate('index_t ib, jb;'),
            self.init(browptr_comp, bcolidx_comp),
            For(Assign(SymbolRef('ib'), Constant(0)),
                    Lt(SymbolRef('ib'), SymbolRef('mb')),
                    PreInc(SymbolRef('ib')),
                    self.do_tilerow(format, b_m, b_n, b_transpose, basis)
                )
            ]
        )

    def bcsr_spmv_rowlist(self, format, b_m, b_n, b_transpose, browptr_comp, bcolidx_comp, basis):
        params = '\nconst struct bcsr_t *__restrict__ A,\nconst value_t *__restrict__ x,\nvalue_t *__restrict__ y,'
        if basis == 1:
            params += '\nvalue_t coeff,'
        params += '\nindex_t mb,\nconst index_t *__restrict__ computation_seq,\nindex_t seq_len'

        return FunctionDecl(None, 'bcsr_spmv' + format + '_rowlist', [StringTemplate(params)], [
                StringTemplate('index_t q, ib, jb;'),
                self.init(browptr_comp, bcolidx_comp),
                For(Assign(SymbolRef('q'), Constant(0)),
                    Lt(SymbolRef('q'), SymbolRef('seq_len')),
                    PostInc(SymbolRef('q')),
                    [
                        Assign(SymbolRef('ib'), ArrayRef(SymbolRef('computation_seq'), SymbolRef('q'))),
                        self.do_tilerow(format, b_m, b_n, b_transpose, basis)
                    ]
                )
            ]
        )

    def bcsr_spmv_stanzas(self, format, b_m, b_n, b_transpose, browptr_comp, bcolidx_comp, basis):
        params = '\nconst struct bcsr_t *__restrict__ A,\nconst value_t *__restrict__ x,\nvalue_t *__restrict__ y,'
        if basis == 1:
            params += '\nvalue_t coeff,'
        params += '\nindex_t mb,\nconst index_t *__restrict__ computation_seq,\nindex_t seq_len'

        return FunctionDecl(None, 'bcsr_spmv' + format +'_stanzas', [StringTemplate(params)], [
                StringTemplate('index_t q, ib, jb;'),
                self.init(browptr_comp, bcolidx_comp),
                For(Assign(SymbolRef('q'), Constant(0)),
                    Lt(SymbolRef('q'), SymbolRef('seq_len')),
                    AddAssign(SymbolRef('q'), Constant(2)),
                    For(Assign(SymbolRef('ib'), ArrayRef(SymbolRef('computation_seq'), SymbolRef('q'))),
                        Lt(SymbolRef('ib'), ArrayRef(SymbolRef('computation_seq'), Add(SymbolRef('q'), Constant(1)))),
                        PostInc(SymbolRef('ib')),
                        self.do_tilerow(format, b_m, b_n, b_transpose, basis)
                    )
                )
            ]
        )

    def epilogue_dispatch(self, b_m, b_n, b_transpose, browptr_comp, bcolidx_comp, basis):
        return FileTemplate('../templates/' + ('epilogue_newton' if basis else 'epilogue_powers'),
                                {
                                    'b_m': Constant(b_m),
                                    'b_n': Constant(b_n),
                                    'b_transpose': Constant(b_transpose),
                                    'browptr_comp': Constant(browptr_comp),
                                    'bcolidx_comp': Constant(bcolidx_comp)
                                }
                            )


class Akx(object):
    """
    A class for generating jit akx c source code.
    """

    def __init__(self):
        """Instantiate translator."""
        self.c_apply_all = AkxGenerator(None)

    def __call__(self, b_m, b_n, b_transpose, browptr_comp, bcolidx_comp, basis):
        """Apply the operator to the arguments via a generated function."""
        return self.c_apply_all(b_m, b_n, b_transpose, browptr_comp, bcolidx_comp, basis)

    def nothing(self):
        pass


# ---------------------------------------------------------------------------
# User code


def main():

    c_akx = Akx()
    c_akx(4, 4, 0, 4, 4, 0)

if __name__ == '__main__':
    main()