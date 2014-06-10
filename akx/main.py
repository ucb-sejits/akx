"""
Parses the python AST below, transforms it to C, JITs it, and runs it.
"""

import logging

logging.basicConfig(level=20)

import numpy as np

from ctree.frontend import get_ast
from ctree.c.nodes import *
from ctree.c.types import *
from ctree.dotgen import to_dot
from ctree.transformations import *
from ctree.jit import LazySpecializedFunction
from ctree.types import get_ctree_type

# ---------------------------------------------------------------------------
# Specializer code


class OpTranslator(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        """
        Analyze arguments and return a 'subconfig', a hashable object
        that classifies them. Arguments with identical subconfigs
        might be processed by the same generated code.
        """
        A = args[0]
        B = args[1]
        return {
            'A_len': len(A[0]),
            'A_dtype': A.dtype,
            'A_ndim': A.ndim,
            'A_powers': len(A),
            'A_shape': A.shape,
            'B_len': len(B),
            'B_dtype': B.dtype,
            'B_ndim': B.ndim,
            'B_shape': B.shape,
        }

    def transform(self, py_ast, program_config):
        """
        Convert the Python AST to a C AST according to the directions
        given in program_config.
        """
        arg_config, tuner_config = program_config
        len_A = arg_config['A_len']
        A_dtype = arg_config['A_dtype']
        A_ndim = arg_config['A_ndim']
        A_shape = arg_config['A_shape']
        A_powers = arg_config['A_powers']

        len_B = arg_config['B_len']
        B_dtype = arg_config['B_dtype']
        B_ndim = arg_config['B_ndim']
        B_shape = arg_config['B_shape']

        inner_type_A = get_ctree_type(A_dtype)
        array_type_A = NdPointer(A_dtype, A_ndim, A_shape)

        inner_type_B = get_ctree_type(B_dtype)
        array_type_B = NdPointer(B_dtype, B_ndim, B_shape)

        apply_one_typesig = FuncType(inner_type_A, [inner_type_A, inner_type_B])

        tree = CFile("generated", [
            py_ast.body[0],
            FunctionDecl(
                Void(), "apply_all",
                params=[SymbolRef("A", array_type_A), SymbolRef("B", array_type_B)],
                defn=[
                    For(Assign(SymbolRef("k", Int()), Constant(1)),
                        Lt(SymbolRef("k"), Constant(A_powers)),
                        PostInc(SymbolRef("k")),
                        [
                            For(Assign(SymbolRef("i", Int()), Constant(0)),
                                Lt(SymbolRef("i"), Constant(len_A)),
                                PostInc(SymbolRef("i")),
                                [
                                    Assign(ArrayRef(SymbolRef("A"), Add(Mul(Constant(len_A), SymbolRef("k")), SymbolRef("i"))),
                                           FunctionCall(SymbolRef("apply"), [ArrayRef(SymbolRef("A"),
                                                        Add(Mul(Constant(len_A), Sub(SymbolRef("k"), Constant(1))),
                                                        SymbolRef("i"))),ArrayRef(SymbolRef("B"), SymbolRef("i"))]))
                                ]
                            )
                        ]
                    ),
                ]
            ),
        ])

        tree = PyBasicConversions().visit(tree)

        apply_one = tree.find(FunctionDecl, name="apply")
        apply_one.set_static().set_inline()
        apply_one.set_typesig(apply_one_typesig)

        entry_point_typesig = tree.find(FunctionDecl, name="apply_all").get_type().as_ctype()

        return Project([tree]), entry_point_typesig


class ArrayOp(object):
    """A class for managing operation on elements in numpy arrays."""

    def __init__(self):
        """Instantiate translator."""
        self.c_apply_all = OpTranslator(get_ast(self.apply), "apply_all")

    def __call__(self, A, B):
        """Apply the operator to the arguments via a generated function."""
        return self.c_apply_all(A, B)


# ---------------------------------------------------------------------------
# User code

class Akx(ArrayOp):
    """Multiplies elements of two arrays."""

    def apply(n, m):
        return n * m


def py_akx(vecs, matrix):
    for i in xrange(1, len(vecs)):
        vecs[i] = matrix * vecs[i - 1]


def main():
    c_akx = Akx()

    #only the first row of vecs is relevant, all other rows are
    #overwritten by A to progressively higher powers times x.
    actualvecs_i = np.random.randint(-5, 5, (3, 10))
    expectedvecs_i = np.copy(actualvecs_i)

    actualmatrix_i = np.random.randint(-5, 5, 10)
    expectedmatrix_i = np.copy(actualmatrix_i)

    c_akx(actualvecs_i, actualmatrix_i)
    py_akx(expectedvecs_i, expectedmatrix_i)

    np.testing.assert_array_equal(actualvecs_i, expectedvecs_i)

    actualvecs_f = 10 * np.random.random((3, 10)) - 5
    expectedvecs_f = np.copy(actualvecs_f)

    actualmatrix_f = 10 * np.random.random(10) - 5
    expectedmatrix_f = np.copy(actualmatrix_f)

    c_akx(actualvecs_f, actualmatrix_f)
    py_akx(expectedvecs_f, expectedmatrix_f)

    np.testing.assert_array_equal(actualvecs_f, expectedvecs_f)

    print("Success.")


if __name__ == '__main__':
    main()