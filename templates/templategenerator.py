import sys

from mako.template import *
import os

thisdir = os.path.dirname(__file__)

template_powers = Template(filename=os.path.join(thisdir, 'akx-powers.tpl'))

try:
    temp = template_powers.render(variants=set(eval(sys.argv[1])), basis=0)
    generated = open('generated', 'w')
    generated.write('-' * 5 + ' C generated from akx-powers.tpl with: '+ '-' * 6 + '\n')
    generated.write('variants = ' + sys.argv[1] + '\n')
    generated.write('-' * 50 +'\n\n')
    generated.write(temp)
    generated.close()
except (TypeError, NameError, IndexError):
    print('\nPlease enter a command line argument of the form:\n'+
          '\'([(b_m, b_n, b_transpose, browptr_comp, bcolidx_comp)])\'\n'+
          'where the above five variables are replaced with integers.\n')


