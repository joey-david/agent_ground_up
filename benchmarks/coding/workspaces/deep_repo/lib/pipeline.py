from .mod1 import helper1
from .mod2 import helper2
from .mod3 import helper3
from .mod4 import helper4
from .mod5 import helper5
from .mod6 import helper6
from .mod7 import helper7

def run(x):
    for fn in (helper1, helper2, helper3, helper4, helper5, helper6, helper7):
        x = fn(x)
    return x
