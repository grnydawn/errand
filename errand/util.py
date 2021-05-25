'''Errand utility module'''

import ast
from collections import OrderedDict

exclude_list = ["exec", "eval", "breakpoint", "memoryview"]

errand_builtins = dict((k, v) for k, v in __builtins__.items()
                       if k not in exclude_list)
del exclude_list

def _p(*argv, **kw_str):
    return list(argv), kw_str


def appeval(text, env):

    if not text or not isinstance(text, str):
        return text

    val = None
    lenv = {}

    stmts = ast.parse(text).body

    if len(stmts) == 1 and isinstance(stmts[-1], ast.Expr):
        val = eval(text, env, lenv)

    else:
        exec(text, env, lenv)

    return val, lenv


def funcargseval(text, lenv):

    env = dict(errand_builtins)
    if isinstance(lenv, (dict, OrderedDict)):
        env.update(lenv)

    env["_appeval_p"] = _p
    fargs, out = appeval("_appeval_p(%s)" % text, env)

    return fargs
