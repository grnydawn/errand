'''Errand utility module'''

import os, ast
from collections import OrderedDict
import subprocess as subp

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


def which(pgm):
    path=os.getenv('PATH')
    for p in path.split(os.path.pathsep):
        p=os.path.join(p,pgm)
        if os.path.exists(p) and os.access(p,os.X_OK):
            return p


def parse_literal_args(expr):

    lv = []
    lk = {}

    expr_items = expr.split(",")
    text = ""

    while expr_items:

        expr_item = expr_items.pop(0).strip()

        if not expr_item:
            continue

        if text:
            text = text + "," + expr_item

        else:
            text = expr_item

        #if not text:
        #    continue

        try:
            tree = ast.parse("func({0})".format(text))
            args = tree.body[0].value.args
            keywords = tree.body[0].value.keywords

            if len(args) > 0 and len(keywords):
                raise UsageError("Both of args and keywords are found"
                                 " during argument parsing.")
            text = text.strip()

            if not text:
                continue

            if len(args) > 0:
                lv.append(text)

            elif len(keywords) > 0:
                key, val = text.split("=", 1)

                if val:
                    lk[key] = val

            text = ""

        except Exception:
            pass

    #if not lv and not lk:
    #    lv.append(expr)

    return lv, lk


def shellcmd(cmd, shell=True, stdout=subp.PIPE, stderr=subp.PIPE,
             check=False): 

    return subp.run(cmd, shell=shell, stdout=stdout, stderr=stderr,
                    check=check)

