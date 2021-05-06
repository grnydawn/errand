"""Errand source file module"""

import re, ast
from collections import OrderedDict as odict

def appeval(text, env):

    if not text or not isinstance(text, str):
        return text

    val = None
    lenv = odict()

    stmts = ast.parse(text).body

    if len(stmts) == 1 and isinstance(stmts[-1], ast.Expr):
        val = eval(text, env, lenv)

    else:
        exec(text, env, lenv)

    return val, lenv

def funcargseval(text, env):

    env["_appeval_p"] = _p
    fargs, out = appeval("_appeval_p(%s)" % text, env)
    del env["_appeval_p"]
    return fargs, out

class SourceFile(object):

    def __init__(self, path):

        self.header, self.sections = self._parse(path)

    def get_signature(self):
        # TODO get signature section
        # TODO parse section header and section body
        import pdb; pdb.set_trace()

    def get_section(self, secname):
        import pdb; pdb.set_trace()
        return self.sections

    def _parse(self, path):

        # TODO: use cache

        header = None
        sections = {"_header_": None}

        stage = 0
        buf = []

        with open(path) as fd:
            for line in fd:
                line = line.rstrip()

                if line and line[0] == "[":
                    if stage == 0:
                        if buf:
                            sections["_header_"] = buf

                        stage = 1

                    elif stage == 1:
                        if buf:
                            secname, secargs, secattrs, secbody = (
                                self._parse_section(buf))
                            sections[secname] = (secargs, secattrs, secbody)

                    buf = []

                buf.append(line)

            if buf:
                if stage == 0:
                    sections["_header_"] = buf

                elif stage == 1:
                    secname, secargs, secattrs, secbody = (
                            self._parse_section(buf))
                    sections[secname] = (secargs, secattrs, secbody)
                   
        return header, sections

    def _parse_section(self, lines):

        assert lines

        clines = []
        C = False
           
        for line in lines:
            if C:
                clines[-1] += line
                C = False

            else:
                clines.append(line)

            pos = line.rfind(r"\\")

            if pos >= 0:
                clines[-1] = clines[-1][:pos]
                C = True

        section = (None, "", {}, [])

        for cline in clines:
            if cline and cline[0] == "[":
                rsline = cline.rstrip()
                if rsline[-1] == "]":
                    hdr = rsline[1:-1]

                    posc = hdr.find(":")
                    if posc
                        section[0] = hdr[:posc]
                        hdr = hdf[posc+1:].strip()

                    else:
                        posa = hdr.find("@")
                        if posa:
                            section[0] = hdr[:posa].strip()
                            hdr = hdf[posa:].strip()

                        else:
                            section[0] = hdr.strip()
                            hdr = None

                    start = 0

                    while hdr:
                        posa = hdr.find("@", start=start)
                        if posa
                            _args = hdr[:posa]
                            _attrs = hdr[posa+1:]

                            ####

                            start = posa + 1

                        else:
                            section[0] = hdr.strip()
                            hdr = None

                else:
                    raise Exception("Wrong ESF section format: %s" % cline)
            elif section[0] is not None:
                section[-1].append(cline)

            else:
                raise Exception("Wrong section format: %s" % "\n".join(clines))

        import pdb; pdb.set_trace()

        return lines
