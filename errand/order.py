"""Errand order module


"""

import os, ast

from errand.util import funcargseval

class Order(object):

    def __init__(self, order):

        if isinstance(order, Order):
            self.sections = order.sections

        elif os.path.isfile(order):

            with open(order) as fd:
                self.sections = self._parse(fd.readlines())

        elif isinstance(order, str):
            self.sections = self._parse(order.split("\n"))

        else:
            raise Exception("Wrong order: %s" % str(order))


    def _parse(self, lines):

        header = None
        sections = {"_header_": None}

        stage = 0
        buf = []

        for line in lines:
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
                   
        return sections

    def _parse_section(self, lines):

        assert lines

        clines = []
        C = False
        lenv = None
           
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

        # sec name(str), sec args(str), control arguments(dict), section body(list of strings)
        section = [None, "", None, []]

        for cline in clines:
            if cline and cline[0] == "[":
                rsline = cline.rstrip()
                if rsline[-1] == "]":
                    hdr = rsline[1:-1]

                    posc = hdr.find(":")
                    if posc>=0:
                        section[0] = hdr[:posc].strip()
                        hdr = hdr[posc+1:].strip()

                    start = 0

                    while hdr:
                        posa = hdr.find("@", start)

                        if posa >= 0:
                            _args = hdr[:posa].strip()
                            _attrs = hdr[posa+1:].strip()

                            try:
                                parsed = ast.parse(_attrs)
                                if section[0]:
                                    section[1] = _args

                                else:
                                    section[0] = _args

                                _, section[2] = funcargseval(_attrs, lenv)
                                break

                            except SyntaxError as err:
                                start = posa + 1

                            else:
                                raise

                        else:
                            if hdr:
                                if section[0]:
                                    raise Exception("Wrong section header format: %s" % hdr)

                                else:
                                    section[0] = hdr.strip()

                            hdr = None

                else:
                    raise Exception("Wrong ESF section format: %s" % cline)

            elif section[0] is not None:
                section[-1].append(cline)

            else:
                raise Exception("Wrong section format: %s" % "\n".join(clines))

        return section

    def get_argnames(self):
        return ("a", "b"), ("c",)