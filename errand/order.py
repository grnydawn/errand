"""Errand order module


"""

import os

from errand.util import parse_literal_args, appeval

# NOTE: _header_ should have only one section
# NOTE: Other sections has a list of sections under a same section name
# NOTE: section enable is evaluated at loading the order file

class Section(object):

    def __init__(self, arg, attr, body):

        self.arg = arg
        self.attr = attr
        self.body = body

    def __str__(self):
            return "\n".join(self.body)

    def isenabled(self, env):

        return (not self.attr or ("enable" not in self.attr) or
            appeval(self.attr["enable"], env)[0])


class HeaderSection(Section):
    pass


class SignatureSection(Section):

    def get_argnames(self):

        inargs = []
        outargs = []

        s1 = self.arg.split("->", 1)

        if len(s1) > 1:
            inargs = [s.strip() for s in s1[0].split(",")]
            outargs = [s.strip() for s in s1[1].split(",")]

        elif len(s1) == 1:
            inargs = [s.strip() for s in s1[0].split(",")]

        return (inargs, outargs)


class SectionList(object):

    def __init__(self, secs=None):

        if secs is None:
            secs = []

        self.sections = secs

    def select_one(self, env):

        for sec in self.sections:
            if sec.isenabled(env):
                return sec

    def select_many(self, env):

        secs = []

        for sec in self.sections:
            if sec.isenabled(env):
                secs.append(sec)

        return secs


class Order(object):

    def __init__(self, order, env):

        self._env = env

        if isinstance(order, Order):
            self.header = order.header
            self.sections = order.sections

        elif os.path.isfile(order):

            with open(order) as fd:
                self.header, self.sections = self._parse(fd.readlines())

        elif isinstance(order, str):
            self.header, self.sections = self._parse(order.split("\n"))

        else:
            raise Exception("Wrong order: %s" % str(order))

        if self.header.body:
            val, lenv = appeval(str(self.header), self._env)
            self._env.update(lenv)

    def _parse(self, lines):

        header = HeaderSection("", None, [])
        sections = {}

        stage = 0
        buf = []

        for line in lines:
            line = line.rstrip()

            if line and line[0] == "[":
                if stage == 0:
                    if buf:
                        header.body.extend(buf)

                    stage = 1

                elif stage == 1:
                    if buf:
                        for name, arg, attr, body in self._parse_section(buf):
                            if name not in sections:
                                slist = SectionList()
                                sections[name] = slist

                            else:
                                slist = sections[name]

                            #section.append((arg, attr, body))

                            if name == "signature":
                                slist.sections.append(SignatureSection(arg, attr, body))

                            else:
                                slist.sections.append(Section(arg, attr, body))

                buf = []

            buf.append(line)

        if buf:
            if stage == 0:
                header.body.extend(buf)

            elif stage == 1:
                for name, arg, attr, body in self._parse_section(buf):
                    if name not in sections:
                        slist = SectionList()
                        sections[name] = slist

                    else:
                        slist = sections[name]

                    if name == "signature":
                        slist.sections.append(SignatureSection(arg, attr, body))

                    else:
                        slist.sections.append(Section(arg, attr, body))

        return header, sections

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

            pos = clines[-1].rfind("\\")

            if pos >= 0 and not clines[-1][pos+1:].strip():
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
                                #parsed = ast.parse(_attrs)
                                if section[0]:
                                    section[1] = _args

                                else:
                                    section[0] = _args

                                _, section[2] = parse_literal_args(_attrs)
                                break

                            except SyntaxError as err:
                                start = posa + 1

                            else:
                                raise

                        else:
                            if hdr:
                                if section[0]:
                                    section[1] = hdr

                                else:
                                    section[0] = hdr.strip()

                            hdr = None

                else:
                    raise Exception("Wrong ESF section format: %s" % cline)

            elif section[0] is not None:
                section[-1].append(cline)

            else:
                raise Exception("Wrong section format: %s" % "\n".join(clines))

        output = []

        if section[0] is not None:
            for secname in section[0].split(","):
                newsec = []
                newsec.append(secname.strip())
                newsec += section[1:]
                output.append(newsec)

        return output

    def get_argnames(self):

        if "signature" in self.sections:

            sigsec = self.sections["signature"].select_one(self._env) 

            if isinstance(sigsec, Section):
                return sigsec.get_argnames()
   
        return ([], [])

    def get_targetnames(self):

        tnames = []

        for secname, slist in self.sections.items():
            if secname.startswith("_") or secname == "signature":
                continue

            sec = slist.select_one(self._env)

            if isinstance(sec, Section):
                tnames.append(secname)

        return tnames

    def get_section(self, name):

        if name in self.sections:
            return self.sections[name].select_one(self._env)
