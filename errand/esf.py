"""Errand source file module"""

import re

class SourceFile(object):

    def __init__(self, path):

        self.header, self.sections = self._parse(path)

    def _parse(self, path):

        # TODO: use cache

        header = None
        sections = []

        stage = 0
        buf = []

        with open(path) as fd:
            for line in fd:
                if line and line[0] == "[":
                    if stage == 0:
                        if buf:
                            header = self._parse_header(buf)

                        stage = 1

                    elif stage == 1:
                        if buf:
                            sections.append(self._parse_section(buf))

                    buf = []

                buf.append(line)

            if buf:
                if stage == 0:
                    header = self._parse_header(buf)

                elif stage == 1:
                    sections.append(self._parse_section(buf))
                    
        return header, sections

    def _parse_header(self, lines):

        pass

    def _parse_section(self, lines):
        import pdb; pdb.set_trace()
