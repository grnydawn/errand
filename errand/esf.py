"""Errand source file module"""

import re

class SourceFile(object):

    def __init__(self, path):

        self.header, self.sections = self._parse(path)

    def get_signature(self):
        pass

    def get_section(self, secname):
        pass

    def _parse(self, path):

        # TODO: use cache

        header = None
        sections = []

        stage = 0
        buf = []

        with open(path) as fd:
            for line in fd:
                line = line.rstrip()

                if line and line[0] == "[":
                    if stage == 0:
                        if buf:
                            header = buf

                        stage = 1

                    elif stage == 1:
                        if buf:
                            sections.append(self._parse_section(buf))

                    buf = []

                buf.append(line)

            if buf:
                if stage == 0:
                    header = buf

                elif stage == 1:
                    sections.append(self._parse_section(buf))
                    
        return header, sections

    def _parse_section(self, lines):

        return lines
