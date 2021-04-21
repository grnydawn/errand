"""Errand source file module"""


class SourceFile(object):

    def __init__(self, path):

        self.header, self.sections = self._parse(path)

    def _parse(self, path):

        # TODO: use cache

        attrs = {}
        stmts = []

        header = (attrs, stmts)
        sections = []

        return header, sections
