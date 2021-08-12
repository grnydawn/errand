"""Errand gofers module


"""

class Gofers(object):
    """Errand gofers class

"""

    def __init__(self, *sizes):

        if len(sizes) == 2:
            self.sizes = sizes

        elif len(sizes) == 1:
            self.sizes = [1, sizes[0]]

        else:
            raise Exception("Wrong # of Gofers initialization: %d" % len(sizes))

    def run(self, workshop):

        return workshop.open(*self.sizes)

