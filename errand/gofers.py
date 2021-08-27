"""Errand gofers module


"""

class Gofers(object):
    """Errand gofers class

"""
    def __init__(self, *sizes):

        self._norm_sizes(*sizes)

    def _norm_sizes(self, *sizes):

        def _n(s):
            
            if isinstance(s, int):
                return ((s,1,1))

            elif isinstance(s, (list, tuple)):
                l = len(s)
                if l >= 3:
                    return tuple(s[:3])

                elif l == 2:
                    return (s[0], s[1], 1)

                elif l == 1:
                    return (s[0], 1, 1)

                else:
                    raise Exception("Wrong number of size dimension: %d" % l)
            else:
                raise Exception("Unsupported size type: %s" % str(s))

        # (members, teams, assignments)
        if len(sizes) == 3:
            self.sizes = [_n(sizes[1]), _n(sizes[0]), _n(sizes[2])]

        # (members, teams)
        elif len(sizes) == 2:
            self.sizes = [_n(sizes[1]), _n(sizes[0]), _n(1)]

        # (members)
        elif len(sizes) == 1:
            self.sizes = [_n(1), _n(sizes[0]), _n(1)]

        else:
            raise Exception("Wrong # of Gofers initialization: %d" % len(sizes))

    def run(self, workshop):

        return workshop.open(*self.sizes)

