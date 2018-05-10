""" a module for for utility functions """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



def gen_var_dict(list_var_names):
    """
    generate a dict of variables

    :param list_var_names: e.g. ['a', 'b', 'c'], note that the var must exit
    :return:  dict, e.g. {'a': a, 'b': b, 'c': c}
    """

    return {k: eval(k) for k in list_var_names}


class ObjDict(dict):
    """ class to access dict keys as attributes """

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)