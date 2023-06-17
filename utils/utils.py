#! /usr/bin/env python
import torch.nn.functional as F
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def FT(x):
    return F.normalize(x.view(x.size(0), -1))