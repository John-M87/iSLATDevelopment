__version__ = 'v5.03.04'

__all__ = [
    '__version__',
    'iSLAT',
]

def __getattr__(name):
    if name == "iSLAT":
        from .iSLATClass import iSLAT
        return iSLAT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")