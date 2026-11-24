__all__ = [
    "Intensity",
    "Spectrum",
    "Molecule",
    "MoleculeDict",
    "MoleculeLineList",
    "MoleculeLine",
    "ParameterSet",
    "QuantumField",
    "QuantumStateSchema",
    "GenericDelimitedSchema",
    "QuantumStateRegistry",
]

_LAZY_IMPORTS = {
    "Intensity": ".Intensity",
    "Spectrum": ".Spectrum",
    "Molecule": ".Molecule",
    "MoleculeDict": ".MoleculeDict",
    "MoleculeLineList": ".MoleculeLineList",
    "MoleculeLine": ".MoleculeLine",
    "ParameterSet": ".ParameterSet",
    "QuantumField": ".QuantumStateSchema",
    "QuantumStateSchema": ".QuantumStateSchema",
    "GenericDelimitedSchema": ".QuantumStateSchema",
    "QuantumStateRegistry": ".QuantumStateSchema",
}

def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")