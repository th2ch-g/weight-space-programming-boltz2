from rdkit import Chem


def atom_names(mol):
    ranks = list(Chem.CanonicalRankAtoms(mol))
    symbols = [atom.GetSymbol().upper() for atom in mol.GetAtoms()]
    names = [f"{symbol}{rank + 1}" for symbol, rank in zip(symbols, ranks, strict=True)]
    if all(len(name) <= 4 for name in names):
        return names
    counts = {}
    for index in sorted(range(len(ranks)), key=ranks.__getitem__):
        symbol = symbols[index]
        counts[symbol] = counts.get(symbol, 0) + 1
        names[index] = f"{symbol}{counts[symbol]}"
    if max(map(len, names)) > 4 or len(set(names)) != len(names):
        raise ValueError("Cannot assign unique PDB-width atom names")
    return names
