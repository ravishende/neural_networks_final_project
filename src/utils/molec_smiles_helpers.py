from rdkit import Chem


def smiles_to_mol(smiles):
    """
    Convert a SMILES string to an RDKit Mol object.

    Args:
        smiles: SMILES string representation of a molecule.

    Returns:
        RDKit Mol object, or None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return None
    return mol


def randomize_smiles(smiles):
    """
    Randomize a SMILES string.

    Args:
        smiles: SMILES string representation of a molecule.

    Returns:
        randomized SMILES string, or None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return smiles

    return Chem.MolToSmiles(
        mol,
        canonical=False,
        doRandom=True,
        isomericSmiles=True,
    )

def randomize_smiles_components(smiles):
    """
    Randomize the molecules in a multi-molecule SMILES string (separated by '.')

    Args:
        smiles: SMILES string representation of several molecules.

    Returns:
        randomized SMILES string where each molecule was individually randomized.
    """
    randomized = []
    for component in smiles.split("."):
        random_smiles = randomize_smiles(component)
        if random_smiles is None:
            randomized.append(component)
        else:
            randomized.append(random_smiles)
    return ".".join(randomized)


def print_mol_basic_info(smiles):
    """
    Print basic RDKit molecule information and return the molecule.

    Args:
        smiles: SMILES string representation of a molecule.

    Returns:
        RDKit Mol object, or None if the SMILES is invalid.
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    print("Original SMILES:", smiles)
    print("Canonical SMILES:", canonical_smiles)
    print("Number of atoms:", mol.GetNumAtoms())
    print("Number of bonds:", mol.GetNumBonds())
    return mol


def remove_atom_mapping(smiles):
    """
    Remove atom-mapping numbers from a SMILES string.

    Args:
        smiles: Atom-mapped SMILES string.

    Returns:
        Canonical SMILES without atom mappings, or None if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def parse_reaction_line(line):
    reaction = line.strip().split()[0]
    src, tgt = reaction.split(">>")

    src = remove_atom_mapping(src)
    tgt = remove_atom_mapping(tgt)

    if src is None or tgt is None:
        return None

    return src, tgt