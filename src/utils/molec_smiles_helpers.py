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

    return Chem.MolToSmiles(mol, canonical=True)


