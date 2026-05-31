import torch
from rdkit import Chem
from torch_geometric.data import Data

def atom_features(atom):
    """
    Extract atom-level features for graph construction.

    Args:
        atom: RDKit Atom object.

    Returns:
        List containing atomic number, degree, formal charge, and aromaticity.
    """
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic())
    ]


def bond_features(bond):
    """
    Extract bond-level features for graph construction.

    Args:
        bond: RDKit Bond object.

    Returns:
        List of 1s or 0s to represent [is single bond, is double bond, is triple bond, is aromatic bond, is conjugated, is in ring]
    """
    bond_type = bond.GetBondType()
    return [
        int(bond_type == Chem.rdchem.BondType.SINGLE),
        int(bond_type == Chem.rdchem.BondType.DOUBLE),
        int(bond_type == Chem.rdchem.BondType.TRIPLE),
        int(bond_type == Chem.rdchem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing())
    ]


def smiles_to_pyg_data(reactant_smiles):
    """
    Convert a SMILES string into a PyTorch Geometric graph.

    Args:
        reactant_smiles: SMILES string representing a molecule.

    Returns:
        PyTorch Geometric Data object containing node and edge features.

    Raises:
        ValueError: If the SMILES string is invalid.
    """
    mol = Chem.MolFromSmiles(reactant_smiles)
    if mol is None:
        raise ValueError(f"Invalid reactant SMILES: {reactant_smiles}")
    x = []
    for atom in mol.GetAtoms():
        x.append(atom_features(atom))

    x = torch.tensor(x, dtype=torch.float)

    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()

        bond_feat = bond_features(bond)

        edge_index.append([start, end])
        edge_index.append([end, start])

        edge_attr.append(bond_feat)
        edge_attr.append(bond_feat)

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )