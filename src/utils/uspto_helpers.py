import pandas as pd
from .molec_smiles_helpers import remove_atom_mapping

def parse_uspto_reaction(rxn_smiles):
    """
    Parse a USPTO reaction SMILES into input and product molecules.

    Args:
        rxn_smiles: Reaction SMILES in reactants>reagents>products format.

    Returns:
        Dictionary containing cleaned reactant and product SMILES, or
        None if parsing fails.
    """
    parts = rxn_smiles.split(">")

    if len(parts) != 3:
        return None

    reactants, reagents, products = parts

    if reagents.strip():
        input_smiles = reactants + "." + reagents
    else:
        input_smiles = reactants

    input_smiles = remove_atom_mapping(input_smiles)
    product_smiles = remove_atom_mapping(products)

    if input_smiles is None or product_smiles is None:
        return None

    return {
        "reactants": input_smiles,
        "product": product_smiles
    }


def load_uspto_csv(csv_path, limit=None):
    """
    Load and parse reaction records from a USPTO CSV file.

    Args:
        csv_path: Path to the USPTO CSV file.
        limit: Maximum number of parsed reactions to return.

    Returns:
        List of parsed reaction dictionaries.
    """
    df = pd.read_csv(csv_path)

    reaction_data = []

    for rxn in df["reactants>reagents>production"]:
        parsed = parse_uspto_reaction(rxn)

        if parsed is not None:
            reaction_data.append(parsed)

        if limit is not None and len(reaction_data) >= limit:
            break

    return reaction_data
