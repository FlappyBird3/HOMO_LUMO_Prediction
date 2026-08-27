
"""
Convert a SMILES string into the tensors the bond-only model expects.

The 11 atom features must be built exactly the way QM9 built them, in the same
column order, or predictions degrade silently. Column layout:

    0-4   one-hot element: H, C, N, O, F
    5     atomic number
    6     is aromatic (0 or 1)
    7-9   one-hot hybridization: sp, sp2, sp3
    10    number of attached hydrogens

QM9 stores hydrogens as explicit atoms, so Chem.AddHs is required before any
feature is read. Skipping it changes the atom count and produces a different
graph.
"""

import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType, HybridizationType
from torch_geometric.data import Data

ALLOWED_ELEMENTS = {"H", "C", "N", "O", "F"}
MAX_HEAVY_ATOMS = 9

TYPE_INDEX = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
BOND_INDEX = {
    BondType.SINGLE: 0,
    BondType.DOUBLE: 1,
    BondType.TRIPLE: 2,
    BondType.AROMATIC: 3,
}
HYBRID_INDEX = {
    HybridizationType.SP: 0,
    HybridizationType.SP2: 1,
    HybridizationType.SP3: 2,
}


class InvalidMolecule(Exception):
    """Raised when the input cannot be used, with a message safe to show a user."""


def parse_smiles(smiles):
    """Parse and validate a SMILES string. Returns an RDKit molecule with
    explicit hydrogens, or raises InvalidMolecule with a readable reason."""

    if not smiles or not smiles.strip():
        raise InvalidMolecule("No molecule entered.")

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise InvalidMolecule("Could not parse that SMILES string.")

    heavy = mol.GetNumAtoms()
    if heavy > MAX_HEAVY_ATOMS:
        raise InvalidMolecule(
            "This molecule has %d heavy atoms. The model was trained on "
            "molecules with at most %d, so its prediction would be unreliable."
            % (heavy, MAX_HEAVY_ATOMS)
        )

    found = {a.GetSymbol() for a in mol.GetAtoms()}
    outside = found - ALLOWED_ELEMENTS
    if outside:
        raise InvalidMolecule(
            "Contains %s. The model was trained only on H, C, N, O and F."
            % ", ".join(sorted(outside))
        )

    if any(a.GetFormalCharge() != 0 for a in mol.GetAtoms()):
        raise InvalidMolecule(
            "Charged species are outside the training set, which contains "
            "only neutral molecules."
        )

    mol = Chem.AddHs(mol)      # QM9 has explicit hydrogens
    Chem.Kekulize(mol, clearAromaticFlags=True)
    
    if mol.GetNumAtoms() < 2:
        raise InvalidMolecule("Needs at least two atoms.")

    return mol


def mol_to_data(mol):
    """Build the three tensors the model consumes from an RDKit molecule."""

    n = mol.GetNumAtoms()
    x = torch.zeros(n, 11)

    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        x[i, TYPE_INDEX[symbol]] = 1.0                    # cols 0-4
        x[i, 5] = atom.GetAtomicNum()                     # col 5
        x[i, 6] = 1.0 if atom.GetIsAromatic() else 0.0    # col 6

        # Columns 7-9 (hybridization) are all zero in the pre-processed QM9 file
        # this model was trained on, so they are left zero here to match.

        # Hydrogens are explicit atoms, so count neighbours rather than
        # asking RDKit for an implicit hydrogen count.
        x[i, 10] = sum(1 for nb in atom.GetNeighbors() if nb.GetSymbol() == "H")

    senders, receivers, bond_rows = [], [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        order = BOND_INDEX.get(bond.GetBondType())
        if order is None:
            raise InvalidMolecule("Unsupported bond type in this molecule.")

        row = [0.0, 0.0, 0.0, 0.0]
        row[order] = 1.0

        # Every bond is stored in both directions.
        senders += [i, j]
        receivers += [j, i]
        bond_rows += [row, row]

    if not senders:
        raise InvalidMolecule("This molecule has no bonds.")

    edge_index = torch.tensor([senders, receivers], dtype=torch.long)
    edge_attr = torch.tensor(bond_rows, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def smiles_to_data(smiles):
    """SMILES string in, Data object out."""
    return mol_to_data(parse_smiles(smiles))


def describe(mol):
    """Human-readable structure for the frontend to display."""
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]

    counts = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    formula = "".join(
        el + (str(counts[el]) if counts[el] > 1 else "")
        for el in ["C", "H", "N", "O", "F"]
        if el in counts
    )

    bonds = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bonds.append(
            {
                "source": i,
                "target": j,
                "order": str(bond.GetBondType()).lower(),
            }
        )

    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()

    return {
        "formula": formula,
        "atoms": [{"index": i, "element": s} for i, s in enumerate(symbols)],
        "bonds": bonds,
        "num_atoms": n_atoms,
        "num_bonds": n_bonds,
        "num_rings": n_bonds - n_atoms + 1,     # connected graph
    }
