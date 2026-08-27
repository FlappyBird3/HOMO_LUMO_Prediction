
"""
FastAPI backend for HOMO-LUMO gap prediction.

Run locally:
    pip install fastapi uvicorn
    uvicorn api:app --reload

Then open http://127.0.0.1:8000/docs for an interactive test page.

The model is loaded once at startup rather than per request, because loading
PyTorch weights takes a second or two and doing it per request would make every
prediction slow.
"""

import torch
from torch import nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing, global_mean_pool

from smiles_to_graph import parse_smiles, mol_to_data, describe, InvalidMolecule

# ------------------------------------------------------------------ constants
# These must match the values the model was trained with. Changing any of them
# means the saved weights will not load.

NODE_DIM = 11
EDGE_DIM = 4
HIDDEN = 128
LAYERS = 4

Y_MEAN = 7.0135      # from the training split
Y_STD = 1.4051

WEIGHTS = "best_model_bondonly.pt"

# ---------------------------------------------------------------- model classes
# Copied verbatim from the notebook. The class definitions must be identical or
# load_state_dict will fail on mismatched layer names.


class BondMessagePassing(MessagePassing):
    def __init__(self, hidden, edge_dim):
        super().__init__(aggr="add")
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden + edge_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x, edge_index, edge_attr):
        aggregated = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return x + self.update_mlp(torch.cat([x, aggregated], dim=-1))

    def message(self, x_i, x_j, edge_attr):
        return self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


class GapNet(nn.Module):
    def __init__(self, hidden=HIDDEN, layers=LAYERS):
        super().__init__()
        self.embed = nn.Linear(NODE_DIM, hidden)
        self.rounds = nn.ModuleList(
            BondMessagePassing(hidden, EDGE_DIM) for _ in range(layers)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def forward(self, data):
        x = self.embed(data.x)
        for round_ in self.rounds:
            x = round_(x, data.edge_index, data.edge_attr)
        molecule = global_mean_pool(x, data.batch)
        return self.head(molecule).squeeze(-1)


# ------------------------------------------------------------------- app setup

app = FastAPI(title="HOMO-LUMO Gap Prediction")

# Allows the React frontend to call this API from a different port or domain.
# Replace "*" with your deployed frontend URL once you have one.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

device = torch.device("cpu")     # free hosting tiers have no GPU
model = GapNet().to(device)
model.load_state_dict(torch.load(WEIGHTS, map_location=device, weights_only=True))
model.eval()


class PredictRequest(BaseModel):
    smiles: str


@app.get("/health")
def health():
    """Used to check the server is awake. Free hosts sleep after inactivity."""
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        mol = parse_smiles(req.smiles)
    except InvalidMolecule as e:
        return {"ok": False, "error": str(e)}

    try:
        data = mol_to_data(mol)
    except InvalidMolecule as e:
        return {"ok": False, "error": str(e)}

    with torch.no_grad():
        raw = model(Batch.from_data_list([data]).to(device))
        gap = float(raw.item() * Y_STD + Y_MEAN)

    structure = describe(mol)

    return {
        "ok": True,
        "smiles": req.smiles.strip(),
        "gap_ev": round(gap, 4),
        "structure": structure,
        "note": (
            "Approximates a B3LYP/6-31G(2df,p) DFT calculation. Test MAE on QM9 "
            "is 0.110 eV against a 1.078 eV mean-prediction baseline."
        ),
    }
