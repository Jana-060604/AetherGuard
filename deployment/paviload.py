import pickle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Redefine the GNN model class
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize Model with Correct Input Size (47 as per your saved model)
model = GNN(in_channels=47, hidden_channels=16, out_channels=2)

# Load model state dictionary
with open(r"C:\\Users\\janam\\model\\gnn_model.pkl", "rb") as file:
    model_state_dict = pickle.load(file)

# Load the weights into the model
model.load_state_dict(model_state_dict)
model.eval()

print("Model successfully loaded!")
