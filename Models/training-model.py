import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd
import networkx as nx
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("/content/train_data.csv")

# Drop unnecessary columns
df = df.drop(columns=["Unnamed: 0", "Index"], errors="ignore").fillna(0)

# Map unique addresses to node indices
address_mapping = {addr: idx for idx, addr in enumerate(df["Address"].unique())}
df["Node_ID"] = df["Address"].map(address_mapping)

# Convert categorical features to numeric
for col in df.select_dtypes(include=["object"]).columns:
    if col not in ["Address", "FLAG"]:  # Ignore target and address
        df[col] = df[col].astype(str)  # Ensure uniform data type
        df[col] = LabelEncoder().fit_transform(df[col])

# Create edges based on transactions (Modify as needed)
edges = list(zip(df["Node_ID"], df["Node_ID"].shift(-1, fill_value=df["Node_ID"].iloc[0])))

# Convert to NetworkX Graph
G = nx.Graph()
G.add_edges_from(edges)

# Convert features to tensor
node_features = df.drop(columns=["Address", "FLAG", "Node_ID"], errors="ignore").values.astype(float)
node_labels = df["FLAG"].values.astype(int)

# Convert to PyG Data object
graph_data = from_networkx(G)
graph_data.x = torch.tensor(node_features, dtype=torch.float)
graph_data.y = torch.tensor(node_labels, dtype=torch.long)

# Define GNN Model
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

# Initialize model
model = GNN(in_channels=graph_data.x.shape[1], hidden_channels=16, out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss_fn = torch.nn.NLLLoss()

# Train the GNN Model
def train():
    model.train()
    optimizer.zero_grad()
    out = model(graph_data.x, graph_data.edge_index)
    loss = loss_fn(out, graph_data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

# Train for 100 epochs
for epoch in range(100):
    loss = train()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Save trained model
torch.save(model.state_dict(), "gnn_model.pth")
print("Model saved as gnn_model.pth")

# Evaluate model
model.eval()
with torch.no_grad():
    pred = model(graph_data.x, graph_data.edge_index).argmax(dim=1)
accuracy = (pred == graph_data.y).sum().item() / graph_data.y.size(0)

print(f"Model Accuracy: {accuracy:.4f}")
