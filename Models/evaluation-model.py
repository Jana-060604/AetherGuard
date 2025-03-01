# Load test dataset
test_df = pd.read_csv("/content/test_data.csv")

# Drop unnecessary columns
test_df = test_df.drop(columns=["Unnamed: 0", "Index"], errors="ignore").fillna(0)

# Map unique addresses using training data mapping (ensure consistency)
test_df["Node_ID"] = test_df["Address"].map(address_mapping).fillna(-1).astype(int)

# Remove unmapped nodes
test_df = test_df[test_df["Node_ID"] != -1]

# Convert categorical features to numeric
for col in test_df.select_dtypes(include=["object"]).columns:
    if col not in ["Address", "FLAG"]:
        test_df[col] = test_df[col].astype(str)
        test_df[col] = LabelEncoder().fit_transform(test_df[col])

# Create edges
test_edges = list(zip(test_df["Node_ID"], test_df["Node_ID"].shift(-1, fill_value=test_df["Node_ID"].iloc[0])))

# Convert to NetworkX Graph
test_G = nx.Graph()
test_G.add_edges_from(test_edges)

# Convert features to tensor
test_features = test_df.drop(columns=["Address", "FLAG", "Node_ID"], errors="ignore").values.astype(float)
test_labels = test_df["FLAG"].values.astype(int)

# Convert to PyG Data object
test_graph_data = from_networkx(test_G)
test_graph_data.x = torch.tensor(test_features, dtype=torch.float)
test_graph_data.y = torch.tensor(test_labels, dtype=torch.long)

# Load trained model
model.load_state_dict(torch.load("gnn_model.pth"))
model.eval()

# Evaluate model on test data
with torch.no_grad():
    test_pred = model(test_graph_data.x, test_graph_data.edge_index).argmax(dim=1)

# Calculate accuracy
test_accuracy = (test_pred == test_graph_data.y).sum().item() / test_graph_data.y.size(0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Display actual vs predicted labels
result_df = pd.DataFrame({"Actual": test_graph_data.y.tolist(), "Predicted": test_pred.tolist()})
print(result_df)
