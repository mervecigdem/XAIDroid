import torch
import torch.nn.functional as F
import os
import pandas as pd
import re
import json
from openpyxl import load_workbook
from torch_geometric.explain import Explainer, AttentionExplainer, GNNExplainer
from torch_geometric.nn import GATv2Conv, global_mean_pool
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from helpers.gat_utilities import save_graph_results_to_excel


class CustomGATv2Conv(GATv2Conv):
    def forward(self, x, edge_index, return_attention_weights=False):
        out, (edge_index, alpha) = super().forward(x, edge_index, return_attention_weights=True)
        if return_attention_weights:
            return out, (edge_index, alpha)
        return out


class GATv2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout):
        super(GATv2, self).__init__()
        self.conv1 = CustomGATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_channels * heads, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, batch, return_attention_weights=False):
        x = x.float()
        x = F.dropout(x, p=self.dropout, training=self.training)
        if return_attention_weights:
            x, attention_weights = self.conv1(x, edge_index, return_attention_weights=True)
        else:
            x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        if return_attention_weights:
            return x, attention_weights
        return x


def initialize_model(in_channels, hidden_channels, out_channels, heads, dropout):
    return GATv2(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,
                 heads=heads, dropout=dropout)


def initialize_explainers(model):
    attention_explainer = Explainer(
        model=model,
        algorithm=AttentionExplainer(reduce='mean'),
        explanation_type='model',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='log_probs'
        )
    )

    gnn_explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='log_probs'
        )
    )

    return {
        "attention": attention_explainer,
        "gnn": gnn_explainer
    }


def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for data in loader:
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def test(model, explainers, loader, epoch, total_epochs, dataset_type, graph_classification_file, output_dir,
         api_dict_path, attention_saving):
    model.eval()
    all_preds, all_labels, graph_results, all_probs = [], [], [], []
    attention_weights_list = []
    is_final_epoch = (epoch + 1) == total_epochs

    for data in loader:
        labels = data.y.tolist()
        if is_final_epoch and dataset_type == "testing":
            with torch.no_grad():
                output, (edges, attention_weights) = model(data.x, data.edge_index, data.batch,
                                                           return_attention_weights=True)
                preds = output.argmax(dim=1).tolist()
                probs = F.softmax(output, dim=1)[:, 1].tolist()
                attention_weights_list = [(attention_weights, data.edge_index, data.batch)]
        else:
            with torch.no_grad():
                output = model(data.x, data.edge_index, data.batch)
                preds = output.argmax(dim=1).tolist()
                probs = F.softmax(output, dim=1)[:, 1].tolist()

        all_labels.extend(labels)
        all_preds.extend(preds)
        all_probs.extend(probs)

        if is_final_epoch and dataset_type == "testing":
            graph_results.extend(log_graph_results(data.name, labels, preds, probs))
            if attention_saving:
                for attention_weights, edges, batch_attr in attention_weights_list:
                    edge_masks = {}
                    for name, explainer in explainers.items():
                        explanation = explainer(data.x, data.edge_index, target=data.y, batch=data.batch)
                        edge_masks[name] = explanation.edge_mask

                    save_attentions(
                        attention_weights=attention_weights,
                        batch_attr=batch_attr,
                        edges=edges,
                        output_dir=output_dir,
                        api_dict_path=api_dict_path,
                        data_name=data.name,
                        api_ids=data.x[:, 0].long(),
                        edge_masks=edge_masks
                    )

    if is_final_epoch and dataset_type == "testing":
        save_graph_results_to_excel(graph_results, graph_classification_file)

    return compute_metrics(all_preds, all_labels)


def log_graph_results(graph_names, labels, preds, probs):
    """Logs graph names, predicted labels, actual labels, and probabilities."""
    return [f"{name},{label},{pred},{prob:.4f}" for name, label, pred, prob in zip(graph_names, labels, preds, probs)]


def compute_metrics(preds, labels):
    """Computes accuracy, precision, recall, F1 score, FPR, and FNR."""
    accuracy = (torch.tensor(preds) == torch.tensor(labels)).sum().item() / len(labels)
    precision = precision_score(labels, preds, average='binary', pos_label=1, zero_division=0)
    recall = recall_score(labels, preds, average='binary', pos_label=1, zero_division=0)
    f1 = f1_score(labels, preds, average='binary', pos_label=1, zero_division=0)

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return accuracy, precision, recall, f1, fpr, fnr


def save_attentions(attention_weights, batch_attr, edges, output_dir, api_dict_path, data_name, api_ids,
                    edge_masks=None):
    if attention_weights.shape[0] > edges.shape[1]:
        attention_weights = attention_weights[:edges.shape[1]]

    print(f"\nNumber of unique edges: {edges.unique(dim=1).size(1)}")
    print(f"Total number of edges: {edges.size(1)}")
    print(f"edges.shape: {edges.shape}")
    print(f"attention_weights.shape: {attention_weights.shape}")
    print(f"batch_attr.shape: {batch_attr.shape}")
    print(f"len(data_name): {len(data_name)}")
    print(f"len(api_ids): {len(api_ids)}")
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    attention_weights = attention_weights.squeeze()  # Flatten (num_edges, 1) -> (num_edges)
    num_graphs = batch_attr.max().item() + 1  # Total number of graphs in the batch
    print(f"num_graphs: {num_graphs}")

    node_to_api_dict = load_node_api_mapping(api_dict_path)

    for i in range(num_graphs):  # Iterate over graphs in the batch
        try:
            # Mask for edges in the current graph
            mask = batch_attr[edges[0]] == i  # Matches batch_attr of source nodes
            source_nodes = edges[0][mask].cpu().numpy()
            target_nodes = edges[1][mask].cpu().numpy()
            attention_values = attention_weights[mask].cpu().detach().numpy()

            # Map global node indices back to local node indices for graph `i`
            offset = batch_attr.eq(i).nonzero(as_tuple=True)[0][0].item()  # Start index for graph `i`
            source_nodes = (source_nodes - offset).reshape(-1)  # Ensure 1D
            target_nodes = (target_nodes - offset).reshape(-1)  # Ensure 1D
            attention_values = attention_values.reshape(-1, attention_weights.shape[1])

            # Create DataFrame for this graph
            df = pd.DataFrame({
                'source_node': source_nodes,
                'target_node': target_nodes
            })

            print(f"i: {i} - data_name[{i}]: {data_name[i]}")
            file_name = data_name[i][:-3]

            df['source_API_no'] = df['source_node'].apply(lambda x: api_ids[offset + x].item())
            df['target_API_no'] = df['target_node'].apply(lambda x: api_ids[offset + x].item())

            # Map source and target nodes to their corresponding API names
            df['source_api_name'] = df['source_API_no'].map(node_to_api_dict)
            df['target_api_name'] = df['target_API_no'].map(node_to_api_dict)

            # Reorder columns
            df = df[['source_node', 'source_API_no', 'source_api_name', 'target_node', 'target_API_no',
                     'target_api_name']]

            # Add attention values for each head as separate columns
            for head_idx in range(attention_values.shape[1]):
                df[f'head-{head_idx + 1} attention'] = attention_values[:, head_idx]

            # Add "average attention" column
            df['average_attention'] = attention_values.mean(axis=1)

            # Add attention explanations from each explainer
            if edge_masks is not None:
                for explainer_name, edge_mask in edge_masks.items():
                    edge_mask = edge_mask.squeeze()
                    explainer_values = edge_mask[mask].cpu().detach().numpy()
                    df[f'{explainer_name}_explanation'] = explainer_values

            # Save DataFrame to an individual Excel file
            file_path = os.path.join(output_dir, f'{file_name}.xlsx')
            df.to_excel(file_path, index=False)

            # Load the Excel workbook and sheet to adjust column widths
            wb = load_workbook(file_path)
            ws = wb.active

            # Set the column widths (adjust these values as needed)
            column_widths = {
                'A': 13,
                'B': 13,
                'C': 30,
                'D': 13,
                'E': 13,
                'F': 30,
                'G': 15,
                'H': 15,
                'I': 15,
                'J': 15,
                'K': 15,
                'L': 15,
                'M': 15,
                'N': 15,
                'O': 16,
                'P': 20,
                'R': 20,
                'S': 20,
            }

            # Apply the column widths
            for col, width in column_widths.items():
                ws.column_dimensions[col].width = width

            # Save the changes
            wb.save(file_path)
            print(f"Saved: {file_name}")
        except Exception as e:
            print(f"Error: {e}")


def load_node_api_mapping(file_path):
    node_api_dict = {}

    # Open and read the file
    with open(file_path, 'r') as file:
        for line in file:
            # Use regular expression to match key-value pairs
            match = re.match(r"'(.+)': (\d+)", line.strip())
            if match:
                key = match.group(1)  # Extract the API name
                value = int(match.group(2))  # Extract the node index as an integer
                node_api_dict[value] = key  # Map node index to API name

    return node_api_dict
