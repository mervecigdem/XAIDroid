import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import json


def load_attention_data(file_path):
    """Load the attention data from an Excel file."""
    return pd.read_excel(file_path)


def create_base_graph_without_attention(file_path):
    """Build and return a directed graph based on the attention data."""
    df = load_attention_data(file_path)

    G = nx.DiGraph()  # Directed graph for API flows

    # Add nodes and edges with attention weights
    for _, row in df.iterrows():
        G.add_node(row['source_node'], label=row['source_node'])  # label=row['source_api_name']
        G.add_node(row['target_node'], label=row['target_node'])  # label=row['target_api_name']
        G.add_edge(row['source_node'], row['target_node'], weight=row['attention_explanation'])

    return G


def get_edge_weights(G):
    """Return the edge weights and colors for drawing."""
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    edge_colors = edge_weights  # Use attention directly as color
    return edge_weights, edge_colors


def scale_edge_widths(edge_weights, max_width=4):
    """Scale the edge widths based on attention weights."""
    return [max_width * (w / max(edge_weights)) for w in edge_weights]


def plot_edge_attention(graph, edge_weights, edge_colors, widths, save_path):
    """Draw the attention graph with nodes, edges, and attention-based attributes."""
    pos = nx.circular_layout(graph)

    # Set the figure size to make the graph larger
    plt.figure(figsize=(12, 9))

    # Draw nodes and labels
    nx.draw_networkx_nodes(graph, pos, node_size=300, node_color="#88c0d0")
    nx.draw_networkx_labels(graph, pos, labels=nx.get_node_attributes(graph, 'label'), font_size=9)

    # Draw edges with attention-based width and color
    cmap = plt.cm.Reds
    nx.draw_networkx_edges(
        graph, pos, edge_color=edge_colors, edge_cmap=cmap, width=widths, arrows=True
    )

    # Add colorbar for attention scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
    sm.set_array([])
    ax = plt.gca()
    plt.colorbar(sm, ax=ax, label='GAT Edge Attention')

    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved GAT edge attention visualization to {save_path}")
    else:
        plt.show()


def plot_node_attention(graph, labels, attention_map, values, save_path, title):
    plt.figure(figsize=(12, 9))

    # Assign colors; use NaN for missing values
    node_colors = [
        attention_map.get(labels[node], float("nan")) if node in labels else float("nan")
        for node in graph.nodes()
    ]

    pos = nx.circular_layout(graph)

    # Draw nodes
    cmap = plt.cm.Reds
    nx.draw_networkx_nodes(
        graph, pos, node_size=300, node_color=node_colors,
        cmap=cmap, vmin=min(values), vmax=max(values)
    )

    # Draw only labels for existing nodes
    nx.draw_networkx_labels(
        graph,
        pos,
        labels={n: str(n) for n in graph.nodes()},  # note: labels[n] shows the API names
        font_size=9
    )
    nx.draw_networkx_edges(graph, pos, arrows=True, alpha=0.3)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(values), vmax=max(values)))
    sm.set_array([])
    ax = plt.gca()
    plt.colorbar(sm, ax=ax, label=title)

    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {title} node attention visualization to {save_path}")


def clean_api_name(name: str) -> str:
    """Remove surrounding quotes and whitespace from API names."""
    return str(name).strip().strip("'").strip('"')


def visualize_node_attentions(
    graph,
    file_path,
    selected_apis_dictionary_file_path,
    json_api_call_graph_file,
    gam_node_attention_visual,
    gat_node_attention_visual
):
    """
    Visualize node-level attentions from Excel file (api_attentions sheet).
    Colors nodes according to GAM and GAT attentions, and saves two separate images.
    """

    # ----------------------------------------------------------------------
    # Step 1. Load Excel with node attentions (and clean names)
    # ----------------------------------------------------------------------
    df = pd.read_excel(file_path, sheet_name="api_attentions")

    df["API Name"] = df["API Name"].apply(clean_api_name)

    gam_attention = dict(zip(df["API Name"], df["GAM attention"]))
    gat_attention = dict(zip(df["API Name"], df["GAT attention"]))

    # ----------------------------------------------------------------------
    # Step 2. Load API dictionary: api_number -> api_name (cleaned)
    # ----------------------------------------------------------------------
    api_dict = {}
    with open(selected_apis_dictionary_file_path, "r") as f:
        for line in f:
            if line.strip():
                name, idx = line.split(":")
                api_dict[int(idx.strip())] = clean_api_name(name)

    # ----------------------------------------------------------------------
    # Step 3. Load call graph JSON and map node_id -> api_number -> api_name
    # ----------------------------------------------------------------------
    with open(json_api_call_graph_file, "r") as f:
        call_graph = json.load(f)

    node_to_api_num = call_graph["labels"]  # { "3": 98, "4": 99, ... }

    # Build node_id -> api_name mapping
    labels = {}
    for node_id, api_num in node_to_api_num.items():
        api_num = int(api_num)
        node_id = int(node_id)
        if api_num in api_dict:
            labels[node_id] = api_dict[api_num]

    # ----------------------------------------------------------------------
    # Step 4. Prepare normalization values
    # ----------------------------------------------------------------------
    gam_values = list(gam_attention.values())
    gat_values = list(gat_attention.values())

    # ----------------------------------------------------------------------
    # Step 5. Visualization
    # ----------------------------------------------------------------------
    plot_node_attention(
        graph,
        labels,
        gam_attention,
        gam_values,
        gam_node_attention_visual,
        "GAM Node Attention"
    )

    plot_node_attention(
        graph,
        labels,
        gat_attention,
        gat_values,
        gat_node_attention_visual,
        "GAT Node Attention"
    )


def visualize_edge_attention(graph, save_path):
    # Get edge weights and corresponding colors
    edge_weights, edge_colors = get_edge_weights(graph)

    # Scale the edge widths for visibility
    widths = scale_edge_widths(edge_weights)

    # Draw the graph
    plot_edge_attention(graph, edge_weights, edge_colors, widths, save_path)
