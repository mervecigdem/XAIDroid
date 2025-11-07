import os

from helpers.functions import load_config, ensure_directories_exist

from helpers.visualization_helpers import create_base_graph_without_attention, \
    visualize_node_attentions, visualize_edge_attention


def tenth_stage():
    # Load configuration
    config = load_config('config.json')

    an_example_apk = "sms_ba48a05e03073cdbd8230ba9194eb1866d9f790ab0f10d949295e5479825b927"
    apk_subfoler = "agent_aax"

    # Inputs
    gam_gat_node_attentions_file = os.path.join(config["mcl_calculation_dir"], apk_subfoler, an_example_apk + ".xlsx")
    gat_edge_attentions_file = os.path.join(config["gat_edge_attentions_dir"], an_example_apk + ".xlsx")
    selected_apis_dictionary_file_path = os.path.join(config["selected_apis_list_dir"],
                                                      config["selected_apis_dictionary_file"])
    json_api_call_graph_file = os.path.join(config["api_call_graphs_json_dir"], an_example_apk + ".json")

    # Ensure output directory exists
    visualization_dir = config["visualization_dir"]
    directories = [visualization_dir]
    ensure_directories_exist(directories, [])

    # Output files for class localization
    gam_node_attention_visual = os.path.join(visualization_dir, config["gam_node_attention_visual"])
    gat_edge_attention_visual = os.path.join(visualization_dir, config["gat_edge_attention_visual"])
    gat_node_attention_visual = os.path.join(visualization_dir, config["gat_node_attention_visual"])

    # Build the graph and color attentions
    base_graph = create_base_graph_without_attention(gat_edge_attentions_file)
    visualize_node_attentions(base_graph,
                              gam_gat_node_attentions_file,
                              selected_apis_dictionary_file_path,
                              json_api_call_graph_file,
                              gam_node_attention_visual,
                              gat_node_attention_visual)
    visualize_edge_attention(base_graph, gat_edge_attention_visual)


tenth_stage()
