import torch
import json
import os
import glob

from helpers.functions import load_config, ensure_directories_exist, convert_json_to_pyg


def fourth_stage():
    # Load configuration
    config = load_config('config.json')

    api_call_graphs_json_dir = config["api_call_graphs_json_dir"]
    api_call_graphs_pt_dir = config["api_call_graphs_pt_dir"]
    bw_apk_folder = config["benignware"]
    mw_apk_folder = config["malware"]

    # Ensure output directories exists
    directories = [api_call_graphs_pt_dir]
    subdirectories = [
        bw_apk_folder,
        mw_apk_folder,
    ]
    ensure_directories_exist(directories, subdirectories)

    # Traverse through all JSON files in the input directory and its subdirectories
    for json_file in glob.glob(os.path.join(api_call_graphs_json_dir, '**/*.json'), recursive=True):
        # Determine the output path by replicating the folder structure
        relative_path = os.path.relpath(json_file, api_call_graphs_json_dir)
        output_path = os.path.join(api_call_graphs_pt_dir, os.path.dirname(relative_path))
        os.makedirs(output_path, exist_ok=True)

        # Load the JSON file
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        # Convert JSON to PyTorch Geometric data
        pyg_data = convert_json_to_pyg(json_data)

        # Save the PyTorch Geometric data object
        output_file = os.path.join(output_path, os.path.basename(json_file).replace('.json', '.pt'))
        torch.save(pyg_data, output_file)


fourth_stage()
