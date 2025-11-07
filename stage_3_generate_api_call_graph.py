import warnings
import os

from helpers.functions import load_config, ensure_directories_exist, obtain_api_dict, \
    find_remaining_apks_for_third_stage, all_steps_for_api_cg_creation


def third_stage():
    # Disable the user warnings
    warnings.filterwarnings("ignore", category=UserWarning, module=".*")

    # Load configuration
    config = load_config('config.json')

    df_normalized_csv_dir = config["df_normalized_csv_dir"]
    custom_methods_dir = config["custom_methods_dir"]
    android_apis_dir = config["android_apis_dir"]
    selected_apis_list_dir = config["selected_apis_list_dir"]
    selected_apis_used_in_apks = os.path.join(selected_apis_list_dir, config["selected_apis_list_file"])
    selected_apis_dictionary_file_path = os.path.join(selected_apis_list_dir, config["selected_apis_dictionary_file"])
    api_call_graphs_json_dir = config["api_call_graphs_json_dir"]
    api_call_graph_type = config["api_call_graph_type"]
    bw_apk_folder = config["benignware"]
    mw_apk_folder = config["malware"]
    apk_type_bw, apk_type_mw = 0, 1
    api_dict = obtain_api_dict(selected_apis_used_in_apks, selected_apis_dictionary_file_path)

    # Ensure directories exists
    directories = [api_call_graphs_json_dir]
    subdirectories = [
        bw_apk_folder,
        mw_apk_folder,
    ]
    ensure_directories_exist(directories, subdirectories)

    # Process Benignware APK folders
    remaining_files = find_remaining_apks_for_third_stage(android_apis_dir, api_call_graphs_json_dir, bw_apk_folder)
    all_steps_for_api_cg_creation(bw_apk_folder, remaining_files, android_apis_dir, custom_methods_dir,
                                  df_normalized_csv_dir, api_call_graphs_json_dir, api_call_graph_type,
                                  api_dict, apk_type_bw)

    # Process Malware APK folders
    remaining_files = find_remaining_apks_for_third_stage(android_apis_dir, api_call_graphs_json_dir, mw_apk_folder)
    all_steps_for_api_cg_creation(mw_apk_folder, remaining_files, android_apis_dir, custom_methods_dir,
                                  df_normalized_csv_dir, api_call_graphs_json_dir, api_call_graph_type,
                                  api_dict, apk_type_mw)


third_stage()
