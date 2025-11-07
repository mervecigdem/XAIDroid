import os
import warnings

from helpers.functions import load_config, sum_up_file_content, merge_bw_and_mw_android_apis, choose_api_list, \
    write_into_excel


def second_stage():
    # Load configuration
    config = load_config('config.json')

    benignware_folder = config["benignware"]
    malware_folder = config["malware"]
    selected_apis_list_dir = config["selected_apis_list_dir"]
    android_apis_dir = config["android_apis_dir"]
    critical_api_full_list = config["critical_api_full_list"]
    merged_output_file = os.path.join(selected_apis_list_dir, "all_apis.xlsx")
    output_list_file = os.path.join(selected_apis_list_dir, config["selected_apis_list_file"])
    output_dictionary_file = os.path.join(selected_apis_list_dir, config["selected_apis_dictionary_file"])

    if not os.path.exists(selected_apis_list_dir):
        os.makedirs(selected_apis_list_dir)

    # Disable user warnings
    warnings.filterwarnings("ignore", category=UserWarning, module=".*")

    with open(critical_api_full_list, 'r', encoding="utf-8") as f:
        critical_apis = set(f.read().splitlines())

    # Bw Android APIs
    bw_input_dir = os.path.join(android_apis_dir, benignware_folder)
    bw_df = sum_up_file_content(bw_input_dir, critical_apis)

    # Mw Android APIs
    mw_input_dir = os.path.join(android_apis_dir, malware_folder)
    mw_df = sum_up_file_content(mw_input_dir, critical_apis)

    # Merge Bw and Mw Android APIs
    merged_df = merge_bw_and_mw_android_apis(bw_df, mw_df)
    write_into_excel(merged_df, merged_output_file)

    # Select APIs
    choose_api_list(merged_df, output_list_file, output_dictionary_file)


second_stage()
