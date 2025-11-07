import os
import pandas as pd

from helpers.functions import load_config, ensure_directories_exist

from helpers.mcl_utilities import kaspersky_to_folder_name, find_common_filenames, process_apk_methods_and_classes, \
    save_metrics_to_excel


def eighth_stage():
    # Load configuration
    config = load_config('config.json')

    # Ensure directories exists
    mcl_calculation_dir = config["mcl_calculation_dir"]
    directories = [mcl_calculation_dir]
    ensure_directories_exist(directories, [])

    apk_info_excel_path = os.path.join(config["mcl_prep_folder"], config["mcl_prep_filename"])

    # Access configuration values
    number_of_top_apis = config['mcl_number_of_top_apis']
    method_mcl_baseline_path = config['method_mcl_baseline_path']
    class_mcl_baseline_path = config['class_mcl_baseline_path']
    gam_attention_path = config['gam_node_attentions_dir']
    gat_attention_path = config['gat_node_attentions_dir']
    custom_methods_path = os.path.join(config['custom_methods_dir'], config['mw'])
    custom_methods_and_called_apis_path = os.path.join(config['custom_methods_and_called_apis_dir'], config['mw'])
    excluded_classes = tuple(config['excluded_classes'])
    kaspersky_values = set(config["mw_types"].values())

    # Filter APK DataFrame
    apk_info_df = pd.read_excel(apk_info_excel_path)
    print(f"Total APKs: {len(apk_info_df)}")

    common_files = find_common_filenames(class_mcl_baseline_path, gam_attention_path, gat_attention_path,
                                         custom_methods_path, custom_methods_and_called_apis_path)

    # Iterate through APK file information and process accordingly
    for apk_filename in common_files:
        try:
            print(f"apk_filename: {apk_filename}")
            row = apk_info_df[apk_info_df["APK Name"].str.startswith(apk_filename)].iloc[0]
            kaspersky_value = row["Kaspersky"]
            mw_type = kaspersky_to_folder_name(apk_filename, kaspersky_value)
            print(f"Processing {apk_filename} - mw_type: {mw_type}")
            try:
                gam_gat_df, method_df, class_df = process_apk_methods_and_classes(
                        apk_filename,
                        gam_attention_path, gat_attention_path,
                        custom_methods_and_called_apis_path,
                        custom_methods_path, class_mcl_baseline_path, method_mcl_baseline_path,
                        excluded_classes, number_of_top_apis
                )

                # Save the merged DataFrame to an Excel file
                output_filename_sum = apk_filename + '.xlsx'
                output_path = f'{mcl_calculation_dir}{mw_type}/'
                os.makedirs(output_path, exist_ok=True)
                output_file_path_sum = os.path.join(output_path, output_filename_sum)

                save_metrics_to_excel(output_file_path_sum, gam_gat_df, method_df, class_df)

            except Exception as e:
                print(f"Error processing {apk_filename}: {e}")
                continue
        except Exception as e:
            print(f"Error: {e}")


eighth_stage()
