import warnings

from helpers.functions import load_config, ensure_directories_exist, find_remaining_apks_for_first_stage, \
    create_files_and_graphs


def first_stage():
    # Disable user warnings
    warnings.filterwarnings("ignore", category=UserWarning, module=".*")

    # Load configuration
    config = load_config('config.json')

    # Access file paths from the configuration
    apks_input_dir = config["apks_input_dir"]
    apk_info_dir = config["apk_info_dir"]
    df_normalized_csv_dir = config["df_normalized_csv_dir"]
    custom_methods_and_called_apis_dir = config["custom_methods_and_called_apis_dir"]
    custom_methods_dir = config["custom_methods_dir"]
    android_apis_dir = config["android_apis_dir"]

    # Define directories
    directories = [
        apk_info_dir,
        df_normalized_csv_dir,
        custom_methods_and_called_apis_dir,
        custom_methods_dir,
        android_apis_dir
    ]
    subdirectories = [
        config["benignware"],
        config["malware"],
    ]

    # Ensure directories exists
    ensure_directories_exist(directories, subdirectories)

    # Find remaining apk files and create files and graphs
    for directory in subdirectories:
        remaining_files = find_remaining_apks_for_first_stage(apks_input_dir, android_apis_dir,
                                                              directory)
        create_files_and_graphs(
            directory, apks_input_dir, apk_info_dir, custom_methods_dir,
            android_apis_dir, custom_methods_and_called_apis_dir,
            df_normalized_csv_dir, remaining_files
        )


first_stage()
