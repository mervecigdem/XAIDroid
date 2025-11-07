import os

from helpers.functions import load_config, ensure_directories_exist

from helpers.mcl_utilities import localize_malicious_codes


def ninth_stage():
    # Load configuration
    config = load_config('config.json')

    # Input directory
    mcl_calculation_dir = config["mcl_calculation_dir"]

    # Ensure output directory exists
    mcl_metrics_dir = config["mcl_metrics_dir"]
    directories = [mcl_metrics_dir]
    ensure_directories_exist(directories, [])

    # Output files for class localization
    class_mcl_metrics_vs_threshold_plot_filepath = os.path.join(mcl_metrics_dir,
                                                                config["class_mcl_metrics_vs_threshold_plot_filename"])
    class_mcl_metrics_vs_threshold_txt_filepath = os.path.join(mcl_metrics_dir,
                                                               config["class_mcl_metrics_vs_threshold_txt_filename"])
    class_gam_gat_localization_filepath = os.path.join(mcl_metrics_dir, config["class_gam_gat_localization_filename"])
    class_gam_gat_localization_metrics_filepath = os.path.join(mcl_metrics_dir,
                                                               config["class_gam_gat_localization_metrics_filename"])
    class_gam_gat_localization_roc_curve_filepath = os.path.join(mcl_metrics_dir,
                                                                 config[
                                                                     "class_gam_gat_localization_roc_curve_filename"])
    class_gam_gat_localization_pr_curve_filepath = os.path.join(mcl_metrics_dir,
                                                                config["class_gam_gat_localization_pr_curve_filename"])

    # Output files for method localization
    method_mcl_metrics_vs_threshold_plot_filepath = os.path.join(mcl_metrics_dir,
                                                                 config[
                                                                     "method_mcl_metrics_vs_threshold_plot_filename"])
    method_mcl_metrics_vs_threshold_txt_filepath = os.path.join(mcl_metrics_dir,
                                                                config[
                                                                    "method_mcl_metrics_vs_threshold_txt_filename"])
    method_gam_gat_localization_filepath = os.path.join(mcl_metrics_dir, config["method_gam_gat_localization_filename"])
    method_gam_gat_localization_metrics_filepath = os.path.join(mcl_metrics_dir,
                                                                config["method_gam_gat_localization_metrics_filename"])
    method_gam_gat_localization_roc_curve_filepath = os.path.join(mcl_metrics_dir,
                                                                  config[
                                                                      "method_gam_gat_localization_roc_curve_filename"])
    method_gam_gat_localization_pr_curve_filepath = os.path.join(mcl_metrics_dir,
                                                                 config[
                                                                     "method_gam_gat_localization_pr_curve_filename"])

    localize_malicious_codes(mcl_calculation_dir,
                             "class",
                             class_mcl_metrics_vs_threshold_plot_filepath,
                             class_mcl_metrics_vs_threshold_txt_filepath,
                             class_gam_gat_localization_filepath,
                             class_gam_gat_localization_metrics_filepath,
                             class_gam_gat_localization_roc_curve_filepath,
                             class_gam_gat_localization_pr_curve_filepath,
                             0.97)

    localize_malicious_codes(mcl_calculation_dir,
                             "method",
                             method_mcl_metrics_vs_threshold_plot_filepath,
                             method_mcl_metrics_vs_threshold_txt_filepath,
                             method_gam_gat_localization_filepath,
                             method_gam_gat_localization_metrics_filepath,
                             method_gam_gat_localization_roc_curve_filepath,
                             method_gam_gat_localization_pr_curve_filepath,
                             0.96)


ninth_stage()
