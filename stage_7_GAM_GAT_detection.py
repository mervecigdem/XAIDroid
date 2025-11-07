import os

from helpers.functions import load_config, ensure_directories_exist, load_and_merge_gam_and_gat_detection, \
    gam_and_gat_detection_fusion, rename_detection_excel_columns, save_detection_metrics, plot_roc_and_pr_curves


def seventh_stage():
    """
    Load, process, merge, and save GAM and GAT data based on configuration.
    """
    config = load_config('config.json')
    if not config:
        return None

    # Ensure output directory exists
    gam_gat_detection_dir = config["gam_gat_detection_dir"]
    directories = [
        gam_gat_detection_dir,
    ]
    ensure_directories_exist(directories, [])

    gam_detection_file_path = os.path.join(config.get("gam_logs_dir"), config["gam_predictions_filename"])
    gat_detection_file_path = os.path.join(config.get("gat_logs_dir"), config["gat_classification_results_filename"])

    fusion_file = os.path.join(gam_gat_detection_dir, config["gam_gat_detection_filename"])
    metrics_file = os.path.join(gam_gat_detection_dir, config["gam_gat_detection_metrics_filename"])
    roc_curve_file = os.path.join(gam_gat_detection_dir, config["gam_gat_detection_roc_curve_filename"])
    pr_curve_file = os.path.join(gam_gat_detection_dir, config["gam_gat_detection_pr_curve_filename"])

    df = load_and_merge_gam_and_gat_detection(gam_detection_file_path, gat_detection_file_path)

    df = gam_and_gat_detection_fusion(df)

    # Save fusion file
    df.to_excel(fusion_file, index=False)
    print(f"Fusion predictions saved.")

    df = rename_detection_excel_columns(df)

    predictors = {
        "GAT": ("gat_pred", "gat_prob"),
        "GAM": ("gam_pred", "gam_prob"),
        "Fusion_AND": ("fusion_and", None),
        "Fusion_OR": ("fusion_or", None),
        "Fusion_Weighted": ("fusion_weighted", "fusion_weighted_score")
    }

    save_detection_metrics(df, predictors, metrics_file)

    plot_roc_and_pr_curves(df, predictors, roc_curve_file, pr_curve_file)


seventh_stage()
