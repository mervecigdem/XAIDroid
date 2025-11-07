import os.path
import torch

from models.GAT_model import initialize_model, initialize_explainers, train, test
from helpers.functions import load_config, ensure_directories_exist
from helpers.gat_utilities import set_seed, prepare_dataloaders, initialize_logging, write_log, plot_metrics, \
    find_apk_names, get_critical_api_list, calculate_gat_node_attentions


def sixth_stage():
    # Load configuration
    config = load_config('config.json')

    # Set the seed for reproducibility
    set_seed(config["gat_seed"])

    # Ensure directories exists
    gat_logs_dir = config["gat_logs_dir"]
    gat_edge_attentions_folder = config["gat_edge_attentions_dir"]
    gat_node_attentions_folder = config["gat_node_attentions_dir"]
    directories = [
        gat_logs_dir,
        gat_edge_attentions_folder,
        gat_node_attentions_folder
    ]
    ensure_directories_exist(directories, [])

    # Prepare dataloaders
    train_data_path = config["gat_train_data_path"]
    test_data_path = config["gat_test_data_path"]
    batch_size = config["gat_batch_size"]
    shuffle_train = config["gat_shuffle_train"]
    shuffle_test = config["gat_shuffle_test"]
    train_loader, test_loader = prepare_dataloaders(train_data_path, test_data_path, batch_size, shuffle_train,
                                                    shuffle_test)

    # Initialize model, loss, and optimizer
    in_channels = config["gat_in_channels"]
    hidden_channels = config["gat_hidden_channels"]
    out_channels = config["gat_out_channels"]
    heads = config["gat_heads"]
    dropout = config["gat_dropout"]
    lr = config["gat_lr"]
    weight_decay = config["gat_weight_decay"]
    model = initialize_model(in_channels, hidden_channels, out_channels, heads, dropout)
    explainers = initialize_explainers(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # Initialize logging
    log_file = os.path.join(config["gat_logs_dir"], config["gat_log_filename"])
    initialize_logging(log_file)

    total_epochs = config["gat_epochs"]
    classification_file = os.path.join(config["gat_logs_dir"], config["gat_classification_results_filename"])
    api_dict = os.path.join(config["selected_apis_list_dir"], config["selected_apis_dictionary_file"])
    test_json_path = config["gam_test_data_path"]
    save_attention = config["save_attention"]

    # Training Loop
    for epoch in range(total_epochs):
        # Training step
        loss = train(model, train_loader, optimizer, criterion)

        # Evaluate only every 5 epochs
        if epoch % 5 == 0 or epoch == total_epochs - 1:
            train_metrics = test(model, explainers, train_loader, epoch, total_epochs, "training", classification_file,
                                 gat_edge_attentions_folder, api_dict, save_attention)
            test_metrics = test(model, explainers, test_loader, epoch, total_epochs, "testing", classification_file,
                                gat_edge_attentions_folder, api_dict, save_attention)

            # Log the metrics
            write_log(log_file, epoch, total_epochs, loss, *train_metrics, *test_metrics)

        # Step the scheduler after each epoch
        scheduler.step(loss)

    # Plot metrics after training
    plot_file = os.path.join(config["gat_logs_dir"], config["gat_plot_filename"])
    plot_metrics(log_file, plot_file)

    # Obtain apk names
    apk_names = find_apk_names(gat_edge_attentions_folder)

    # Load API dictionary
    api_dict_path = os.path.join(config["selected_apis_list_dir"], config["selected_apis_dictionary_file"])
    critical_api_set = get_critical_api_list(api_dict_path)

    # Calculate GAT node attentions
    calculate_gat_node_attentions(apk_names, critical_api_set, gat_edge_attentions_folder, gat_node_attentions_folder)


sixth_stage()
