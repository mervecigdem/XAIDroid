import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import torch
import random
import numpy as np
import openpyxl
from collections import defaultdict
from torch_geometric.data import DataLoader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_dataloaders(train_data_path, test_data_path, batch_size, shuffle_train, shuffle_test):
    train_data_list = load_dataset(train_data_path)
    test_data_list = load_dataset(test_data_path)

    train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=shuffle_train)
    test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=shuffle_test)

    count_labels(train_loader, "Train set")
    count_labels(test_loader, "Test set")

    return train_loader, test_loader


def load_dataset(folder):
    data_list = []
    for file_name in os.listdir(folder):
        if file_name.endswith('.pt'):
            file_path = os.path.join(folder, file_name)
            data = torch.load(file_path)
            data.name = file_name
            data_list.append(data)
    return data_list


def count_labels(loader, graph_type):
    label_count = defaultdict(int)  # Dictionary to count the number of graphs per label
    for data in loader:
        for label in data.y.tolist():
            label_count[label] += 1  # Increment the count for each label

    for label, count in label_count.items():
        print(f"{graph_type} - Label {label}: {count} graphs")


def initialize_logging(log_file):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss',
                         'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1', 'Train FPR', 'Train FNR',
                         'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1', 'Test FPR', 'Test FNR'])


def write_log(log_file, epoch, epochs, loss, train_acc, train_precision, train_recall, train_f1, train_fpr, train_fnr,
              test_acc, test_precision, test_recall, test_f1, test_fpr, test_fnr):
    # Log the epoch results into the CSV file
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, loss,
                         train_acc, train_precision, train_recall, train_f1, train_fpr, train_fnr,
                         test_acc, test_precision, test_recall, test_f1, test_fpr, test_fnr])

    print(f'\nEpoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')
    print(
        f'Train Accuracy: {train_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, '
        f'F1-score: {train_f1:.4f}, FPR: {train_fpr:.4f}, FNR: {train_fnr:.4f}')
    print(
        f'Test  Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, '
        f'F1-score: {test_f1:.4f}, FPR: {test_fpr:.4f}, FNR: {test_fnr:.4f}')


def save_graph_results_to_excel(graph_results, file_path):
    """Saves graph results to an Excel file."""
    wb = openpyxl.Workbook()
    ws = wb.active

    # Set column widths
    ws.column_dimensions['A'].width = 80
    ws.column_dimensions['B'].width = 15
    ws.column_dimensions['C'].width = 15
    ws.column_dimensions['D'].width = 15

    # Set headers
    ws.append(["Graph_Name", "Actual_Label", "Predicted_Label", "Probability"])

    # Write results to rows
    for result in graph_results:
        graph, actual_label, predicted_label, prob = result.split(",")
        ws.append([graph, actual_label, predicted_label, prob])

    # Save Excel file
    wb.save(file_path)


def plot_metrics(log_file, output_file):
    # Load the data from the CSV log file
    data = pd.read_csv(log_file)

    # Extract values for plotting
    epochs = data['Epoch']
    train_acc = data['Train Accuracy']
    train_precision = data['Train Precision']
    train_recall = data['Train Recall']
    train_f1 = data['Train F1']
    train_fpr = data['Train FPR']
    train_fnr = data['Train FNR']

    test_acc = data['Test Accuracy']
    test_precision = data['Test Precision']
    test_recall = data['Test Recall']
    test_f1 = data['Test F1']
    test_fpr = data['Test FPR']
    test_fnr = data['Test FNR']

    # Plot
    plt.figure(figsize=(10, 8))

    # Plot for Accuracy
    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_acc, label='Train Accuracy', marker='o')
    plt.plot(epochs, test_acc, label='Test Accuracy', marker='x')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Percentage')
    plt.ylim(0, 1)
    plt.legend()

    # Plot for Precision
    plt.subplot(2, 3, 2)
    plt.plot(epochs, train_precision, label='Train Precision', marker='o')
    plt.plot(epochs, test_precision, label='Test Precision', marker='x')
    plt.title('Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Percentage')
    plt.ylim(0, 1)
    plt.legend()

    # Plot for Recall
    plt.subplot(2, 3, 3)
    plt.plot(epochs, train_recall, label='Train Recall', marker='o')
    plt.plot(epochs, test_recall, label='Test Recall', marker='x')
    plt.title('Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Percentage')
    plt.ylim(0, 1)
    plt.legend()

    # Plot for F1-score
    plt.subplot(2, 3, 4)
    plt.plot(epochs, train_f1, label='Train F1', marker='o')
    plt.plot(epochs, test_f1, label='Test F1', marker='x')
    plt.title('F1-score')
    plt.xlabel('Epoch')
    plt.ylabel('Percentage')
    plt.ylim(0, 1)
    plt.legend()

    # Plot for FPR
    plt.subplot(2, 3, 5)
    plt.plot(epochs, train_fpr, label='Train FPR', marker='o')
    plt.plot(epochs, test_fpr, label='Test FPR', marker='x')
    plt.title('FPR')
    plt.xlabel('Epoch')
    plt.ylabel('Percentage')
    plt.ylim(0, 1)
    plt.legend()

    # Plot for FNR
    plt.subplot(2, 3, 6)
    plt.plot(epochs, train_fnr, label='Train FNR', marker='o')
    plt.plot(epochs, test_fnr, label='Test FNR', marker='x')
    plt.title('FNR')
    plt.xlabel('Epoch')
    plt.ylabel('Percentage')
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()

    # Save the figure
    plt.savefig(output_file)
    plt.close()  # Close the plot to free memory


def find_apk_names(folder):
    apk_names = set(os.path.splitext(f)[0] for f in os.listdir(folder)
                    if os.path.splitext(f)[1] in ['.xlsx'])
    return apk_names


def calculate_gat_node_attentions(apk_names, critical_api_set, gat_edge_attentions_folder, gat_node_attentions_folder):
    for apk_file in apk_names:
        gat_edge_attentions_df = read_gat_attention_excel_file(apk_file, gat_edge_attentions_folder)
        critical_apis_attention = []
        for critical_api in critical_api_set:
            api_edge_attentions = gat_edge_attentions_df[
                (gat_edge_attentions_df["source_api_name"] == critical_api) |
                (gat_edge_attentions_df["target_api_name"] == critical_api)
                ]
            if len(api_edge_attentions) > 0:
                average_attention = api_edge_attentions["average_attention"].mean()

                if (api_edge_attentions["source_api_name"] == critical_api).any():
                    api_id = api_edge_attentions.loc[
                        api_edge_attentions["source_api_name"] == critical_api, "source_API_no"
                    ].iloc[0]
                else:
                    api_id = api_edge_attentions.loc[
                        api_edge_attentions["target_api_name"] == critical_api, "target_API_no"
                    ].iloc[0]

                number_of_edges = len(api_edge_attentions)

                critical_apis_attention.append({
                    "API ID": api_id,
                    "API Name": f"'{critical_api}'",
                    "Number of Edges": number_of_edges,
                    "Avg. Attention of Agents Classifying Mw": average_attention,
                })
        critical_apis_attention_df = pd.DataFrame(critical_apis_attention)

        if critical_apis_attention_df.empty:
            print(f"The DataFrame of {apk_file} is empty!")
        else:
            total_sum = critical_apis_attention_df["Avg. Attention of Agents Classifying Mw"].sum()
            critical_apis_attention_df["Avg. Attention of Agents Classifying Mw"] = \
                critical_apis_attention_df["Avg. Attention of Agents Classifying Mw"] / total_sum

            critical_apis_attention_df = critical_apis_attention_df.sort_values(
                by="Avg. Attention of Agents Classifying Mw",
                ascending=False)
            write_api_node_attentions_to_excel(critical_apis_attention_df, gat_node_attentions_folder, apk_file)


def read_gat_attention_excel_file(file_name, folder_path):
    file_path = os.path.join(folder_path, file_name + '.xlsx')

    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        columns_to_keep = ['source_API_no', 'source_api_name', 'target_API_no', 'target_api_name', 'average_attention']
        df = df.loc[:, columns_to_keep]
        return df
    else:
        raise FileNotFoundError(f"The file {file_name} is not in the folder {folder_path}")


def get_critical_api_list(file_path):
    critical_api_set = set()
    with open(file_path, 'r') as file:
        for line in file:
            key = line.split(':')[0].strip("'")
            critical_api_set.add(key)
    return critical_api_set


def write_api_node_attentions_to_excel(dataframe, folder, file):
    output_file = f"{folder}/{file}.xlsx"

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        dataframe.to_excel(writer, index=False)
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        for col in worksheet.columns:
            max_length = 0
            column = col[0].column_letter  # Get the column name
            for cell in col:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass
            adjusted_width = (max_length + 2)  # Add extra space for better readability
            worksheet.column_dimensions[column].width = adjusted_width

    print(f"Results of {file} saved.")
