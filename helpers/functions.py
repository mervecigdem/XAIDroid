from androguard.misc import AnalyzeAPK
import os
import pandas as pd
import re
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import json
import networkx as nx
import ast
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve
)

from torch_geometric.data import Data


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def ensure_directories_exist(directories, subdirectories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
        for subdirectory in subdirectories:
            subdirectory_path = os.path.join(directory, subdirectory)
            if not os.path.exists(subdirectory_path):
                os.makedirs(subdirectory_path)


def find_remaining_apks_for_first_stage(input_dir_whole, output_dir_existing, apk_group):
    # Create the whole APK set
    directory_whole_files = os.path.join(input_dir_whole, apk_group)

    # Ensure the input directory exists
    if not os.path.isdir(directory_whole_files):
        raise FileNotFoundError(f"The directory {directory_whole_files} does not exist.")

    # Gather all APK file names, keeping track of whether they originally had the ".apk" extension
    whole_apk_set_with_extension = {
        file_name: file_name.endswith(".apk")
        for file_name in os.listdir(directory_whole_files)
    }

    # Normalize the APK set by stripping ".apk" when present
    whole_apk_set = set(
        file_name[:-4] if is_apk else file_name
        for file_name, is_apk in whole_apk_set_with_extension.items()
    )

    # Existing output files
    directory_existing_files = os.path.join(output_dir_existing, apk_group)

    # Ensure the output directory exists
    if not os.path.isdir(directory_existing_files):
        raise FileNotFoundError(f"The directory {directory_existing_files} does not exist.")

    # Gather all existing output files with ".txt" extension
    existing_output_files = set()
    for output_file in os.listdir(directory_existing_files):
        if output_file.endswith(".txt"):
            existing_output_files.add(os.path.splitext(output_file)[0])

    # Calculate the remaining APK files
    remaining_apk_files = whole_apk_set - existing_output_files

    # Restore the ".apk" extension for files that originally had it
    remaining_apk_files_with_extension = {
        filename + ".apk" if whole_apk_set_with_extension.get(filename + ".apk") else filename
        for filename in remaining_apk_files
    }

    print(f"Number of remaining APK files for {apk_group.rstrip('/')} is:\t{len(remaining_apk_files_with_extension)}")
    return remaining_apk_files_with_extension


def create_files_and_graphs(apk_group, input_dir, apk_info_output_dir, custom_method_set_output_dir,
                            android_apis_output_dir,
                            custom_methods_and_called_apis_output_dir,
                            df_normalized_csv_dir, apk_files):

    # Process each APK file
    for apk_file in apk_files:
        try:
            print("Analyzing", apk_file)
            # Get apk and output file paths
            apk_path = input_dir + apk_group + apk_file

            # Analyze the apk file using AnalyzeAPK
            a, d, dx = AnalyzeAPK(apk_path)  # a: APK object, d: DalvikVMFormat object, dx: Analysis object

            # Get output file paths using get_all_paths function
            f1, f2, f3, f4, f5 = \
                get_all_paths(apk_group, apk_info_output_dir, custom_method_set_output_dir, android_apis_output_dir, custom_methods_and_called_apis_output_dir,
                              df_normalized_csv_dir, apk_file)

            # Get call graph information of apk using apk_info_generation functions
            info_apk(a, f1)
            df_disassembled, general_android_api_set = get_disassembled_instructions(dx)
            df_normalized = get_normalized_instructions(df_disassembled, f2)
            custom_methods_and_call_apis = get_called_apis_from_custom_methods(df_normalized, f3)
            get_api_set(custom_methods_and_call_apis, general_android_api_set, f4, f5)
        except Exception as e:
           print("Exception: ", e)


def get_all_paths(apk_group, apk_info_output_dir, custom_method_set_output_dir, android_apis_output_dir,
                  custom_methods_and_called_apis_output_dir,
                  df_normalized_csv_dir, apk_file):

    if os.path.sep in apk_file or '/' in apk_file or '\\' in apk_file:
        directory_name, file_name = os.path.split(apk_file)
    else:
        file_name = apk_file
    name, ext = os.path.splitext(file_name)

    # Define the output file names
    f1 = os.path.join(apk_info_output_dir, apk_group, name + '.txt')
    f2 = os.path.join(df_normalized_csv_dir, apk_group, name + '.csv')
    f3 = os.path.join(custom_methods_and_called_apis_output_dir, apk_group, name + '.csv')
    f4 = os.path.join(custom_method_set_output_dir, apk_group, name + '.txt')
    f5 = os.path.join(android_apis_output_dir, apk_group, name + '.txt')

    # Return all the file paths as a tuple
    return f1, f2, f3, f4, f5


def info_apk(a, output_file):
    with open(output_file, 'w', encoding="utf-8") as f:
        print("APK permissions: ", a.get_permissions(), file=f)
        print("APK activities:", a.get_activities(), file=f)
        print("Package name:", a.get_package(), file=f)
        print("App name:", a.get_app_name(), file=f)
        print("App icon:", a.get_app_icon(), file=f)
        print("Android version code: ", a.get_androidversion_code(), file=f)
        print("Android version name: ", a.get_androidversion_name(), file=f)
        print("Min SDK version: ", a.get_min_sdk_version(), file=f)
        print("Max SDK version: ", a.get_max_sdk_version(), file=f)
        print("Target SDK version: ", a.get_target_sdk_version(), file=f)
        print("Effective target SDK version: ", a.get_effective_target_sdk_version(), file=f)
        print("Android manifest file: ", a.get_android_manifest_axml().get_xml(), file=f)


def get_disassembled_instructions(dx):
    data = []

    android_api_set = set()

    for method in dx.get_methods():
        m = method.get_method()

        if method.is_android_api():
            class_name = m.get_class_name()
            method_name = m.get_name()
            api_call = f"'{class_name}->{method_name}'"
            android_api_set.add(api_call)

        if method.is_external():
            continue

        # List of class name prefixes to be excluded or ignored -- known benign libraries can be added here
        excluded_prefixes = ["Landroid/support/"]

        if not any(m.get_class_name().startswith(prefix) for prefix in excluded_prefixes):
            for idx, ins in m.get_instructions_idx():
                op_value = ins.get_op_value()
                opcode = ins.get_name()
                op_output = ins.get_output()

                row_data = {
                    'method_name': m,
                    'idx': idx,
                    'op_value': op_value,
                    'opcode': opcode,
                    'opcode_output': op_output,
                }

                data.append(row_data)

    df = pd.DataFrame(data)
    return df, android_api_set


def get_normalized_instructions(df, file):
    # Op-values of necessary instructions to create a control flow graph
    return_op_value_ranges = [(14, 17)]
    invoke_op_value_ranges = [(110, 120)]
    goto_switch_if_else_op_value_ranges = [(40, 44), (50, 61)]
    op_value_ranges = return_op_value_ranges + invoke_op_value_ranges + goto_switch_if_else_op_value_ranges

    # Drop rows with opcodes outside the given range
    op_value_mask = [not any(start <= val <= end for start, end in op_value_ranges) for val in df['op_value']]
    df = df.drop(df[op_value_mask].index)

    # Normalize method names
    df['method_name'] = df['method_name'].apply(lambda m: f"'{m.get_class_name()}->{m.get_name()}'")

    # Normalize return instructions
    for start, end in return_op_value_ranges:
        condition = (df['op_value'] >= start) & (df['op_value'] <= end)
        df.loc[condition, 'opcode_output'] = ""

    # Normalize invoked method names
    clone_pattern = r'[BCDFIJSZ]->clone()'
    call_pattern = r'L[^\(]*'

    # Update for clone_pattern
    for start, end in invoke_op_value_ranges:
        condition = (df['op_value'] >= start) & (df['op_value'] <= end)
        df.loc[condition & df['opcode_output'].str.contains(clone_pattern, regex=True), 'opcode_output'] \
            = 'Ljava/lang/Object;->clone'

    # Update for call_pattern
    for start, end in invoke_op_value_ranges:
        condition = (df['op_value'] >= start) & (df['op_value'] <= end)
        df.loc[condition, 'opcode_output'] = df.loc[condition, 'opcode_output'].apply(
            lambda x: f"'{re.search(call_pattern, x).group(0)}'" if re.search(call_pattern, x) else x
        )

    # Normalize goto addresses
    hexadecimal_pattern = r'[-+][\da-f]+h'
    for start, end in goto_switch_if_else_op_value_ranges:
        condition = (df['op_value'] >= start) & (df['op_value'] <= end)
        for idx, row in df[condition].iterrows():
            hex_numbers = re.findall(hexadecimal_pattern, row['opcode_output'])
            goto_addresses = []
            for hex_num in hex_numbers:
                hex_val = re.sub('h$', '', hex_num)
                decimal_val = 2 * int(hex_val, 16)
                goto_address = row['idx'] + decimal_val
                goto_addresses.append(goto_address)
                df.at[idx, 'opcode_output'] = goto_addresses

    df.to_csv(file, index=False)

    return df


def get_called_apis_from_custom_methods(df_normalized, file):
    custom_methods_and_call_apis = df_normalized[(df_normalized['op_value'] >= 110) &
                                                 (df_normalized['op_value'] <= 120)]

    custom_methods_and_call_apis = custom_methods_and_call_apis[['method_name', 'opcode_output']]

    custom_methods_and_call_apis.to_csv(file, index=False)

    return custom_methods_and_call_apis


def get_api_set(custom_methods_and_called_apis, general_android_api_set,
                custom_method_set_output_file, android_apis_output_file):

    # All custom methods and API calls
    all_called_methods_and_apis_set = set(custom_methods_and_called_apis['method_name'].tolist() +
                                          custom_methods_and_called_apis['opcode_output'].tolist())

    # All Android APIs
    all_android_api_set = all_called_methods_and_apis_set.intersection(general_android_api_set)

    # Custom method set
    custom_method_set = all_called_methods_and_apis_set - all_android_api_set

    # Write sets to files
    custom_method_set = sorted(custom_method_set)
    android_api_set = sorted(all_android_api_set)
    with open(custom_method_set_output_file, 'w', encoding="utf-8") as f1, \
            open(android_apis_output_file, 'w', encoding="utf-8") as f2:
        f1.writelines('\n'.join(custom_method_set))
        f2.writelines('\n'.join(android_api_set))


def sum_up_file_content(input_dir, critical_apis):
    line_count_dict = {}  # Dictionary to store line counts

    # Iterate over all text files in the folder
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    for line in file:
                        line = line.strip()
                        if line in line_count_dict:
                            line_count_dict[line]["Usage Count"] += 1
                        else:
                            is_dangerous = "Yes" if line in critical_apis else "No"
                            line_count_dict[line] = {"Usage Count": 1, "Dangerous API?": is_dangerous}
            except Exception as e:
                print(f"Exception in sum_up_file_content: {e}")

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(list(line_count_dict.values()))
    df.index = list(line_count_dict.keys())  # Set the line as the index
    df.index.name = "Line"
    df = df.reset_index()

    # Sort the DataFrame by the "Usage Count" column
    df = df.sort_values(by="Usage Count", ascending=False)

    return df


def merge_bw_and_mw_android_apis(benignware_df, malware_df):
    # Merge the two DataFrames on the 'Line' column
    merged_df = pd.merge(
        benignware_df[['Line', 'Usage Count', 'Dangerous API?']],
        malware_df[['Line', 'Usage Count']],
        on='Line',
        how='outer',
        suffixes=('_BW', '_MW')
    )

    # Fill NaN values with 0 for usage counts
    merged_df['Usage Count_BW'].fillna(0, inplace=True)
    merged_df['Usage Count_MW'].fillna(0, inplace=True)

    # If Dangerous API is NaN, fill it with 'No'
    merged_df['Dangerous API?'].fillna('No', inplace=True)

    # Rename columns to desired names
    merged_df.rename(columns={
        'Usage Count_BW': 'BW Usage Count',
        'Usage Count_MW': 'MW Usage Count'
    }, inplace=True)

    # Add Total Usage Count column
    merged_df['Total Usage Count'] = merged_df['BW Usage Count'] + merged_df['MW Usage Count']

    # Reorder columns
    merged_df = merged_df[['Line', 'BW Usage Count', 'MW Usage Count', 'Total Usage Count', 'Dangerous API?']]

    return merged_df


def choose_api_list(merged_df, list_file, dict_file):
    # Select rows where 'Dangerous API?' is 'Yes'
    dangerous_apis = merged_df[merged_df['Dangerous API?'] == 'Yes']

    # Select rows where 'Total Usage Count' is greater than or equal to 10
    dangerous_apis = dangerous_apis[dangerous_apis['Total Usage Count'] >= 10]

    # Select the 'Line' column and sort it in alphabetical order
    api_list = dangerous_apis['Line'].sort_values().tolist()

    # Write the list to a text file
    with open(list_file, 'w', encoding="utf-8") as f:
        for api in api_list:
            f.write(f"{api}\n")

    print(f"{list_file} is created successfully.")

    # Create a dictionary with the API as key and its index as value
    api_dict = {api: idx for idx, api in enumerate(api_list)}

    # Write the dictionary to a text file
    with open(dict_file, 'w', encoding="utf-8") as f:
        for api, idx in api_dict.items():
            f.write(f"{api}: {idx}\n")

    print(f"{dict_file} is created successfully.")


def write_into_excel(df, output_excel_file):
    # Create a new workbook and add the DataFrame to a worksheet
    wb = Workbook()
    ws = wb.active
    for row in dataframe_to_rows(df, index=False, header=True):
        ws.append(row)

    # Add filters to the worksheet
    ws.auto_filter.ref = ws.dimensions

    # Adjust column widths
    ws.column_dimensions['A'].width = 80
    ws.column_dimensions['B'].width = 20
    ws.column_dimensions['C'].width = 20
    ws.column_dimensions['D'].width = 20
    ws.column_dimensions['E'].width = 20

    # Save the workbook to an Excel file
    wb.save(output_excel_file)

    print(f"{output_excel_file} is created successfully.")


def obtain_api_dict(selected_apis_used_in_apks, selected_apis_dictionary_file_path):
    # Dictionary of selected APIs used in APKs
    api_id = 0
    api_dict = {}
    with open(selected_apis_used_in_apks, 'r') as file:
        for line in file:
            api_name = line.strip()
            api_dict[api_name] = api_id
            api_id += 1
    with open(selected_apis_dictionary_file_path, "w") as file:
        for key, value in api_dict.items():
            # Write each key-value pair on a separate line
            file.write(f"{key}: {value}\n")

    return api_dict


def find_remaining_apks_for_third_stage(input_dir_whole, output_dir_existing, apk_group):
    # Whole apk set
    directory_whole_files = os.path.join(input_dir_whole, apk_group)
    whole_apk_set = set([file_name[:-4] for file_name in os.listdir(directory_whole_files)])

    # Existing output files
    directory_existing_files = os.path.join(output_dir_existing, apk_group)
    existing_output_files = set()
    for output_file in os.listdir(directory_existing_files):
        if output_file.endswith(".json"):
            existing_output_files.add(os.path.splitext(output_file)[0])

    # Remaining APK files
    remaining_apk_files = whole_apk_set - existing_output_files
    print("Number of remaining apk files for", apk_group[:-1], "is:\t", len(remaining_apk_files))
    return remaining_apk_files


def all_steps_for_api_cg_creation(apk_group, remaining_apk_files, android_api_dir, custom_method_dir,
                                  df_normalized_dir, api_call_graphs_json_dir, api_call_graph_type, selected_api_dict,
                                  apk_type):

    # Analyze each APK file
    for apk_name in remaining_apk_files:
        try:
            print("Analyzing", apk_name)

            # Get Android APIs used in the APK
            android_api_path = os.path.join(android_api_dir, apk_group, apk_name + '.txt')
            with open(android_api_path, 'r') as f:
                android_apis = set(f.read().splitlines())
                print("number of Android APIs used in the APK:", len(android_apis))

            # Get selected Android APIs used in the APK
            dangerous_android_apis = android_apis.intersection(selected_api_dict)

            # Get custom methods used in the apk
            custom_method_path = os.path.join(custom_method_dir, apk_group, apk_name + '.txt')
            with open(custom_method_path, 'r', encoding='utf-8') as f:
                custom_methods = set(f.read().splitlines())
            print("number of custom_methods:", len(custom_methods))

            if len(custom_methods) < 2000000:
                # Get df_normalized
                df_normalized_path = os.path.join(df_normalized_dir, apk_group, apk_name + '.csv')
                df_normalized = pd.read_csv(df_normalized_path)
                df_normalized['opcode_output'] = df_normalized['opcode_output'].apply(deserialize_opcode_output)

                # Generate control flow graph
                graph = generate_cfg(df_normalized, custom_methods, android_apis)

                # Create output file paths
                api_cg_path = os.path.join(api_call_graphs_json_dir, apk_group, apk_name + '.json')

                # Create API Call Graphs
                if api_call_graph_type == "Small":
                    get_api_call_graph(apk_type, graph, selected_api_dict, dangerous_android_apis, api_cg_path)
                elif api_call_graph_type == "Big":
                    get_homogeneous_api_call_graph(apk_type, graph, selected_api_dict, dangerous_android_apis, api_cg_path)
            else:
                print("This apk is passed beacuse of its high custom method number")
        except Exception as e:
            print(f"Exception: {e}")


def deserialize_opcode_output(cell_value):
    if isinstance(cell_value, float) and math.isnan(cell_value):
        return cell_value
    try:
        if cell_value.startswith('['):
            return ast.literal_eval(cell_value)
        else:
            return cell_value
    except (ValueError, SyntaxError):
        return cell_value


def generate_cfg(df, custom_method_set, android_api_set):
    # create a new graph with only the critical APIs and their edges
    graph = nx.DiGraph()

    # add nodes for custom method calls
    for api in custom_method_set:
        return_api = f"return-{api}"
        graph.add_node(api)
        graph.add_node(return_api)

    # add nodes for Android API calls used in the APK
    graph.add_nodes_from(android_api_set)

    grouped = df.groupby('method_name')
    for method_name, sub_df in grouped:
        graph = generate_subgraph(graph, method_name, sub_df, android_api_set, custom_method_set)

    # Find isolated nodes
    isolated_nodes = [node for node in graph if graph.degree(node) == 0]

    # Remove isolated nodes from the graph
    graph.remove_nodes_from(isolated_nodes)

    # write_graph_info(graph, textfile, graphml)

    return graph


def generate_subgraph(graph, current_method, sub_df, android_api_set, custom_method_set):
    return_op_value_ranges = {(14, 17)}
    goto_op_value_ranges = {(40, 42)}
    if_else_switch_op_value_ranges = {(43, 44), (50, 61)}
    invoke_op_value_ranges = {(110, 120)}

    last_node = current_method
    return_method = f"return-{current_method}"
    for index, row in sub_df.iterrows():
        # "invoke" instructions
        if any(start <= row[2] <= end for start, end in invoke_op_value_ranges):
            api_call = row[4]
            if api_call in android_api_set:
                graph.add_edge(last_node, api_call)
                last_node = api_call
            elif api_call in custom_method_set:
                graph.add_edge(last_node, api_call)
                return_api_call = f"return-{api_call}"
                last_node = return_api_call

        # "return" instructions
        elif any(start <= row[2] <= end for start, end in return_op_value_ranges):
            graph.add_edge(last_node, return_method)

        # "if-else" and "switch" instructions
        elif any(start <= row[2] <= end for start, end in if_else_switch_op_value_ranges):
            branch_rows = []
            visited_rows = set()
            for row_index in row[4]:
                filtered_df = sub_df[sub_df['idx'] >= row_index]
                if not filtered_df.empty:
                    nearest_row = filtered_df.iloc[0]
                    if nearest_row[1] not in visited_rows:
                        visited_rows.add(nearest_row[1])
                        branch_rows.append(nearest_row)
            for branch_row in branch_rows:
                if any(start <= branch_row[2] <= end for start, end in invoke_op_value_ranges):
                    api_call = branch_row[4]
                    graph.add_edge(last_node, api_call)
                elif any(start <= branch_row[2] <= end for start, end in return_op_value_ranges):
                    graph.add_edge(last_node, return_method)
                elif any(start <= branch_row[2] <= end for start, end in if_else_switch_op_value_ranges):
                    filtered_next_row_add = sub_df[sub_df['idx'] > branch_row[1]]
                    if not filtered_next_row_add.empty:
                        next_row_add = filtered_next_row_add.iloc[0]
                        if next_row_add[1] not in visited_rows:
                            visited_rows.add(next_row_add[1])
                            branch_rows.append(next_row_add)
                    for branch_row_index in branch_row[4]:
                        filtered_branch_df = sub_df[sub_df['idx'] >= branch_row_index]
                        if not filtered_branch_df.empty:
                            nearest_row_add = filtered_branch_df.iloc[0]
                            if nearest_row_add[1] not in visited_rows:
                                visited_rows.add(nearest_row_add[1])
                                branch_rows.append(nearest_row_add)

        # "go-to" instructions
        elif any(start <= row[2] <= end for start, end in goto_op_value_ranges):
            goto_rows = []
            visited_goto = set()
            goto_row_index = row[4][0]
            filtered_df = sub_df[sub_df['idx'] >= goto_row_index]
            if not filtered_df.empty:
                goto_row = filtered_df.iloc[0]
                if goto_row[1] not in visited_goto:
                    visited_goto.add(goto_row[1])
                    goto_rows.append(goto_row)
            for goto_row in goto_rows:
                if any(start <= goto_row[2] <= end for start, end in invoke_op_value_ranges):
                    api_call = goto_row[4]
                    graph.add_edge(last_node, api_call)
                elif any(start <= goto_row[2] <= end for start, end in return_op_value_ranges):
                    graph.add_edge(last_node, return_method)
                elif any(start <= goto_row[2] <= end for start, end in goto_op_value_ranges):
                    new_goto_row_index = goto_row[4][0]
                    filtered_new_goto_row = sub_df[sub_df['idx'] >= new_goto_row_index]
                    if not filtered_new_goto_row.empty:
                        new_goto_row = filtered_new_goto_row.iloc[0]
                        if new_goto_row[1] not in visited_goto:
                            visited_goto.add(new_goto_row[1])
                            goto_rows.append(new_goto_row)
                elif any(start <= goto_row[2] <= end for start, end in if_else_switch_op_value_ranges):
                    current_row_index = goto_row[1]
                    filtered_next_row = sub_df[sub_df['idx'] > current_row_index]
                    if not filtered_next_row.empty:
                        next_row = filtered_next_row.iloc[0]
                        if next_row[1] not in visited_goto:
                            visited_goto.add(next_row[1])
                            goto_rows.append(next_row)
                    for row_index in goto_row[4]:
                        filtered_df = sub_df[sub_df['idx'] >= row_index]
                        if not filtered_df.empty:
                            nearest_row = filtered_df.iloc[0]
                            if nearest_row[1] not in visited_goto:
                                visited_goto.add(nearest_row[1])
                                goto_rows.append(nearest_row)

    return graph


def get_homogeneous_api_call_graph(apk_type, call_graph, given_api_dict, given_api_set, json_file):
    homogeneous_api_call_graph = nx.DiGraph()

    # set the graph-level attribute "is_malware"
    homogeneous_api_call_graph.graph['is_malware'] = apk_type  # 1 if malware, 0 if benignware

    # add nodes for critical Android API calls
    for api_name, api_id in given_api_dict.items():
        if api_name in given_api_set:
            attributes = {'api_name': api_name, 'api_id': api_id, 'color': 'red',
                          'is_used_in_program': 1}
        else:
            attributes = {'api_name': api_name, 'api_id': api_id, 'color': 'white',
                          'is_used_in_program': 0}
        homogeneous_api_call_graph.add_node(api_name, **attributes)

    # create edges
    for node in given_api_set:
        api_successors = set()
        visited = set()
        successor_list = []
        for neighbor in call_graph.successors(node):
            successor_list.append(neighbor)
        while successor_list:
            successor = successor_list[0]
            successor_list.remove(successor)
            if successor not in visited:
                visited.add(successor)
                if successor in given_api_set:
                    api_successors.add(successor)
                else:
                    new_successors = call_graph.successors(successor)
                    for item in new_successors:
                        successor_list.append(item)
        for target_api in api_successors:
            homogeneous_api_call_graph.add_edge(node, target_api)

    # relabel nodes with unique API ID
    homogeneous_critical_api_call_graph = nx.relabel_nodes(homogeneous_api_call_graph, given_api_dict)

    # write_graph_info(homogeneous_critical_api_call_graph, textfile, graphml)

    api_call_graph_json(apk_type, homogeneous_critical_api_call_graph, json_file)


def write_graph_info(graph, textfile, graphml):
    # create a list of dictionaries to store the edge information
    edge_info = []

    # edge information
    for i, edge in enumerate(graph.edges()):
        edge_info.append({
            'Edge Number': i,
            'Source Method': str(edge[0]),
            'Target Method': str(edge[1])
        })

    # create a pandas DataFrame from the edge information
    df = pd.DataFrame(edge_info)

    df.to_csv(textfile, sep='\t', index=False)

    # write graph into graphml file
    nx.write_graphml(graph, graphml)


def api_call_graph_json(apk_type, graph, json_file):
    # get the method names for the node labels
    node_labels = {}
    for i, node in enumerate(graph.nodes()):
        node_labels[i] = node

    # get edge information
    edges = []
    for source, target in graph.edges():
        edges.append([list(graph.nodes()).index(source), list(graph.nodes()).index(target)])

    # create dictionary to store data
    data = {"target": apk_type, "edges": edges, "labels": {}, "inverse_labels": {}}

    # add labels for each node
    for node, label in node_labels.items():
        data["labels"][str(node)] = label

    # add inverse labels
    for node, label in node_labels.items():
        if label not in data["inverse_labels"]:
            data["inverse_labels"][label] = []
        data["inverse_labels"][label].append(node)

    # write data to JSON file
    with open(json_file, "w") as f:
        json.dump(data, f)


def get_api_call_graph(apk_type, call_graph, given_api_dict, given_api_set, json_file):
    api_call_graph = nx.DiGraph()

    # set the graph-level attribute "is_malware"
    api_call_graph.graph['is_malware'] = apk_type  # 1 if malware, 0 if benignware

    # add nodes for critical Android API calls
    for api_name, api_id in given_api_dict.items():
        if api_name in given_api_set:
            attributes = {'api_name': api_name, 'api_id': api_id, 'color': 'red',
                          'is_used_in_program': 1}
            api_call_graph.add_node(api_name, **attributes)

    # create edges
    for node in given_api_set:
        api_successors = set()
        visited = set()
        successor_list = []
        if call_graph.has_node(node):
            for neighbor in call_graph.successors(node):
                successor_list.append(neighbor)
            while successor_list:
                successor = successor_list[0]
                successor_list.remove(successor)
                if successor not in visited:
                    visited.add(successor)
                    if successor in given_api_set:
                        api_successors.add(successor)
                    else:
                        new_successors = call_graph.successors(successor)
                        for item in new_successors:
                            successor_list.append(item)
            for target_api in api_successors:
                api_call_graph.add_edge(node, target_api)

    # relabel nodes with unique API ID
    critical_api_call_graph = nx.relabel_nodes(api_call_graph, given_api_dict)

    # write_graph_info(critical_api_call_graph, textfile, graphml)

    api_call_graph_json(apk_type, critical_api_call_graph, json_file)


def convert_json_to_pyg(json_data):
    # Step 1: Extract edges and convert to tensor
    edges = torch.tensor(json_data["edges"], dtype=torch.long).t().contiguous()  # Transpose to [2, num_edges]

    # Step 2: Extract node labels (features) and convert to tensor
    num_nodes = len(json_data["labels"])
    node_labels = torch.tensor([json_data["labels"][str(i)] for i in range(num_nodes)], dtype=torch.long)

    # Step 3: Extract target (graph-level label)
    target = torch.tensor([json_data["target"]], dtype=torch.long)

    # Step 4: Create PyTorch Geometric Data object
    data = Data(x=node_labels.view(-1, 1), edge_index=edges, y=target)

    return data


def load_and_merge_gam_and_gat_detection(gam_path, gat_path):
    gat_df = pd.read_excel(gat_path)
    gam_df = pd.read_csv(gam_path)

    # Clean names
    gat_df["Graph_Name"] = gat_df["Graph_Name"].apply(lambda x: os.path.splitext(x)[0])
    gam_df["Graph_Name"] = gam_df["graph_id"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

    # Select/rename GAM
    gam_df = gam_df.rename(
        columns={
            "true_label": "Actual_Label",
            "predicted_label": "GAM_Predicted_Label",
            "neuron_1_value": "GAM_Neuron_1_Value",
            "neuron_2_value": "GAM_Neuron_2_Value",
        }
    )[["Graph_Name", "Actual_Label", "GAM_Predicted_Label", "GAM_Neuron_1_Value", "GAM_Neuron_2_Value"]]

    # Select/rename GAT
    gat_df = gat_df.rename(
        columns={
            "Predicted_Label": "GAT_Predicted_Label",
            "Probability": "GAT_Probabilities",
        }
    )[["Graph_Name", "Actual_Label", "GAT_Predicted_Label", "GAT_Probabilities"]]

    # Merge
    df = pd.merge(gat_df, gam_df, on=["Graph_Name", "Actual_Label"], how="inner")

    return df


def gam_and_gat_detection_fusion(df):
    def to_bool(val):
        if isinstance(val, str):
            return val.lower() in ["true", "1", "yes", "positive"]
        return bool(val)

    df["GAT_Predicted_Label"] = df["GAT_Predicted_Label"].apply(to_bool)
    df["GAM_Predicted_Label"] = df["GAM_Predicted_Label"].apply(to_bool)
    df["Actual_Label"] = df["Actual_Label"].apply(to_bool)

    # AND/OR rules
    df["Fusion_AND"] = df["GAT_Predicted_Label"] & df["GAM_Predicted_Label"]
    df["Fusion_OR"] = df["GAT_Predicted_Label"] | df["GAM_Predicted_Label"]

    # Softmax for GAM logits
    logits = df[["GAM_Neuron_1_Value", "GAM_Neuron_2_Value"]].values
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    df["gam_prob"] = probs[:, 1]

    # Weighted fusion
    alpha = 0.5
    df["Fusion_Weighted_Score"] = alpha * df["GAT_Probabilities"] + (1 - alpha) * df["gam_prob"]
    df["Fusion_Weighted_Label"] = df["Fusion_Weighted_Score"] >= 0.5

    return df


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1, "FPR": fpr, "FNR": fnr}


def rename_detection_excel_columns(df):
    df = df.rename(columns={
        "Graph_Name": "graph",
        "Actual_Label": "actual",
        "GAT_Predicted_Label": "gat_pred",
        "GAM_Predicted_Label": "gam_pred",
        "Fusion_AND": "fusion_and",
        "Fusion_OR": "fusion_or",
        "Fusion_Weighted_Label": "fusion_weighted",
        "GAT_Probabilities": "gat_prob",
        "Fusion_Weighted_Score": "fusion_weighted_score",
        "gam_prob": "gam_prob"
    })

    df["actual"] = df["actual"].astype(int)
    for col in ["gat_pred", "gam_pred", "fusion_and", "fusion_or", "fusion_weighted"]:
        df[col] = df[col].astype(int)

    return df


def save_detection_metrics(df, predictors, metrics_file):
    with open(metrics_file, "w") as f:
        f.write("in-distribution detection results\n")
        f.write("\t\tFPR\tFNR\tAccuracy\tPrecision\tRecall\tF1-score\n")
        for name, (pred_col, prob_col) in predictors.items():
            id_data = df[~df["graph"].str.startswith("mw_androzoo_")]
            if not id_data.empty:
                metrics = compute_metrics(id_data["actual"], id_data[pred_col])
                f.write(f"{name}\t{metrics['FPR']:.4f}\t{metrics['FNR']:.4f}\t"
                        f"{metrics['Accuracy']:.4f}\t{metrics['Precision']:.4f}\t"
                        f"{metrics['Recall']:.4f}\t{metrics['F1-Score']:.4f}\n")

        f.write("\n\nout-of-distribution detection results\n")
        f.write("\t\tFNR\tRecall\n")
        for name, (pred_col, prob_col) in predictors.items():
            ood = df[df["graph"].str.startswith("mw_androzoo_")]
            if not ood.empty:
                metrics = compute_metrics(ood["actual"], ood[pred_col])
                f.write(f"{name}\t\t\t{metrics['FNR']:.4f}\t"
                        f"{metrics['Recall']:.4f}\n")

        # --- Subset Metrics (AND) ---
        f.write("\n\n=== Subset Metrics (AND) ===\n")
        prefixes = ["adware_", "banking_", "bw_", "mw_amd", "mw_androzoo_", "mw_mystique", "riskware", "sms"]
        for prefix in prefixes:
            subset = df[df["graph"].str.startswith(prefix)]
            if not subset.empty:
                metrics = compute_metrics(subset["actual"], subset["fusion_and"])
                f.write(f"\nSubset: {prefix.rstrip('_')}\n")
                for k, v in metrics.items():
                    f.write(f"{k}: {v:.4f}\n")

    print(f"Metrics report saved.")


def plot_roc_and_pr_curves(df, predictors, roc_curve_file, pr_curve_file):
    colors = {"GAT": "blue",
              "GAM": "green",
              "Fusion_Weighted": "red",
              "Fusion_AND": "darkred",
              "Fusion_OR": "black"}

    # Filter only ID samples
    id_df = df[~df["graph"].str.startswith("mw_androzoo_")]

    # ROC
    plt.figure(figsize=(7, 6))
    for name, (pred_col, prob_col) in predictors.items():
        if prob_col:
            fpr, tpr, _ = roc_curve(id_df["actual"], id_df[prob_col])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})", color=colors.get(name, None))

    for name in ["Fusion_AND", "Fusion_OR"]:
        y_pred = id_df[name.lower()]
        tn, fp, fn, tp = confusion_matrix(id_df["actual"], y_pred, labels=[0, 1]).ravel()
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        plt.scatter(fpr_val, tpr_val, marker="o", color=colors.get(name, None), s=100, label=f"{name} (point)")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Baseline (Random Classifier)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(roc_curve_file)
    plt.close()

    # PR
    plt.figure(figsize=(7, 6))
    for name, (pred_col, prob_col) in predictors.items():
        if prob_col:
            prec, rec, _ = precision_recall_curve(id_df["actual"], id_df[prob_col])
            plt.plot(rec, prec, label=name, color=colors.get(name, None))

    for name in ["Fusion_AND", "Fusion_OR"]:
        y_pred = id_df[name.lower()]
        tp = ((id_df["actual"] == 1) & (y_pred == 1)).sum()
        fp = ((id_df["actual"] == 0) & (y_pred == 1)).sum()
        fn = ((id_df["actual"] == 1) & (y_pred == 0)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        plt.scatter(recall, precision, marker="o", color=colors.get(name, None), s=100, label=f"{name} (point)")

    pos_rate = id_df["actual"].mean()
    plt.axhline(y=pos_rate, color="gray", linestyle="--", label="Baseline (Random Classifier)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.ylim(0.5, )
    plt.savefig(pr_curve_file)
    plt.close()

    print(f"ROC and PR curves saved.")
