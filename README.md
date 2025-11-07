# XAIDroid

This repository contains the codebase and data utilized in XAIDroid research.

## XAIDroid Codebase

The codebase is divided into nine stages:

#### Stage 1: APK Analysis and Smali File Creation

- Extracts information from benign and malicious APK files.
- Converts APKs into Smali files.
- Identifies custom methods and API calls used by the APKs.

#### Stage 2: API Call Collection and Filtering

- Collects API calls from malware and benign samples.
- Filters the collected API calls based on the sensitive API list.

#### Stage 3: API Call Graph Creation

- Constructs API call graphs for the analyzed APKs.

#### Stage 4: Convert JSON Graphs into PyTorch Graphs

- Converts JSON formatted graphs into PyTorch graph data format.

#### Stage 5: Graph Attention Model (GAM) Training

- This stage applies the Graph Attention Model to the API call graphs (in JSON format).
- **Note**: The code for this stage is directly used from [benedekrozemberczki/GAM](https://github.com/benedekrozemberczki/GAM) and is not included in this repository.

#### Stage 6: Graph Attention Network (GAT) Training

- This stage applies the Graph Attention Network to the API call graphs (in pt format).

#### Stage 7: Malware Detection with GAM and GAT

- This stage makes ensemble-detection using GAM and GAT results.
- Calculates accuracy, precision, recall and F1-score metrics of GAM, GAT and ensemble of GAM and GAT.

#### Stage 8: Method and Class Level Attention Calculation

- Calculates method-level and class-level attentions.

#### Stage 9: GAM and GAT MCL Metrics & Plots

- Optimizes the attention threshold for effective code localization.
- Performs method-level and class-level localization analysis.
- Computes performance metrics to assess localization accuracy.
- Generates ROC and Precision-Recall (PR) curves for evaluation.

#### Stage 10: Visualizations

- Visualizes API call graphs annotated with GAM and GAT attention values.
- Includes a colorbar to indicate attention intensity across nodes and edges.

## Data Folder Structure

### 1. Class and Method-Level Malicious Code Localization (MCL) Baselines

This folder contains the baselines for malicious class and method localization used in the XAIDroid research. These
baselines serve as benchmarks for evaluating the performance of our localization techniques.

### 2. XAIDroid MCL Results

These folders contain the class-level and method-level malicious code localization results of XAIDroid.

### 3. API Lists

- **Comprehensive API List**: Contains a detailed list of 3121 critical APIs.
- **Final API List**: A refined list of 688 APIs used in the XAIDroid analysis.

## Citation

If you use any part of this code or results in your research, please cite the XAIDroid research paper.

"Explainable Android Malware Detection and Malicious Code Localization Using Graph Attention"
Authors: Merve Cigdem IPEK and Sevil SEN
