import os
import pickle

import numpy as np

import matplotlib.pyplot as plt

from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference

from KitPlugin import KitPlugin
from FeatureExtractor import FE
from Kitsune import Kitsune

def kitTester(day, attack_type, newFeatures=False):
    from KitPlugin import KitPlugin
    kitplugin = KitPlugin()
    print('reading labels file')
    labels = kitplugin.read_label_file(f'input_data/attack_types/{day}_{attack_type}.csv')
    iter = 0
    for label in labels:
        if iter == 0:
            iter += 1
            continue
        label.append(str(labels.index(label)))

    if newFeatures:
        if not os.path.exists(f'input_data/{newFeatures}'):
            os.makedirs(f'input_data/{newFeatures}')
            print(f"Directory 'input_data/{newFeatures}' created successfully.")
        else:
            print(f"Directory 'input_data/{newFeatures}' already exists.")
        if not os.path.exists(f'input_data/{newFeatures}/attack_types'):
            os.makedirs(f'input_data/{newFeatures}/attack_types')
            print(f"Directory 'input_data/{newFeatures}/attack_types' created successfully.")
        else:
            print(f"Directory 'input_data/{newFeatures}/attack_types' already exists.")

    print('sampling packets by conversation')
    if newFeatures:
        kitplugin.sample_packets_by_conversation(f'input_data/{day.title()}-WorkingHours.pcap.tsv',
                                                 f'input_data/{newFeatures}/attack_types/{day}_{attack_type}.pcap.tsv', labels)
    else:
        kitplugin.sample_packets_by_conversation(f'input_data/{day.title()}-WorkingHours.pcap.tsv',
                                             f'input_data/attack_types/{day}_{attack_type}.pcap.tsv', labels)

    # Map samples to features of an existing featureList
    if newFeatures:
        with open(f'input_data/{newFeatures}/attack_types/{day}_{attack_type}.pcap.tsv', 'r') as file:
            lines = file.readlines()
        # Remove blank lines
        non_blank_lines = [line for line in lines if line.strip()]
        with open(f'input_data/{newFeatures}/attack_types/{day}_{attack_type}.pcap.tsv', 'w') as file:
            file.writelines(non_blank_lines)
        fe = FE(f'input_data/{newFeatures}/attack_types/{day}_{attack_type}.pcap.tsv')
        fe.get_all_vectors(f'input_data/{newFeatures}/attack_types/{day}_features_{attack_type}.csv')
    else:
        kitplugin.map_packets_to_features(f'input_data/attack_types/{day}_{attack_type}.pcap.tsv',
                                      f'input_data/attack_types/{day}_features.csv',
                                      f'input_data/attack_types/{day}_features_{attack_type}.csv')

    if newFeatures:
        if not os.path.exists(f'pickles/{newFeatures}'):
            os.makedirs(f'pickles/{newFeatures}')
            print(f"Directory 'pickles/{newFeatures}' created successfully.")
        else:
            print(f"Directory 'pickles/{newFeatures}' already exists.")
        oops_we_have_to_train_kitsune_again('input_data/attack_types/monday_sample_medium_15.pcap.tsv', newFeatures)
    if newFeatures:
        results = kitplugin.run_trained_kitsune_from_feature_csv(
            f"input_data/{newFeatures}/attack_types/{day}_features_{attack_type}.csv", 0, np.Inf, kit_path=f"pickles/{newFeatures}/anomDetector.pkl")
    else:
        results = kitplugin.run_trained_kitsune_from_feature_csv(
            f"input_data/attack_types/{day}_features_{attack_type}.csv", 0, np.Inf)

    if newFeatures:
        if not os.path.exists(f'pickles/{newFeatures}/output_pickles_packet_basis'):
            os.makedirs(f'pickles/{newFeatures}/output_pickles_packet_basis')
            print(f"Directory 'pickles/{newFeatures}/output_pickles_packet_basis' created successfully.")
        else:
            print(f"Directory 'pickles/{newFeatures}/output_pickles_packet_basis' already exists.")
        if not os.path.exists(f'pickles/{newFeatures}/output_pickles_conv_basis'):
            os.makedirs(f'pickles/{newFeatures}/output_pickles_conv_basis')
            print(f"Directory 'pickles/{newFeatures}/output_pickles_conv_basis' created successfully.")
        else:
            print(f"Directory 'pickles/{newFeatures}/output_pickles_conv_basis' already exists.")

    if newFeatures:
        with open(f'pickles/{newFeatures}/output_pickles_packet_basis/{day.title()}_{attack_type}_results.pkl', 'wb') as f:
            pickle.dump(results, f)
    else:
        with open(f'pickles/output_pickles_packet_basis/{day.title()}_{attack_type}_results.pkl', 'wb') as f:
            pickle.dump(results, f)

    if newFeatures:
        convs = kitplugin.map_results_to_conversation(results, f"input_data/{newFeatures}/attack_types/{day}_{attack_type}.pcap.tsv")
    else:
        convs = kitplugin.map_results_to_conversation(results, f"input_data/attack_types/{day}_{attack_type}.pcap.tsv")
    print(f"attack: {attack_type}, convs: {len(convs)}")
    maxConvs = []
    for conv in convs:
        maxConvs.append(np.max(convs[conv]))

    if newFeatures:
        path = f'pickles/{newFeatures}/output_pickles_conv_basis/{day.title()}_{attack_type}_maxConvs.pkl'
    else:
        path = f'pickles/output_pickles_conv_basis/{day.title()}_{attack_type}_maxConvs.pkl'
    with open(path, 'wb') as f:
        pickle.dump(maxConvs, f)
    return maxConvs

def oops_we_have_to_train_kitsune_again(path, newFeatures):
    if os.path.isfile(f"pickles/{newFeatures}/anomDetector.pkl"):
        return True
    newKitsOnTheBlock = Kitsune(path, np.Inf, 25, 23940, 239400, 0.00001, 0.25)
    for i in range(0, 239400):
        if i % 20000 == 0:
            print(f"Training KitNET, packet {i} of 239400")
        newKitsOnTheBlock.proc_next_packet()
    newkitnet = newKitsOnTheBlock.giveMeTheKit()
    with open(f"pickles/{newFeatures}/anomDetector.pkl", 'wb') as f:
        pickle.dump(newkitnet, f)

# attacks1 = ["sample_60"]
attacks1 = ["benign - small", "SSH-Patator - Attempted", "SSH-Patator", "FTP-Patator", "FTP-Patator - Attempted"]
# # attacks1 = ["sample_medium_15"]
# # attacks1 = ["benign - small"]
#attacks1 = ["benign - small", "Infiltration", "Infiltration - Attempted", "Infiltration - Portscan", "Web Attack - Brute Force - Attempted", "Web Attack - Brute Force", "Web Attack - SQL Injection", "Web Attack - SQL Injection - Attempted", "Web Attack - XSS", "Web Attack - XSS - Attempted"]
attacks1 = ["benign - small", "Botnet - Attempted", "Botnet", "DDoS", "Portscan"]
attacks1 = ["UNSW_Analysis", "UNSW_Backdoor", "UNSW_Exploits", "UNSW_Generic", "UNSW_Reconnaissance", "UNSW_Shellcode", "UNSW_Worms"]

attacks1 = ['UNSW_Benign_medium', 'UNSW_Benign_medium_validate', 'UNSW_Benign_medium_test']
convs = []
attacks1 = ["benign - small - sanity_check"]
convs = []

attacks1 = ['UNSW_Benign_medium_test']
convs = []


filename = 'input_data/attack_types/noday_features_UNSW_Benign_medium_validate.csv'  # Replace 'example.csv' with your CSV file name

# data = []
# with open(filename, 'r', newline='') as csvfile:
#     csvreader = csv.reader(csvfile)
#     for row in csvreader:
#         data.append(row)

# pickle_file = 'pickles/medium_validate.pkl'
# with open(pickle_file, 'wb') as f:
#     pickle.dump(data, f)

# kitplugin = KitPlugin()
#kitplugin.most_significant_packets_sampler("wednesday", 0.111966)
# #kitplugin.most_significant_packets_sampler("tuesday", 0.111966)
# results = kitplugin.shap_documenter("friday")
#results = kitplugin.shap_documenter("tuesday")
# attacks1 = ["UNSW_Benign_medium"]
# for sample in attacks1:
#     with open(f"input_data/attack_types/noday_features_UNSW_Benign_medium.csv", newline='') as csvfile:
#         csv_reader = csv.reader(csvfile)
#         line_count = sum(1 for row in csv_reader)
#     print(f'lines: {line_count}')
#     kitplugin=KitPlugin()
#     print(f'optimizing kitnet for {sample}')
#     kitplugin.hyper_opt_KitNET("noday", sample, line_count)
#     print('done optimizing')


import csv


def filter_csv_columns(input_path, column_indices, output_path):

    with open(input_path, 'r', newline='') as infile, open(output_path, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            filtered_row = [row[i] for i in column_indices]
            writer.writerow(filtered_row)

    return output_path


# input_file = "input_data/attack_types/tuesday_features.csv"
# output_file = "input_data/attack_types/tuesday_features_removed_tcp"
# columns_to_keep = [2, 5, 10]  # Example: Keep columns 2, 5, and 10
# filtered_file = filter_csv_columns(input_file, columns_to_keep, output_file)
# print("Filtered CSV file saved as:", filtered_file)
import csv

kitplugin = KitPlugin()
kitplugin.train_kitsune()
# convs = []
# attacks2 = ["test_attack"]
# for attack in attacks2:
#     print(attack)
#     convs.append(kitTester("monday", attack))