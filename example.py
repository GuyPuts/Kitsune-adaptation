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

# for day in ['tuesday', 'wednesday', 'thursday', 'friday']:
#     print(f'running {day}')
#     kitplugin = KitPlugin(input_path=f"input_data/{day.title()}-WorkingHours.pcap.tsv", packet_limit=np.Inf, num_autenc=50, FMgrace=None, ADgrace=None, learning_rate=0.1, hidden_ratio=0.75)
#     kitplugin.feature_builder(f"input_data/attack_types/{day}_features_secondhalf.csv")
#     print(f'{day} done')
# quit()

# data = []
# with open(filename, 'r', newline='') as csvfile:
#     csvreader = csv.reader(csvfile)
#     for row in csvreader:
#         data.append(row)

# pickle_file = 'pickles/medium_validate.pkl'
# with open(pickle_file, 'wb') as f:
#     pickle.dump(data, f)

# attacks1 = ["UNSW_Benign_small", "UNSW_Benign_medium"]
# for sample in attacks1:
#     with open(f"input_data/attack_types/noday_features_{sample}.csv", newline='') as csvfile:
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
# columns_to_keep = [2, 5, 10]
# filtered_file = filter_csv_columns(input_file, columns_to_keep, output_file)
# print("Filtered CSV file saved as:", filtered_file)

import csv

# kitplugin = KitPlugin()
# kitplugin.train_kitsune()
# convs = []
# attacks2 = ["test_attack"]
# for attack in attacks2:
#     print(attack)
#     convs.append(kitTester("monday", attack))

# kitplugin = KitPlugin()
#kitplugin.most_significant_packets_sampler("wednesday", 0.111966)
# #kitplugin.most_significant_packets_sampler("tuesday", 0.111966)
# results = kitplugin.shap_documenter("friday")

def replace_entries(file1_path, file2_path, file3_path, file4_path, file5_path, file6_path, file7_path, file8_path, file9_path, output_path):
    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2, open(file3_path, 'r') as f3, open(file4_path, 'r') as f4, open(file5_path, 'r') as f5, open(file6_path, 'r') as f6, open(file7_path, 'r') as f7, open(file8_path, 'r') as f8, open(file9_path, 'r') as f9, open(output_path, 'w', newline='') as output_file:
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)
        reader3 = csv.reader(f3)
        reader4 = csv.reader(f4)
        reader5 = csv.reader(f5)
        reader6 = csv.reader(f6)
        reader7 = csv.reader(f7)
        reader8 = csv.reader(f8)
        reader9 = csv.reader(f9)
        writer = csv.writer(output_file)

        # Process File 1 and File 2
        for i, (row1, row2, row3, row4, row5, row6, row7, row8, row9) in enumerate(zip(reader1, reader2, reader3, reader4, reader5, reader6, reader7, reader8, reader9)):
            new_row = row1 + row2 + row3 + row4 + row5 + row6 + row7 + row8 + row9
            writer.writerow(new_row)

            # Optionally print current line number every 10,000 lines
            if i % 10000 == 0:
                print("Processed {} lines".format(i))

# file1_path = 'input_data/attack_types/wednesday_features_firsthalffirstpartfirst.csv'
# file2_path = 'input_data/attack_types/wednesday_features_hhjit_.csv'
# file3_path = 'input_data/attack_types/wednesday_features_hphp_5.csv'
# file4_path = 'input_data/attack_types/wednesday_features_hphp_3.csv'
# file5_path = 'input_data/attack_types/wednesday_features_hphp_1.csv'
# file6_path = 'input_data/attack_types/wednesday_features_hphp_01.csv'
# file7_path = 'input_data/attack_types/wednesday_features_hphp_0_01.csv'
# file8_path = 'input_data/attack_types/wednesday_features_firsthalfsecondpart.csv'
# file9_path = 'input_data/attack_types/wednesday_features_secondhalf.csv'
# output_path = 'input_data/attack_types/wednesday_features.csv'