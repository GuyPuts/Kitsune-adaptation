import csv
import math
import pickle
from math import floor

import numpy as np
from scapy.utils import rdpcap
from random import sample
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2

from KitPlugin import KitPlugin

inputs = {
    "mirai_malicious": {
        "input_path": "input_data/mirai.pcap",
        "input_path_test": "input_data/mirai.pcap",
        "packet_limit": 200000,
        "maxAE": 10,
        "FMgrace": 5000,
        "ADgrace": 50000,
        "training_min": 0,
        "training_max": 60000,
        "testing_min": 140330,
        "testing_max": 140355
    },
    "mirai_benign": {
        "input_path": "input_data/mirai.pcap",
        "input_path_test": "input_data/mirai.pcap",
        "packet_limit": 200000,
        "maxAE": 10,
        "FMgrace": 5000,
        "ADgrace": 50000,
        "training_min": 0,
        "training_max": 60000,
        "testing_min": 70000,
        "testing_max": 70025
    }
}


# KitPlugin = KitPlugin()
# KitPlugin.hyper_opt("input_data/Monday-WorkingHours_10_percent_random.pcap", 100, 1000000)

# Run series of statistics
# KitPlugin.run_series_stats(inputs)
# Get feature list from pickle file
# KitPlugin.feature_loader()
# Train Kitsune on the training data
# KitPlugin.kit_trainer(training_min, training_max)
# Calculate SHAP-values
# KitPlugin.shap_values_builder(training_min, training_max, testing_min, testing_max)
# Pickle SHAP-values
# KitPlugin.shap_values_pickle()
# Load SHAP-values
# KitPlugin.shap_values_loader()
# Calculate shap summary statistics
# KitPlugin.shap_stats_summary_builder(testing_min, testing_max)
# KitPlugin.shap_stats_excel_export()

# Sample 10 percent of conversations
# SampleKitPlugin = KitPlugin()
##conversations = SampleKitPlugin.sample_percentage_conversations(10, "input_data/Monday_Split/17_01-18_01.pcapng", "input_data/Monday_Split/17_01-18_01-sample-10.pcap")
# conversations = SampleKitPlugin.conversations_loader('pickles/17_01-18_01_sample_10_conv')
# packets = rdpcap('input_data/Monday_Split/17_01-18_01-sample-10.pcap')
# features = SampleKitPlugin.load_pcap_to_features('input_data/Monday_Split/17_01-18_01-sample-10.pcap')

# print('conversations: '+str(len(conversations)))
# print('packets: '+str(len(packets)))
# print('feature lists: '+str(len(features)))

# del SampleKitPlugin

# NewKitPlugin = KitPlugin('input_data/Monday_Split/17_01-18_01-sample-10.pcap', num_autenc=6, FMgrace=int(0.05*len(features)), int(0.95*len(ADgrace)))

# Train Kitsune
# NewKitPlugin.kit_trainer_supplied_features(features)

# Open TSV file and extract features

# KitPlugin = KitPlugin(input_path='input_data/Wednesday-WorkingHours.pcap', packet_limit=np.Inf, num_autenc=6, FMgrace=11000000, ADgrace=12000000, learning_rate=0.1, hidden_ratio=0.75)
# features = KitPlugin.feature_builder('input_data/wednesday_features.csv')
# KitPlugin.feature_pickle()

def kitTester(day, attack_type):
    from KitPlugin import KitPlugin
    kitplugin = KitPlugin()
    print(f'reading labels file {day} {attack_type}')
    labels = kitplugin.read_label_file(f'input_data/attack_types/{day}_{attack_type}.csv')
    iter = 0
    for label in labels:
        if iter == 0:
            iter += 1
            continue
        label.append(str(labels.index(label)))
    print('sampling packets by conversation')
    kitplugin.sample_packets_by_conversation(f'input_data/{day.title()}-WorkingHours.pcap.tsv',
                                             f'input_data/attack_types/{day}_{attack_type}.pcap.tsv', labels)
    # Map samples to features of an existing featureList
    print('mapping packets to conversations')
    kitplugin.map_packets_to_features(f'input_data/attack_types/{day}_{attack_type}.pcap.tsv',
                                      f'input_data/attack_types/{day}_features.csv',
                                      f'input_data/attack_types/{day}_features_{attack_type}.csv')
    return True
    results = kitplugin.run_trained_kitsune_from_feature_csv(
        f"input_data/attack_types/{day}_features_{attack_type}.csv", 0, np.Inf)
    plt.title(f'{day.title()}_{attack_type}')
    plt.plot(results)
    plt.show()
    with open(f'pickles/output_pickles_packet_basis/{day.title()}_{attack_type}_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    convs = kitplugin.map_results_to_conversation(results, f"input_data/attack_types/{day}_{attack_type}.pcap.tsv")

    maxConvs = []
    for conv in convs:
        maxConvs.append(np.percentile(convs[conv], 75))

    path = f'pickles/output_pickles_conv_basis/{day.title()}_{attack_type}_maxConvs.pkl'
    with open(path, 'wb') as f:
        pickle.dump(maxConvs, f)

# kitplugin = KitPlugin(input_path="input_data/Friday-WorkingHours.pcap", packet_limit=np.Inf, num_autenc=50, FMgrace=None, ADgrace=None, learning_rate=0.1, hidden_ratio=0.75)
# kitplugin.feature_builder("input_data/attack_types/friday_features.csv")

day = "monday"
attack_name = "sample_medium_10"
#kitTester(day, attack_name)

kitplugin = KitPlugin()
kitplugin.train_kitsune()
#kitplugin.most_significant_packets_sampler("thursday")
#results = kitplugin.shap_documenter("thursday")

# with open(f"input_data/Wednesday-WorkingHours.pcap.tsv", newline='') as csvfile:
#     csv_reader = csv.reader(csvfile)
#     line_count = sum(1 for row in csv_reader)
# print(f"PCAP: {line_count}")
with open(f"input_data/attack_types/monday_features_sample_medium_10.csv", newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    line_count = sum(1 for row in csv_reader)
print(f"PCAP: {line_count}")

print(f"Optimizing sample {attack_name}")
kitplugin.hyper_opt_KitNET("monday", attack_name, line_count-1)
