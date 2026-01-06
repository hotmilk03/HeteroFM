import re
import matplotlib.pyplot as plt

def parse_log_and_plot(file_paths, label_names, title, output_image):
    rounds = []
    accuracies = []

    pattern = re.compile(r"Round\s+(\d+).*?Accuracy:\s+([\d\.]+)%")

    # graph plotting
    plt.figure(figsize=(10, 6)) # Set figure size
    
    for i in range(len(file_paths)):
        file_path = file_paths[i]
        label = label_names[i]

        rounds = []
        accuracies = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    match = pattern.search(line)
                    if match:
                        round_num = int(match.group(1))
                        acc_val = float(match.group(2)) / 100.0
                        rounds.append(round_num)
                        accuracies.append(acc_val)

            if not rounds:
                print(f"[Skip] '{file_path}': Cannot find log file.")
                continue

            # Plot with points (o) and lines (-)
            # plt.plot(rounds, accuracies, marker='o', linestyle='-', label=label)

            lines = plt.plot(rounds, accuracies, marker='o', linestyle='-', label=label)
            line_color = lines[0].get_color()

            last_round = rounds[-1]
            last_acc = accuracies[-1]

            plt.text(last_round, last_acc, f"  {last_acc*100:.1f}%", color=line_color, va='center', fontweight='bold')

        except FileNotFoundError:
            print(f"Cannot find file : {file_path}")

    plt.title(title)
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.grid(True) # Show grid
    plt.legend()
    
    # plt.xlim(0, 100)
    plt.ylim(0, 1.0)
    
    plt.savefig(output_image)
    plt.clf()

log_filenames = [
    # 'results/perm/MC/251223_noniid_mlp_2layer_classPerClient=unique2_perm_MC.log',
    # 'results/perm/MC/251223_noniid_mlp_2layer_classPerClient=unique5_perm_MC.log',
    # 'results/perm/MC/251223_noniid_mlp_2layer_classPerClient=unique9_perm_MC.log',

    # 'results/perm/mlp_2layer/ME/251223_noniid_mlp_2layer_classPerClient=unique2_perm_ME.log',
    # 'results/perm/mlp_2layer/ME_lr=0.001/251223_noniid_mlp_2layer_classPerClient=unique2_perm_ME_lr=0.001.log'

    # 'results/HeteroFL/mlp_2layer/class_per_client_sbn_nodropout/251221_noniid_mlp_2layer_classPerClient=unique9_sbn.log',
    # 'results/perm/ME/251223_noniid_mlp_2layer_classPerClient=unique9_perm_ME.log',
    # 'results/perm/MC/251223_noniid_mlp_2layer_classPerClient=unique9_perm_MC.log',
    # 'results/perm/ZE/251223_noniid_mlp_2layer_classPerClient=unique9_perm_ZE.log',
    # 'results/perm/ZC/251223_noniid_mlp_2layer_classPerClient=unique9_perm_ZC.log'

    # 'results/HeteroFL/VGG/vgg_0.9_default.log',
    # 'results/perm/VGG/vgg_0.9_ME.log',
    # 'results/perm/VGG/vgg_0.9_MC.log',
    # 'results/perm/VGG/vgg_0.9_ZE.log',
    # 'results/perm/VGG/vgg_0.9_ZC.log'

    'results/HeteroFL/VGG/vgg_0.9_default.log',
    'vgg_0.9_ME.log',
    'vgg_0.9_MC.log',
    'vgg_0.9_ZE.log',
    'vgg_0.9_ZC.log'
]
label_names = [
    # '0.2 class ratio per client',
    # '0.5 class ratio per client',
    # '0.9 class ratio per client',

    # 'lr=0.01',
    # 'lr=0.001'
    
    'HeteroFL',
    'ME',
    'MC',
    'ZE',
    'ZC'
]

# output_image = 'noniid_mlp_2layer_by_perm_classPerClient=unique5.png'
# title = 'Global acc by perm (MLP 2-layer, non-iid, 5 class per client, lr=0.01)'
output_image = 'vgg-re.png' # 'noniid_mlp_2layer_perm_ME_by_lr_classPerClient=unique2.png'
title = 'VGG : HeteroFL vs Permutation Methods' # 'Acc with perm by lr (MLP 2-layer, non-iid, ME, 2 class per client)'
parse_log_and_plot(log_filenames, label_names, title, output_image)