import re
import matplotlib.pyplot as plt
import config

def plot_single_metric(file_paths, label_names, title, output_image, mode_type):
    """Plot a single metric (loss or acc) based on mode_type."""
    if mode_type == 'loss':
        pattern_main = re.compile(r"Round\s+(\d+).*?Test Loss:\s+([\d\.]+)")
        pattern_range = re.compile(r"Client Global Loss:\s+Min:\s+([\d\.]+),\s+Max:\s+([\d\.]+)")
        ylabel_name = 'Test Loss'
        is_percentage = False
    else:  # 'acc'
        pattern_main = re.compile(r"Round\s+(\d+).*?Accuracy:\s+([\d\.]+)%")
        pattern_range = re.compile(r"Client Global Accuracy:\s+Min:\s+([\d\.]+)%,\s+Max:\s+([\d\.]+)%")
        ylabel_name = 'Accuracy'
        is_percentage = True

    plt.figure(figsize=(10, 6))
    
    for i in range(len(file_paths)):
        file_path = file_paths[i]
        label = label_names[i]

        rounds = []
        values = []
        min_values = []
        max_values = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # main value parsing
                    match_main = pattern_main.search(line)
                    if match_main:
                        round_num = int(match_main.group(1))
                        val = float(match_main.group(2))
                        
                        if is_percentage:
                            val /= 100.0
                        
                        rounds.append(round_num)
                        values.append(val)
                        
                        # Initialize Min/Max with the current value
                        min_values.append(val)
                        max_values.append(val)
                        continue

                    # Min/Max parsing
                    match_range = pattern_range.search(line)
                    if match_range and len(values) > 0:
                        min_val = float(match_range.group(1))
                        max_val = float(match_range.group(2))
                        
                        if is_percentage:
                            min_val /= 100.0
                            max_val /= 100.0
                        
                        min_values[-1] = min_val
                        max_values[-1] = max_val

            if not rounds:
                print(f"[Skip] '{file_path}': Cannot find matching log data.")
                continue

            # draw (plot + vlines + cap)
            line = plt.plot(rounds, values, marker='o', linestyle='-', label=label, zorder=3)
            line_color = line[0].get_color()

            plt.vlines(rounds, min_values, max_values, colors=line_color, 
                       linestyle='-', linewidth=2, alpha=0.6, zorder=2)

            plt.plot(rounds, min_values, marker='_', linestyle='None', color=line_color, 
                     markersize=8, markeredgewidth=2, alpha=0.8, zorder=2)
            plt.plot(rounds, max_values, marker='_', linestyle='None', color=line_color, 
                     markersize=8, markeredgewidth=2, alpha=0.8, zorder=2)

            last_round = rounds[-1]
            last_val = values[-1]
            
            if is_percentage:
                txt = f"  {last_val*100:.1f}%"
            else:
                txt = f"  {last_val:.4f}"
                
            plt.text(last_round, last_val, txt, color=line_color, va='center', fontweight='bold')
            # =================================================================

        except FileNotFoundError:
            print(f"Cannot find file : {file_path}")

    plt.title(f"{title} ({ylabel_name})")
    plt.xlabel('Round')
    plt.ylabel(ylabel_name)
    plt.grid(True)
    plt.legend()
    
    if is_percentage:
        plt.ylim(0, 1.0) 
    else:
        plt.ylim(bottom=0)

    plt.savefig(output_image)
    plt.clf()

def parse_log_and_plot(file_paths, label_names, title, output_image):
    mode = getattr(config, 'DRAW', 'all')  # 'acc' or 'loss' or 'all'
    
    # Determine output filenames
    if isinstance(output_image, str):
        base_name = output_image.rsplit('.', 1)[0] if '.' in output_image else output_image
        loss_output = f"{base_name}_loss.png"
        acc_output = f"{base_name}_acc.png"
    else:
        loss_output = 'loss.png'
        acc_output = 'acc.png'

    if mode == 'all':
        # Generate both loss and acc plots        
        plot_single_metric(file_paths, label_names, title, loss_output, 'loss')
        plot_single_metric(file_paths, label_names, title, acc_output, 'acc')
    elif mode == 'acc':
        # Plot accuracy
        plot_single_metric(file_paths, label_names, title, acc_output, 'acc')
    else:
        # Plot single metric
        plot_single_metric(file_paths, label_names, title, loss_output, 'loss')

log_filenames = [
    'results_jan_week4/exp10_vgg11_default_samesize.log',
    'results_jan_week4/exp10_vgg11_ME_samesize.log',
]
label_names = [
    'HeteroFL',
    'ME',
]

output_image = 'results_jan_week4/exp10_vgg11_samesize.png'
title = 'VGG11 (same size)'
parse_log_and_plot(log_filenames, label_names, title, output_image)