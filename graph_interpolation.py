import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import config

pattern_interp = re.compile(
    r"lambda\s*=\s*([\-\d\.]+)\s*\|\s*Test Loss:\s*([\d\.]+)\s*\|\s*Test Accuracy:\s*([\d\.]+)%"
)

def _parse_single_file(file_path):
    """Parse all rounds of interpolation data from a single file."""
    rounds_data = []  # List of (lambdas, losses, accs) for each round
    current_lambdas = []
    current_losses = []
    current_accs = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Check if new interpolation experiment starts
                if "Starting Interpolation Experiment" in line or "Interpolation Experiment Complete" in line:
                    # Save previous round data if exists
                    if current_lambdas:
                        rounds_data.append((current_lambdas[:], current_losses[:], current_accs[:]))
                        current_lambdas = []
                        current_losses = []
                        current_accs = []
                    continue
                
                m = pattern_interp.search(line)
                if m:
                    lam = float(m.group(1))
                    loss = float(m.group(2))
                    acc = float(m.group(3)) / 100.0
                    current_lambdas.append(lam)
                    current_losses.append(loss)
                    current_accs.append(acc)
        
        # Save last round if exists
        if current_lambdas:
            rounds_data.append((current_lambdas, current_losses, current_accs))
            
    except FileNotFoundError:
        print(f"Cannot find file: {file_path}")
    
    return rounds_data

def _create_animation(all_data_by_file, labels, title, ylabel, output_file):
    """
    Create animation where each frame shows all files' data for a specific round.
    Frame 0: All files' round 0 interpolation curves
    Frame 1: All files' round 1 interpolation curves
    ...
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Determine max number of rounds across all files
    max_rounds = max(len(rounds) for rounds in all_data_by_file)
    
    # Calculate global min/max for fixed axes across all rounds and files
    all_lambdas = []
    all_values = []
    for rounds_data in all_data_by_file:
        for lambdas, data in rounds_data:
            all_lambdas.extend(lambdas)
            all_values.extend(data)
    
    x_min, x_max = min(all_lambdas), max(all_lambdas)
    x_margin = (x_max - x_min) * 0.05  # 5% margin
    
    y_min, y_max = min(all_values), max(all_values)
    y_margin = (y_max - y_min) * 0.05  # 5% margin
    
    def animate(frame):
        ax.clear()
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_xlabel('Lambda')
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} - Round {frame}")
        ax.grid(True, alpha=0.3)
        
        # Plot each file's data for this round
        for file_idx, rounds_data in enumerate(all_data_by_file):
            if frame < len(rounds_data):
                lambdas, data = rounds_data[frame]
                ax.plot(lambdas, data, marker='o', label=labels[file_idx], linewidth=2, markersize=6)
        
        ax.legend(loc='best')
    
    # Linger on the last frame by repeating it (pillow writer ignores repeat_delay)
    linger_frames = 3  # show last frame ~3x longer
    frame_sequence = list(range(max_rounds)) + [max_rounds - 1] * (linger_frames - 1)

    anim = animation.FuncAnimation(
        fig, animate, frames=frame_sequence, interval=1000, repeat=True
    )
    anim.save(output_file, writer='pillow', fps=1)
    plt.close()
    print(f"Animation saved: {output_file}")

def parse_interpolation_and_plot(file_paths, label_names, title, output_image):
    mode = config.DRAW  # 'acc', 'loss', 'all'
    use_animation = config.INTERPOLATION_ANIMATION

    # all_data_by_file[file_idx][round_idx] = (lambdas, losses, accs)
    all_rounds_data = []
    labels = []

    for i, file_path in enumerate(file_paths):
        rounds_data = _parse_single_file(file_path)
        if not rounds_data:
            print(f"[Skip] '{file_path}': Cannot find interpolation data.")
            continue
        all_rounds_data.append(rounds_data)
        labels.append(label_names[i])
    
    if not all_rounds_data:
        print("No data to plot.")
        return

    if use_animation:
        # Prepare data by round for animation
        # all_losses_by_file[file_idx][round_idx] = (lambdas, losses)
        all_losses_by_file = []
        all_accs_by_file = []
        
        for rounds_data in all_rounds_data:
            file_losses = [(lambdas, losses) for lambdas, losses, accs in rounds_data]
            file_accs = [(lambdas, accs) for lambdas, losses, accs in rounds_data]
            all_losses_by_file.append(file_losses)
            all_accs_by_file.append(file_accs)
        
        # Create animations
        if mode in ('loss', 'all'):
            out_loss = _change_extension(output_image if mode == 'loss' else _add_suffix(output_image, 'loss'), '.gif')
            _create_animation(all_losses_by_file, labels, title, 'Test Loss', out_loss)
        
        if mode in ('acc', 'all'):
            out_acc = _change_extension(output_image if mode == 'acc' else _add_suffix(output_image, 'acc'), '.gif')
            _create_animation(all_accs_by_file, labels, title, 'Accuracy', out_acc)
    else:
        # Create static plots (use first round data for each file)
        if mode in ('loss', 'all'):
            plt.figure(figsize=(10, 6))
            for rounds_data, label in zip(all_rounds_data, labels):
                # Use first round
                if rounds_data:
                    lambdas, losses, _ = rounds_data[0]
                    plt.plot(lambdas, losses, marker='o', linestyle='-', label=label, zorder=3)
            plt.title(f"{title} (Loss)")
            plt.xlabel('lambda')
            plt.ylabel('Test Loss')
            plt.grid(True)
            plt.legend()
            plt.ylim(bottom=0)
            out_loss = output_image if mode == 'loss' else _add_suffix(output_image, 'loss')
            plt.savefig(out_loss)
            plt.clf()

        if mode in ('acc', 'all'):
            plt.figure(figsize=(10, 6))
            for rounds_data, label in zip(all_rounds_data, labels):
                # Use first round
                if rounds_data:
                    lambdas, _, accs = rounds_data[0]
                    plt.plot(lambdas, accs, marker='o', linestyle='-', label=label, zorder=3)
            plt.title(f"{title} (Accuracy)")
            plt.xlabel('lambda')
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.legend()
            plt.ylim(0, 1.0)
            out_acc = output_image if mode == 'acc' else _add_suffix(output_image, 'acc')
            plt.savefig(out_acc)
            plt.clf()

def _add_suffix(filename, suffix):
    if '.' in filename:
        base, ext = filename.rsplit('.', 1)
        return f"{base}_{suffix}.{ext}"
    return f"{filename}_{suffix}"

def _change_extension(filename, new_ext):
    if '.' in filename:
        base, _ = filename.rsplit('.', 1)
        return f"{base}{new_ext}"
    return f"{filename}{new_ext}"

if __name__ == '__main__':
    n=1
    while n <= 32:
        log_files = [
            f'results_interpolate/exp_vgg11_default_1-{n}_interpolation.log',
            f'results_interpolate/exp_vgg11_ME_1-{n}_interpolation.log',
            f'results_interpolate/exp_vgg11_MC_1-{n}_interpolation.log',
            f'results_interpolate/exp_vgg11_ZE_1-{n}_interpolation.log',
            f'results_interpolate/exp_vgg11_ZC_1-{n}_interpolation.log',
        ]
        labels = [
            'HeteroFL',
            'ME',
            'MC',
            'ZE',
            'ZC',
        ]
        output = f'results_interpolate/interpolation_vgg11_1-{n}.png'
        title = f'VGG11 Interpolation (diff 1/{n})'
        parse_interpolation_and_plot(log_files, labels, title, output)
        n *= 2

    # n=1
    # while n <= 32:
    #     log_files = [
    #         f'results_jan_week4/exp11_vgg11_default_1-{n}_interpolation.log',
    #         f'results_jan_week4/exp11_vgg11_ME_1-{n}_interpolation.log',
    #         f'results_jan_week4/exp11_vgg11_MC_1-{n}_interpolation.log',
    #         f'results_jan_week4/exp11_vgg11_ZE_1-{n}_interpolation.log',
    #         f'results_jan_week4/exp11_vgg11_ZC_1-{n}_interpolation.log',
    #     ]
    #     labels = [
    #         'HeteroFL',
    #         'ME',
    #         'MC',
    #         'ZE',
    #         'ZC',
    #     ]
    #     output = f'results_jan_week4/interpolation_vgg11_1-{n}.png'
    #     title = f'VGG11 Interpolation (diff 1/{n})'
    #     parse_interpolation_and_plot(log_files, labels, title, output)
    #     n *= 2

    # for type in ['default', 'ME', 'MC', 'ZE', 'ZC']:
    #     log_files = [
    #         f'results_jan_week4/exp11_vgg11_{type}_1-1_interpolation.log',
    #         f'results_jan_week4/exp11_vgg11_{type}_1-2_interpolation.log',
    #         f'results_jan_week4/exp11_vgg11_{type}_1-4_interpolation.log',
    #         f'results_jan_week4/exp11_vgg11_{type}_1-8_interpolation.log',
    #         f'results_jan_week4/exp11_vgg11_{type}_1-16_interpolation.log',
    #         f'results_jan_week4/exp11_vgg11_{type}_1-32_interpolation.log',
    #     ]
    #     labels = [
    #         'diff 1-1',
    #         'diff 1-2',
    #         'diff 1-4',
    #         'diff 1-8',
    #         'diff 1-16',
    #         'diff 1-32',
    #     ]
    #     output = f'results_jan_week4/interpolation_vgg11_{type}.png'
    #     title = f'VGG11 Interpolation ({type})'
    #     parse_interpolation_and_plot(log_files, labels, title, output)