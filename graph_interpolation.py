import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import config

pattern_interp = re.compile(
    r"lambda\s*=\s*([\-\d\.]+)\s*\|\s*Test Loss:\s*([\d\.]+)\s*\|\s*Test Accuracy:\s*([\d\.]+)%"
)


def _draw_vlines(ax, vlines):
    """Draw vertical lines on the given axes using positions/labels in vlines."""
    from matplotlib.transforms import blended_transform_factory
    
    for item in vlines:
        if isinstance(item, tuple) and len(item) >= 1:
            x_pos = item[0]
            label = item[1] if len(item) > 1 else None
        else:
            x_pos = item
            label = None
        # Draw vertical line without legend entry
        ax.axvline(x=x_pos, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
        # Add text label below x-axis if provided
        if label:
            trans = blended_transform_factory(ax.transData, ax.transAxes)
            ax.text(x_pos, -0.065, label, ha='center', va='top', fontsize=10, color='black', transform=trans)

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

def _create_animation(all_data_by_file, labels, title, ylabel, output_file, vlines=None):
    """
    Create animation where each frame shows all files' data for a specific round.
    Frame 0: All files' round 0 interpolation curves
    Frame 1: All files' round 1 interpolation curves
    ...
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Use seaborn colorblind palette
    colors = sns.color_palette('colorblind', n_colors=len(all_data_by_file))
    
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
    x_margin = 0  # No margin: fit exactly to lambda min/max
    
    y_min, y_max = min(all_values), max(all_values)
    y_margin = (y_max - y_min) * 0.05  # 5% margin
    
    def animate(frame):
        ax.clear()
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        # Fix y-range and ticks for loss plots only
        if ylabel == 'Test Loss':
            ax.set_ylim(0.5, 2.5)
            ax.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5])
        ax.set_xlabel('Lambda')
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} - Round {frame}")
        ax.grid(True, alpha=0.3)
        
        # Plot each file's data for this round
        for file_idx, rounds_data in enumerate(all_data_by_file):
            if frame < len(rounds_data):
                lambdas, data = rounds_data[frame]
            ax.plot(lambdas, data, linestyle='-', marker=None, label=labels[file_idx], linewidth=3, color=colors[file_idx])
        # Draw vertical guidelines if provided
        if vlines:
            _draw_vlines(ax, vlines)
        
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

def _create_3d_plot(all_data_by_file, labels, title, ylabel, output_file):
    """
    Create 3D plot where each file's rounds are displayed as lines in 3D space.
    x-axis: Communication Round
    y-axis: Lambda
    z-axis: Loss/Accuracy value
    Transparency increases (becomes more opaque) with later rounds.
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Assign one color per file (consistent across all rounds)
    # Use simple colors: green, orange, blue, red, purple, brown, pink, gray
    colors_list = ['green', 'orange', 'blue', 'red', 'purple', 'brown', 'pink', 'gray']
    colors = {file_idx: colors_list[file_idx % len(colors_list)] for file_idx in range(len(all_data_by_file))}
    
    # Plot each file's data
    for file_idx, (rounds_data, label) in enumerate(zip(all_data_by_file, labels)):
        max_rounds = len(rounds_data)
        for round_idx, (lambdas, data) in enumerate(rounds_data):
            # x-axis: round index
            x = [round_idx] * len(lambdas)
            # y-axis: lambda values
            y = lambdas
            # z-axis: loss/accuracy values
            z = data
            
            # Transparency decreases with round progress (early rounds are more opaque)
            alpha = 0.8 - 0.5 * (round_idx / max(1, max_rounds - 1))
            
            # Plot as a 3D line with round-based transparency
            ax.plot(x, y, z, color=colors[file_idx], alpha=alpha, linewidth=3.5, 
                   label=label if round_idx == 0 else '')
    
    ax.set_xlabel('Communication Round', fontsize=11)
    ax.set_ylabel('Lambda', fontsize=11)
    ax.set_zlabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(loc='upper left', fontsize=10)
    ax.view_init(elev=20, azim=45)  # Set viewing angle
    
    # Convert .gif to _3D.pdf
    output_file_3d = output_file.replace('.gif', '_3D.pdf')
    
    plt.savefig(output_file_3d, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"3D plot saved: {output_file_3d}")

def parse_interpolation_and_plot(file_paths, label_names, title, output_image, vline_labels=None):
    mode = config.DRAW  # 'acc', 'loss', 'all'
    use_animation = config.INTERPOLATION_ANIMATION
    use_3d = config.INTERPOLATION_3D

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

    # Prepare vlines from provided labels (expects tuple/list of two strings or None)
    vlines = None
    if vline_labels and len(vline_labels) >= 2:
        vlines = [(0.0, vline_labels[0]), (1.0, vline_labels[1])]

    # 3D mode takes precedence
    if use_3d:
        # Prepare data by round for 3D plotting
        all_losses_by_file = []
        all_accs_by_file = []
        
        for rounds_data in all_rounds_data:
            file_losses = [(lambdas, losses) for lambdas, losses, accs in rounds_data]
            file_accs = [(lambdas, accs) for lambdas, losses, accs in rounds_data]
            all_losses_by_file.append(file_losses)
            all_accs_by_file.append(file_accs)
        
        # Create 3D plots
        if mode in ('loss', 'all'):
            out_loss = output_image if mode == 'loss' else _add_suffix(output_image, 'loss')
            _create_3d_plot(all_losses_by_file, labels, title, 'Test Loss', out_loss)
        
        if mode in ('acc', 'all'):
            out_acc = output_image if mode == 'acc' else _add_suffix(output_image, 'acc')
            _create_3d_plot(all_accs_by_file, labels, title, 'Accuracy', out_acc)
    elif use_animation:
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
            _create_animation(all_losses_by_file, labels, title, 'Test Loss', out_loss, vlines=vlines)
        
        if mode in ('acc', 'all'):
            out_acc = _change_extension(output_image if mode == 'acc' else _add_suffix(output_image, 'acc'), '.gif')
            _create_animation(all_accs_by_file, labels, title, 'Accuracy', out_acc, vlines=vlines)
    else:
        # Create static plots (use first round data for each file)
        # Precompute lambda range from first-round data to set tight x-limits
        first_round_lambdas = []
        for rounds_data in all_rounds_data:
            if rounds_data:
                first_round_lambdas.extend(rounds_data[0][0])
        x_min = min(first_round_lambdas) if first_round_lambdas else None
        x_max = max(first_round_lambdas) if first_round_lambdas else None

        if mode in ('loss', 'all'):
            plt.figure(figsize=(10, 6))
            # Use seaborn colorblind palette
            colors = sns.color_palette('colorblind', n_colors=len(all_rounds_data))
            for idx, (rounds_data, label) in enumerate(zip(all_rounds_data, labels)):
                # Use first round
                if rounds_data:
                    lambdas, losses, _ = rounds_data[0]
                    plt.plot(lambdas, losses, linestyle='-', marker=None, label=label, zorder=3, linewidth=3, color=colors[idx])
            if x_min is not None:
                plt.xlim(x_min, x_max)
                plt.ylim(0.5, 2.5)
                plt.gca().set_yticks([0.5, 1.0, 1.5, 2.0, 2.5])
                if vlines:
                    _draw_vlines(plt.gca(), vlines)
            plt.title(f"{title} (Loss)")
            plt.xlabel('lambda')
            plt.ylabel('Test Loss')
            plt.grid(True)
            plt.legend()
            out_loss = output_image if mode == 'loss' else _add_suffix(output_image, 'loss')
            plt.savefig(out_loss, format='pdf', bbox_inches='tight')
            plt.clf()

        if mode in ('acc', 'all'):
            plt.figure(figsize=(10, 6))
            # Use seaborn colorblind palette
            colors = sns.color_palette('colorblind', n_colors=len(all_rounds_data))
            for idx, (rounds_data, label) in enumerate(zip(all_rounds_data, labels)):
                # Use first round
                if rounds_data:
                    lambdas, _, accs = rounds_data[0]
                    plt.plot(lambdas, accs, linestyle='-', marker=None, label=label, zorder=3, linewidth=3, color=colors[idx])
            if x_min is not None:
                plt.xlim(x_min, x_max)
                if vlines:
                    _draw_vlines(plt.gca(), vlines)
            plt.title(f"{title} (Accuracy)")
            plt.xlabel('lambda')
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.legend()
            out_acc = output_image if mode == 'acc' else _add_suffix(output_image, 'acc')
            plt.savefig(out_acc, format='pdf', bbox_inches='tight')
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
    # log_files = [
    #     'results_interpolate/client_interpolate/smaller_hetero/exp_vgg11_ZC_1-8_interpolation.log',
    #     'results_interpolate/client_interpolate/larger_hetero/exp_vgg11_ZC_1-8_interpolation.log',
    # ]
    # labels = [
    #     'smaller hetero ZC',
    #     'larger hetero ZC',
    # ]
    # output = 'results_interpolate/client_interpolate/larger_hetero/interpolation_vgg11_ZC_1-8_compare.pdf'
    # title = 'VGG11 Interpolation by fixed model size (diff 1-8, ZC)'
    # parse_interpolation_and_plot(log_files, labels, title, output)

    n=1
    while n <= 4:
        log_files = [
            f'results_seed_init/client_interpolate/exp_vgg11_default_1-{n}_interpolation.log',
            f'results_seed_init/client_interpolate/exp_vgg11_ME_1-{n}_interpolation.log',
            f'results_seed_init/client_interpolate/exp_vgg11_MC_1-{n}_interpolation.log',
            f'results_seed_init/client_interpolate/exp_vgg11_ZE_1-{n}_interpolation.log',
            f'results_seed_init/client_interpolate/exp_vgg11_ZC_1-{n}_interpolation.log',
        ]
        labels = [
            'HeteroFL',
            'ME',
            'MC',
            'ZE',
            'ZC',
        ]
        output = f'results_seed_init/client_interpolate/interpolation_vgg11_1-{n}.pdf'
        title = f'VGG11 Interpolation (diff 1/{n})'
        if n == 1:
            parse_interpolation_and_plot(log_files, labels, title, output, vline_labels=('VGG', 'VGG'))
        else:
            parse_interpolation_and_plot(log_files, labels, title, output, vline_labels=('VGG',f'VGG x1/{n}'))

        if n != 1:
            log_files = [
                f'results_seed_init/client_interpolate/exp_vgg11_default_{n}-1_interpolation.log',
                f'results_seed_init/client_interpolate/exp_vgg11_ME_{n}-1_interpolation.log',
                f'results_seed_init/client_interpolate/exp_vgg11_MC_{n}-1_interpolation.log',
                f'results_seed_init/client_interpolate/exp_vgg11_ZE_{n}-1_interpolation.log',
                f'results_seed_init/client_interpolate/exp_vgg11_ZC_{n}-1_interpolation.log',
            ]
            labels = [
                'HeteroFL',
                'ME',
                'MC',
                'ZE',
                'ZC',
            ]
            output = f'results_seed_init/client_interpolate/interpolation_vgg11_{n}-1.pdf'
            title = f'VGG11 Interpolation (diff {n}-1)'
            parse_interpolation_and_plot(log_files, labels, title, output, vline_labels=('VGG',f'VGG x{n}'))
            
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
    #     output = f'results_jan_week4/interpolation_vgg11_1-{n}.pdf'
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
    #     output = f'results_jan_week4/interpolation_vgg11_{type}.pdf'
    #     title = f'VGG11 Interpolation ({type})'
    #     parse_interpolation_and_plot(log_files, labels, title, output)