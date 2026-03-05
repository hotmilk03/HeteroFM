import re
import numpy as np
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
    
    # Determine min number of rounds across all files (only plot up to the common minimum)
    min_rounds = min(len(rounds) for rounds in all_data_by_file)
    
    # Calculate lambda range for fixed x-axis
    all_lambdas = []
    for rounds_data in all_data_by_file:
        for lambdas, data in rounds_data:
            all_lambdas.extend(lambdas)
    
    x_min, x_max = min(all_lambdas), max(all_lambdas)
    x_margin = 0  # No margin: fit exactly to lambda min/max
    
    def animate(frame):
        ax.clear()
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        # Fixed y-range: 0-3 for loss, 0-1 for accuracy
        ax.set_ylim(0, 1 if ylabel == 'Accuracy' else 3)
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
    frame_sequence = list(range(min_rounds)) + [min_rounds - 1] * (linger_frames - 1)

    anim = animation.FuncAnimation(
        fig, animate, frames=frame_sequence, interval=1000, repeat=True
    )
    anim.save(output_file, writer='pillow', fps=1)
    plt.close()
    print(f"Animation saved: {output_file}")

def _create_3d_plot(all_data_by_file, labels, title, ylabel, output_file):
    """
    Create 3D surface plot where each file's rounds are displayed as surfaces.
    x-axis: Lambda
    y-axis: Communication Round
    z-axis: Loss/Accuracy value
    Uses high transparency so overlapping surfaces are visible.
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get a vibrant color palette for different files (non-colorblind, bright and distinct)
    colors_list = ['#FF0000', '#00FF00', '#0000FF', '#FFD700', '#FF00FF', '#00FFFF', '#FF6600', '#00FF99']
    
    # Plot each file's data as a surface
    for file_idx, (rounds_data, label) in enumerate(zip(all_data_by_file, labels)):
        # Convert rounds_data into mesh grid format
        # rounds_data[round_idx] = (lambdas, data)
        # Create X (lambda), Y (round), Z (loss/acc) arrays
        
        max_rounds = len(rounds_data)
        
        # Build X, Y, Z for this file
        X_data = []
        Y_data = []
        Z_data = []
        
        for round_idx, (lambdas, data) in enumerate(rounds_data):
            X_data.append(np.array(lambdas))
            Y_data.append(np.array([round_idx] * len(lambdas)))
            Z_data.append(np.array(data))
        
        # Create meshgrid
        # Find unique lambdas and rounds for consistent grid
        all_lambdas = np.unique(np.concatenate(X_data))
        all_rounds = np.arange(max_rounds)
        
        X_mesh, Y_mesh = np.meshgrid(all_lambdas, all_rounds)
        Z_mesh = np.zeros_like(X_mesh, dtype=float)
        
        # Fill Z_mesh with interpolated values
        for round_idx, (lambdas, data) in enumerate(rounds_data):
            # Use linear interpolation for missing lambda values
            Z_interp = np.interp(all_lambdas, lambdas, data)
            Z_mesh[round_idx, :] = Z_interp
        
        # Plot surface with higher transparency (lower alpha = more transparent) and assigned color
        color = colors_list[file_idx % len(colors_list)]
        ax.plot_surface(X_mesh, Y_mesh, Z_mesh, alpha=0.15, label=label, 
                       shade=False, edgecolor='none', color=color)
    
    ax.set_xlabel('Lambda', fontsize=11)
    ax.set_ylabel('Communication Round', fontsize=11)
    ax.set_zlabel(ylabel, fontsize=11)
    ax.set_zlim(0, 1 if ylabel == 'Accuracy' else 3)  # Fixed z-range: 0-1 for acc, 0-3 for loss
    ax.set_title(title, fontsize=13)
    ax.view_init(elev=20, azim=45)  # Set viewing angle
    
    # Add legend (manual since plot_surface doesn't support it directly)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_list[i % len(colors_list)], 
                            label=labels[i]) for i in range(len(labels))]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Convert .gif to _3D.pdf
    output_file_3d = output_file.replace('.pdf', '_3D.pdf')
    
    plt.savefig(output_file_3d, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"3D surface plot saved: {output_file_3d}")

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
            first_round_losses = []
            for idx, (rounds_data, label) in enumerate(zip(all_rounds_data, labels)):
                # Use first round
                if rounds_data:
                    lambdas, losses, _ = rounds_data[0]
                    first_round_losses.extend(losses)
                    plt.plot(lambdas, losses, linestyle='-', marker=None, label=label, zorder=3, linewidth=3, color=colors[idx])
            if x_min is not None:
                plt.xlim(x_min, x_max)
                plt.ylim(0, 3)  # Fixed y-range: 0-3 for loss
                if vlines:
                    _draw_vlines(plt.gca(), vlines)
            plt.title(f"{title} (Loss)")
            plt.xlabel('lambda')
            plt.ylabel('Test Loss')
            plt.grid(True)
            plt.legend()
            out_loss = output_image if mode == 'loss' else _add_suffix(output_image, 'loss')
            plt.savefig(out_loss, bbox_inches='tight')
            plt.clf()

        if mode in ('acc', 'all'):
            plt.figure(figsize=(10, 6))
            # Use seaborn colorblind palette
            colors = sns.color_palette('colorblind', n_colors=len(all_rounds_data))
            first_round_accs = []
            for idx, (rounds_data, label) in enumerate(zip(all_rounds_data, labels)):
                # Use first round
                if rounds_data:
                    lambdas, _, accs = rounds_data[0]
                    first_round_accs.extend(accs)
                    plt.plot(lambdas, accs, linestyle='-', marker=None, label=label, zorder=3, linewidth=3, color=colors[idx])
            if x_min is not None:
                plt.xlim(x_min, x_max)
                plt.ylim(0, 1)  # Fixed y-range: 0-1 for accuracy
                if vlines:
                    _draw_vlines(plt.gca(), vlines)
            plt.title(f"{title} (Accuracy)")
            plt.xlabel('lambda')
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.legend()
            out_acc = output_image if mode == 'acc' else _add_suffix(output_image, 'acc')
            plt.savefig(out_acc, bbox_inches='tight')
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
    log_files = [
        # 'results_resnet/client_interpolate/classRatio0.01_diff/ln/exp_resnet50_0.01_default_2-1_interpolation_localEpoch40-ln.log',
        # 'results_resnet/client_interpolate/classRatio0.01_diff/ln/exp_resnet50_0.01_ME_2-1_interpolation_localEpoch40-ln.log',
        # 'results_resnet/client_interpolate/classRatio0.01_diff/ln/exp_resnet50_0.01_MC_2-1_interpolation_localEpoch40-ln.log',
        # 'results_resnet/client_interpolate/classRatio0.01_diff/ln/exp_resnet50_0.01_ZE_2-1_interpolation_localEpoch40-ln.log',
        # 'results_resnet/client_interpolate/classRatio0.01_diff/ln/exp_resnet50_0.01_ZC_2-1_interpolation_localEpoch40-ln.log',

        # 'results_resnet/client_interpolate/classRatio0.01_similar/ln/exp_resnet50_0.01_default_4-5_interpolation_localEpoch40-ln.log',
        # 'results_resnet/client_interpolate/classRatio0.01_similar/ln/exp_resnet50_0.01_ME_4-5_interpolation_localEpoch40-ln.log'
        
        # 'results_resnet/permute_interpolate/classRatio0.01_diffGlobalSize_by_theta/ln/exp_resnet50_0.01_MC_ZE_0.0_1.0_2-1_permute_interpolation-ln.log',
        # 'results_resnet/permute_interpolate/classRatio0.01_diffGlobalSize_by_theta/ln/exp_resnet50_0.01_MC_ZE_0.1_0.9_2-1_permute_interpolation-ln.log',
        # 'results_resnet/permute_interpolate/classRatio0.01_diffGlobalSize_by_theta/ln/exp_resnet50_0.01_MC_ZE_0.2_0.8_2-1_permute_interpolation-ln.log',
        # 'results_resnet/permute_interpolate/classRatio0.01_diffGlobalSize_by_theta/ln/exp_resnet50_0.01_MC_ZE_0.3_0.7_2-1_permute_interpolation-ln.log',
        # 'results_resnet/permute_interpolate/classRatio0.01_diffGlobalSize_by_theta/ln/exp_resnet50_0.01_MC_ZE_0.4_0.6_2-1_permute_interpolation-ln.log',
        # 'results_resnet/permute_interpolate/classRatio0.01_diffGlobalSize_by_theta/ln/exp_resnet50_0.01_MC_ZE_0.5_0.5_2-1_permute_interpolation-ln.log',
        # 'results_resnet/permute_interpolate/classRatio0.01_diffGlobalSize_by_theta/ln/exp_resnet50_0.01_MC_ZE_0.6_0.4_2-1_permute_interpolation-ln.log',
        # 'results_resnet/permute_interpolate/classRatio0.01_diffGlobalSize_by_theta/ln/exp_resnet50_0.01_MC_ZE_0.7_0.3_2-1_permute_interpolation-ln.log',
        # 'results_resnet/permute_interpolate/classRatio0.01_diffGlobalSize_by_theta/ln/exp_resnet50_0.01_MC_ZE_0.8_0.2_2-1_permute_interpolation-ln.log',
        # 'results_resnet/permute_interpolate/classRatio0.01_diffGlobalSize_by_theta/ln/exp_resnet50_0.01_MC_ZE_0.9_0.1_2-1_permute_interpolation-ln.log',
        # 'results_resnet/permute_interpolate/classRatio0.01_diffGlobalSize_by_theta/ln/exp_resnet50_0.01_MC_ZE_1.0_0.0_2-1_permute_interpolation-ln.log',
    ]
    labels = [
        # 'default(gn, noPerm)',
        # 'gn',
        # 'bn',
        # 'ln',
        # 'in'
        
        # 'HeteroFL',
        # 'ME',
        # 'MC',
        # 'ZE',
        # 'ZC',

        # 'MC:ZE=0:10',
        # 'MC:ZE=1:9',
        # 'MC:ZE=2:8',
        # 'MC:ZE=3:7',
        # 'MC:ZE=4:6',
        # 'MC:ZE=5:5',
        # 'MC:ZE=6:4',
        # 'MC:ZE=7:3',
        # 'MC:ZE=8:2',
        # 'MC:ZE=9:1',
        # 'MC:ZE=10:0',
    ]
    
    output = f'results_resnet/permute_interpolate/classRatio0.01_diffGlobalSize_by_theta/ln/lr0.01/permute_interpolation_resnet50_0.01_2-1_MC_ZE_localEpoch40_interp_lr0.01-ln.gif'
    title = 'ResNet50 Permute Interpolate (norm LN, diff 2-1, globalSizeMode interp, localEpoch=40, 1% dataset, lr 0.01)'

    parse_interpolation_and_plot(log_files, labels, title, output)