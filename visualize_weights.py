"""Render a trained checkpoint's weights to a PNG.

Shows every convolution kernel, how strongly the first board layer weights each
board cell, every dense weight matrix, per-layer weight distributions, and how
strongly the network weights each metadata feature. Like inference.py, it reads a
private snapshot of the checkpoint, so it is safe to run while the run trains.
"""

import argparse
import math
from pathlib import Path
import pickle

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn

from configuration import REPO_ROOT, load_presets
from game import BLOCK_SIZE
from inference import BOARD_HEIGHT, BOARD_WIDTH, load_model, resolve_checkpoint, snapshot_checkpoint
from state import BOARD_CHANNELS, METADATA_FEATURES

ACTIONS = ('straight', 'right', 'left')
CONV_COLUMNS = 4


def input_layers(model, grid_height, grid_width):
    """Return the first layers that read the board features and the metadata.

    Returns ``(board_layer, board_shape, metadata_layer, metadata_columns)``: the
    board layer's input columns reshape to ``board_shape`` (channels, height,
    width), and ``metadata_columns`` selects the metadata layer's inputs.
    """
    if hasattr(model, 'meta_fc'):
        # Both CNNs max-pool the board twice before flattening it into cnn_fc.
        board_layer = model.cnn_fc[0]
        height, width = grid_height // 4, grid_width // 4
        channels = board_layer.in_features // (height * width)
        return board_layer, (channels, height, width), model.meta_fc[0], slice(None)
    # The MLP reads the flattened board followed by the metadata in one layer.
    first = model.layers[0]
    if not model.has_board:
        return None, None, first, slice(None)
    board_shape = (len(BOARD_CHANNELS), grid_height, grid_width)
    return first, board_shape, first, slice(math.prod(board_shape), None)


def kernel_mosaic(weight):
    """Tile ``(out, in, k, k)`` kernels into one image, with blank gaps between them."""
    out_channels, in_channels, size, _ = weight.shape
    step = size + 1
    tile = torch.full((out_channels * step - 1, in_channels * step - 1), float('nan'))
    for out_channel in range(out_channels):
        for in_channel in range(in_channels):
            tile[out_channel * step:out_channel * step + size,
                 in_channel * step:in_channel * step + size] = weight[out_channel, in_channel]
    return tile


def channel_labels(count, prefix):
    if count == len(BOARD_CHANNELS):
        return list(BOARD_CHANNELS)
    return [f'{prefix}{index}' for index in range(count)]


def plot_convs(subfig, convs):
    cmap = plt.get_cmap('RdBu_r').with_extremes(bad='white')
    rows = math.ceil(len(convs) / CONV_COLUMNS)
    axes = subfig.subplots(rows, CONV_COLUMNS, squeeze=False).flatten()
    for ax, (name, conv) in zip(axes, convs):
        weight = conv.weight.detach().float().cpu()
        out_channels, in_channels, size, _ = weight.shape
        limit = weight.abs().max().item()
        image = ax.imshow(kernel_mosaic(weight), cmap=cmap, vmin=-limit, vmax=limit)
        ax.set_title(f'{name} {tuple(weight.shape)}', fontsize=10)
        centre = size // 2
        for set_ticks, count, prefix in ((ax.set_xticks, in_channels, 'in'),
                                         (ax.set_yticks, out_channels, 'out')):
            # Label only a readable subset when a layer has many channels.
            every = max(1, count // 8)
            set_ticks([index * (size + 1) + centre for index in range(0, count, every)],
                      channel_labels(count, prefix)[::every], fontsize=8)
        if in_channels * 4 <= out_channels:  # a tall, narrow mosaic needs upright input labels
            ax.tick_params(axis='x', labelrotation=90)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        subfig.colorbar(image, ax=ax, shrink=0.7)
    for ax in axes[len(convs):]:
        ax.set_visible(False)
    subfig.suptitle('Convolution kernels (rows: output channel, columns: input channel)')


def plot_board_importance(subfig, layer, shape, pooled_grid):
    weight = layer.weight.detach().float().cpu()[:, :math.prod(shape)]
    # L2 norm of each input column: how strongly the layer reads that board cell.
    importance = weight.norm(dim=0).reshape(shape)
    channels = shape[0]
    if channels == len(BOARD_CHANNELS):
        maps = list(zip(BOARD_CHANNELS, importance))
    else:
        maps = [(f'all {channels} feature channels', importance.norm(dim=0))]
    # A combined map sits in the middle column so it keeps the per-channel maps' size.
    axes = subfig.subplots(1, len(BOARD_CHANNELS), squeeze=False)[0]
    shown = axes if len(maps) == len(axes) else axes[len(axes) // 2:len(axes) // 2 + 1]
    for ax in set(axes) - set(shown):
        ax.set_visible(False)
    for ax, (label, values) in zip(shown, maps):
        image = ax.imshow(values, cmap='viridis')
        ax.set_title(label, fontsize=10)
        subfig.colorbar(image, ax=ax, shrink=0.8)
    pooled = ' (max-pooled grid)' if pooled_grid else ''
    subfig.suptitle(f'Board input importance: L2 norm of each {type(layer).__name__} '
                    f'input column{pooled}')


def plot_dense(subfig, linears, metadata_layer, board_split):
    # Keep narrow matrices wide enough to label and very wide ones from crowding the row.
    widths = [min(max(layer.in_features, 100), 400) for _, layer in linears]
    axes = subfig.subplots(1, len(linears), width_ratios=widths, squeeze=False)[0]
    for ax, (name, layer) in zip(axes, linears):
        weight = layer.weight.detach().float().cpu()
        limit = weight.abs().max().item()
        image = ax.imshow(weight, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto',
                          interpolation='nearest')
        ax.set_title(f'{name} {tuple(weight.shape)}', fontsize=10)
        ax.set_xlabel('input')
        ax.set_ylabel('output')
        if layer is metadata_layer and layer.in_features == len(METADATA_FEATURES):
            ax.set_xticks(range(len(METADATA_FEATURES)), METADATA_FEATURES, rotation=90, fontsize=7)
        if name == 'head.0' and board_split is not None:
            ax.axvline(board_split - 0.5, color='black', linewidth=1)
            ax.set_xticks([board_split / 2, (board_split + layer.in_features) / 2],
                          ['board', 'metadata'])
        if layer.out_features == len(ACTIONS):
            ax.set_yticks(range(len(ACTIONS)), ACTIONS)
        subfig.colorbar(image, ax=ax, shrink=0.8)
    subfig.suptitle('Dense weight matrices')


def plot_summary(subfig, model, metadata_layer, metadata_columns):
    weights = [(name.removesuffix('.weight'), parameter.detach().float().cpu().flatten())
               for name, parameter in model.named_parameters() if name.endswith('weight')]
    distribution_ax, metadata_ax = subfig.subplots(1, 2)
    distribution_ax.boxplot([values for _, values in weights], showfliers=False)
    distribution_ax.set_xticks(range(1, len(weights) + 1), [name for name, _ in weights],
                               rotation=60, fontsize=7)
    distribution_ax.axhline(0, color='grey', linewidth=0.5)
    distribution_ax.set_title('Weight distribution per layer (outliers hidden)')

    importance = metadata_layer.weight.detach().float().cpu()[:, metadata_columns].norm(dim=0)
    names = METADATA_FEATURES if len(importance) == len(METADATA_FEATURES) else [
        f'feature {index}' for index in range(len(importance))]
    metadata_ax.barh(names, importance)
    metadata_ax.invert_yaxis()
    metadata_ax.set_title('Metadata feature importance (L2 norm of input column)')


def plot_weights(model, title, output_path, grid_height, grid_width):
    convs = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
    linears = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    board_layer, board_shape, metadata_layer, metadata_columns = input_layers(model, grid_height, grid_width)
    board_split = model.cnn_fc[0].out_features if hasattr(model, 'cnn_fc') else None

    sections = []
    if convs:
        sections.append((4.2 * math.ceil(len(convs) / CONV_COLUMNS),
                         lambda subfig: plot_convs(subfig, convs)))
    if board_layer is not None:
        sections.append((4.0, lambda subfig: plot_board_importance(
            subfig, board_layer, board_shape, board_shape[1] < grid_height)))
    sections.append((5.0, lambda subfig: plot_dense(subfig, linears, metadata_layer, board_split)))
    sections.append((4.5, lambda subfig: plot_summary(subfig, model, metadata_layer, metadata_columns)))

    heights = [height for height, _ in sections]
    fig = plt.figure(figsize=(18, sum(heights) + 0.8), layout='constrained')
    for subfig, (_, draw) in zip(fig.subfigures(len(sections), 1, height_ratios=heights), sections):
        draw(subfig)
    fig.suptitle(title, fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', help='named preset in models.json (also the default run name)')
    parser.add_argument('--run', help='run directory under runs/ (default: --model)')
    parser.add_argument('--checkpoint', default='best',
                        help="'best', 'latest', or a checkpoint file path (default: best)")
    parser.add_argument('--out', type=Path,
                        help='output PNG path (default: <checkpoint name>_weights.png beside the checkpoint)')
    args = parser.parse_args(argv)

    grid_width, grid_height = BOARD_WIDTH // BLOCK_SIZE, BOARD_HEIGHT // BLOCK_SIZE
    try:
        path, config = resolve_checkpoint(args, load_presets())
        model, config, _, metadata = load_model(
            snapshot_checkpoint(path), config, 'cpu', grid_width, grid_height,
        )
    except (OSError, ValueError, RuntimeError, pickle.UnpicklingError, KeyError, TypeError) as err:
        parser.error(str(err))

    trained = (f'{metadata["n_games"]} games, record {metadata.get("record", "?")}'
               if 'n_games' in metadata else 'no training metadata')
    parameters = sum(parameter.numel() for parameter in model.parameters())
    shown_path = path.resolve().relative_to(REPO_ROOT) if path.resolve().is_relative_to(REPO_ROOT) else path
    output_path = args.out or path.with_name(f'{path.stem}_weights.png')
    plot_weights(model, f'{config["architecture"]}: {shown_path} ({trained}, {parameters:,} parameters)',
                 output_path, grid_height, grid_width)
    print(f'Saved weights plot to {output_path}')


if __name__ == '__main__':
    main()
