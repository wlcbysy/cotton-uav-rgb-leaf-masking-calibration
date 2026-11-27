#!/usr/bin/env python3
"""Run selected analyses and figures from the distributed processed inputs."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '02_Code/src/upgrade_q1'))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['fig3', 'fig4', 'fig5', 'fig6', 'figs1', 'figs2', 'ablation-2x2'])
    task = parser.parse_args().task
    if task == 'ablation-2x2':
        from run_peer_review_ablations import run_r8_2x2_index_mask_ablation
        run_r8_2x2_index_mask_ablation()
    else:
        import plot_q1_upgraded_figures as plots
        plots.fs.apply()
        getattr(plots, 'render_figure_' + task[3:] if task.startswith('figs') else 'render_figure' + task[-1])()

if __name__ == '__main__':
    main()
