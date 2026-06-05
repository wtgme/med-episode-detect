"""
Run the antipsychotic episode detection pipeline end-to-end.

Steps
-----
1. generate_sample_data.py  — create data/antipsychotic_prescriptions.csv (skipped if present)
2. episode_pipe.py          — detect medication episodes via PELT change-point detection
3. analyze_episodes.py      — print descriptive statistics
4. visualize_plotly.py      — produce results/viewer.html

Usage
-----
    python run_pipeline.py            # use existing data if present
    python run_pipeline.py --regen    # regenerate sample data from scratch
    python run_pipeline.py --no-viz   # skip the HTML visualisation step
"""

import argparse
import os
import subprocess
import sys


def run(cmd: list) -> None:
    label = ' '.join(cmd)
    print(f"\n{'─' * 60}\n▶  {label}\n{'─' * 60}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description='Run the episode detection pipeline.')
    parser.add_argument('--regen',  action='store_true',
                        help='Regenerate sample data even if it already exists')
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip the HTML visualisation step')
    args = parser.parse_args()

    py = sys.executable

    # 1. Sample data
    if args.regen or not os.path.exists('data/antipsychotic_prescriptions.csv'):
        run([py, 'generate_sample_data.py'])
    else:
        print('data/antipsychotic_prescriptions.csv already exists — skipping generation '
              '(pass --regen to force)')

    # 2. Episode detection
    run([py, 'episode_pipe.py'])

    # 3. Descriptive analysis
    run([py, 'analyze_episodes.py'])

    # 4. Interactive visualisation
    if not args.no_viz:
        run([py, 'visualize_plotly.py', '--n', '20'])
        print('\nOpen results/viewer.html in any browser to explore the timeline.')

    print('\nPipeline complete.')


if __name__ == '__main__':
    main()
