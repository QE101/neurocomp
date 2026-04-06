"""Rerun stability battery mechanisms 4 (Extreme Sparsity) and 5 (Sleep Consolidation).
Mechanisms 1-3 already completed before Windows restart.
"""

import sys, os, time
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '.')

# Import the exact functions from the battery script
sys.path.insert(0, os.path.join('.', 'scripts'))
from run_stability_battery import test_extreme_sparsity, test_sleep_consolidation

import torch

def main():
    print('=' * 60, flush=True)
    print('  STABILITY BATTERY: Mechanisms 4 + 5 (rerun)', flush=True)
    print('=' * 60, flush=True)
    print(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

    results = []
    results.append(test_extreme_sparsity())
    results.append(test_sleep_consolidation())

    print(f'\n{"="*60}', flush=True)
    print('  RESULTS (Mechanisms 4 + 5)', flush=True)
    print(f'{"="*60}', flush=True)
    print(f'{"Mechanism":<20} | {"Best":>7} | {"Final":>7} | {"Osc 1st":>7} | {"Osc 2nd":>7} | {"Status":>12}', flush=True)
    print('-' * 75, flush=True)
    print(f'{"Undamped (ref)":<20} | {"2.865x":>7} | {"0.603x":>7} | {"2.33":>7} | {"2.32":>7} | {"NO DAMPING":>12}', flush=True)
    for r in results:
        star = ' **' if r['damping'] == 'DAMPED' else ''
        print(f'{r["name"]:<20} | {r["best"]:.3f}x | {r["final"]:.3f}x | {r["fr"]:.3f} | {r["sr"]:.3f} | {r["damping"]:>12}{star}', flush=True)

    torch.save(results, 'stability_battery_4_5_results.pt')
    print(f'\nSaved to stability_battery_4_5_results.pt', flush=True)
    print(f'Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)


if __name__ == '__main__':
    main()
