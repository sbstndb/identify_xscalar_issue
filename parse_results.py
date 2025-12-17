#!/usr/bin/env python3
import re
from collections import defaultdict

# Parse results
results = defaultdict(lambda: defaultdict(dict))
current_build = None

with open('/home/sbstndbs/identify_xscalar_issue/all_results.txt') as f:
    for line in f:
        line = line.strip()
        if line.startswith('=== build_'):
            current_build = line.replace('=== ', '').replace(' ===', '')
        elif line.startswith('"') and current_build:
            # Parse CSV line: "name",iterations,real_time,cpu_time,...
            parts = line.split(',')
            name = parts[0].strip('"')
            cpu_time = float(parts[3])
            # Parse name: xtensor_double_1 or fixed_double_1
            match = re.match(r'(xtensor|fixed)_double_(\d+)', name)
            if match:
                container_type = match.group(1)
                size = int(match.group(2))
                results[current_build][size][container_type] = cpu_time

# Extract compiler info
def parse_build_name(name):
    # build_gcc14_xsimd -> gcc14, xsimd
    name = name.replace('build_', '')
    if '_xsimd' in name:
        compiler = name.replace('_xsimd', '')
        xsimd = 'xsimd'
    else:
        compiler = name.replace('_noxsimd', '')
        xsimd = 'noxsimd'
    return compiler, xsimd

sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64, 128, 256, 512, 1024]

# Print comparison table for each compiler
print("=" * 120)
print("XTENSOR vs XTENSOR_FIXED - Performance Comparison (CPU time in ns, lower is better)")
print("=" * 120)
print()

# Group by compiler
compilers = {}
for build in sorted(results.keys()):
    compiler, xsimd_status = parse_build_name(build)
    if compiler not in compilers:
        compilers[compiler] = {}
    compilers[compiler][xsimd_status] = results[build]

for compiler in sorted(compilers.keys()):
    print(f"\n{'='*100}")
    print(f"COMPILER: {compiler.upper()}")
    print(f"{'='*100}")

    for xsimd_status in ['xsimd', 'noxsimd']:
        if xsimd_status not in compilers[compiler]:
            continue
        data = compilers[compiler][xsimd_status]

        print(f"\n--- {xsimd_status.upper()} ---")
        print(f"{'Size':>6} | {'xtensor':>10} | {'fixed':>10} | {'ratio':>8} | {'Winner':>10}")
        print("-" * 60)

        for size in sizes:
            if size in data:
                xtensor_time = data[size].get('xtensor', 0)
                fixed_time = data[size].get('fixed', 0)
                if xtensor_time > 0 and fixed_time > 0:
                    ratio = xtensor_time / fixed_time
                    winner = "fixed" if ratio > 1 else "xtensor"
                    winner_str = f"{winner} ({ratio:.1f}x)" if ratio > 1 else f"{winner} ({1/ratio:.1f}x)"
                    print(f"{size:>6} | {xtensor_time:>10.2f} | {fixed_time:>10.2f} | {ratio:>8.2f} | {winner_str:>10}")

# Create summary table: best configuration per size
print("\n\n")
print("=" * 120)
print("BEST CONFIGURATIONS PER SIZE (minimum CPU time)")
print("=" * 120)
print()

print(f"{'Size':>6} | {'Best Config':>30} | {'Time (ns)':>10} | {'Worst Config':>30} | {'Time (ns)':>10}")
print("-" * 100)

for size in sizes:
    best_time = float('inf')
    best_config = ""
    worst_time = 0
    worst_config = ""

    for build in results:
        compiler, xsimd_status = parse_build_name(build)
        data = results[build]

        for container_type in ['xtensor', 'fixed']:
            if size in data and container_type in data[size]:
                time = data[size][container_type]
                config = f"{compiler}_{xsimd_status}_{container_type}"
                if time < best_time:
                    best_time = time
                    best_config = config
                if time > worst_time:
                    worst_time = time
                    worst_config = config

    if best_config:
        print(f"{size:>6} | {best_config:>30} | {best_time:>10.2f} | {worst_config:>30} | {worst_time:>10.2f}")

# Summary: fixed vs xtensor wins
print("\n\n")
print("=" * 120)
print("SUMMARY: FIXED vs XTENSOR wins by compiler and xsimd status")
print("=" * 120)

for compiler in sorted(compilers.keys()):
    print(f"\n{compiler.upper()}:")
    for xsimd_status in ['xsimd', 'noxsimd']:
        if xsimd_status not in compilers[compiler]:
            continue
        data = compilers[compiler][xsimd_status]

        fixed_wins = 0
        xtensor_wins = 0
        total_fixed_speedup = 0

        for size in sizes:
            if size in data:
                xtensor_time = data[size].get('xtensor', 0)
                fixed_time = data[size].get('fixed', 0)
                if xtensor_time > 0 and fixed_time > 0:
                    if fixed_time < xtensor_time:
                        fixed_wins += 1
                        total_fixed_speedup += xtensor_time / fixed_time
                    else:
                        xtensor_wins += 1

        avg_speedup = total_fixed_speedup / fixed_wins if fixed_wins > 0 else 0
        print(f"  {xsimd_status}: fixed wins {fixed_wins}/{fixed_wins+xtensor_wins} sizes (avg speedup when fixed wins: {avg_speedup:.2f}x)")
