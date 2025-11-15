#!/usr/bin/env python3
"""
Wrapper script to run prediction_only_script.py across all bootstrap directories.
"""

import os
import glob
import subprocess
import sys

def main():
    # Base directory
    base_dir = "/global/scratch/users/anoonan/BRaVE"
    
    # Find all bootstrap directories
    pattern = os.path.join(base_dir, "*/phage_bootstrapping/*_modeling_phage_*")
    directories = glob.glob(pattern)
    directories = [d for d in directories if os.path.isdir(d)]
    directories.sort()
    
    if not directories:
        print("No directories found matching pattern!")
        return 1
    
    print(f"Found {len(directories)} directories:")
    for i, directory in enumerate(directories, 1):
        rel_path = os.path.relpath(directory, base_dir)
        print(f"  {i:2d}. {rel_path}")
    print()
    
    # SLURM parameters
    slurm_params = {
        'account': 'ac_mak',
        'partition': 'lr7', 
        'qos': 'lr_normal',
        'environment': 'phage_modeling',
        'mem_per_job': '32',
        'time_limit': '2:00:00',
        'threads': '8',
        'n_iterations': '20'
    }
    
    print("SLURM parameters:")
    for key, value in slurm_params.items():
        print(f"  {key}: {value}")
    print()
    
    # Process each directory
    success_count = 0
    failed_count = 0
    
    for i, directory in enumerate(directories, 1):
        rel_path = os.path.relpath(directory, base_dir)
        print(f"[{i}/{len(directories)}] Processing: {rel_path}")
        
        # Build command
        cmd = [
            'python3', 'run_prediction_only.py',
            '--output_dir', directory,
            '--n_iterations', slurm_params['n_iterations'],
            '--account', slurm_params['account'],
            '--partition', slurm_params['partition'],
            '--qos', slurm_params['qos'],
            '--environment', slurm_params['environment'],
            '--mem_per_job', slurm_params['mem_per_job'],
            '--time_limit', slurm_params['time_limit'],
            '--threads', slurm_params['threads']
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  ✓ Success")
            success_count += 1
            
            # Print any job IDs from the output
            for line in result.stdout.split('\n'):
                if 'Job array submitted:' in line:
                    print(f"    {line}")
                    
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed: {e}")
            if e.stderr:
                print(f"    Error: {e.stderr}")
            failed_count += 1
        
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total directories: {len(directories)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    
    if success_count > 0:
        print(f"\nMonitor jobs with:")
        print(f"  squeue -u $USER")
        print(f"  squeue -u $USER --name=prediction_only")
    
    return failed_count

if __name__ == "__main__":
    sys.exit(main())