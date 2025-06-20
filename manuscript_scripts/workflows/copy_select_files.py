#!/usr/bin/env python3
"""
Script to export bootstrap validation results from HPC cluster.
Extracts key results and performance metrics while avoiding intermediate files.
"""

import argparse
import shutil
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path


def get_directory_size(path):
    """Calculate total size of directory in human-readable format."""
    try:
        total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        # Convert to human-readable format
        for unit in ['B', 'KB', 'MB', 'GB']:
            if total_size < 1024.0:
                return f"{total_size:.1f} {unit}"
            total_size /= 1024.0
        return f"{total_size:.1f} TB"
    except Exception:
        return "Unknown"


def copy_file_safe(src, dest, file_description=""):
    """Safely copy a file with error handling."""
    try:
        if src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            return True
        else:
            print(f"  ⚠ Missing: {file_description or src.name}")
            return False
    except Exception as e:
        print(f"  ✗ Error copying {src.name}: {e}")
        return False


def copy_directory_safe(src, dest, dir_description=""):
    """Safely copy a directory with error handling."""
    try:
        if src.exists() and src.is_dir():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dest, dirs_exist_ok=True)
            return True
        else:
            print(f"  ⚠ Missing: {dir_description or src.name}")
            return False
    except Exception as e:
        print(f"  ✗ Error copying directory {src.name}: {e}")
        return False


def get_best_cutoff(model_perf_metrics_file):
    """
    Read model performance metrics and return the best cutoff.
    
    Args:
        model_perf_metrics_file (Path): Path to model_performance_metrics.csv
        
    Returns:
        str: Best cutoff identifier (e.g., "cutoff_5") or None if not found
    """
    try:
        if not model_perf_metrics_file.exists():
            return None
            
        df = pd.read_csv(model_perf_metrics_file)
        
        # Try to determine the best cutoff based on available metrics
        if 'MCC' in df.columns:
            # For classification, use MCC (higher is better)
            best_row = df.loc[df['MCC'].idxmax()]
            metric_name = 'MCC'
            metric_value = best_row['MCC']
        elif 'r2' in df.columns:
            # For regression, use R2 (higher is better)
            best_row = df.loc[df['r2'].idxmax()]
            metric_name = 'r2'
            metric_value = best_row['r2']
        else:
            print(f"  ⚠ Warning: No recognized performance metric found in {model_perf_metrics_file}")
            return None
        
        best_cutoff = best_row['cut_off']
        print(f"  ✓ Best cutoff: {best_cutoff} (best {metric_name}: {metric_value:.4f})")
        return best_cutoff
        
    except Exception as e:
        print(f"  ✗ Error reading performance metrics: {e}")
        return None


def export_bootstrap_results(source_dir, dest_dir, minimal=False):
    """
    Export bootstrap validation results.
    
    Args:
        source_dir (Path): Source directory containing bootstrap_validation
        dest_dir (Path): Destination directory for export
        minimal (bool): If True, skip large files and only copy best cutoff
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    
    # Validate source directory exists
    if not source_dir.exists():
        print(f"✗ Error: Source directory '{source_dir}' does not exist")
        sys.exit(1)
    
    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    mode_text = "MINIMAL" if minimal else "FULL"
    print(f"Exporting bootstrap validation results ({mode_text} mode)...")
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    if minimal:
        print("📦 Minimal mode: Skipping predictive_proteins, only copying best cutoff feature tables")
    print("")
    
    # Copy top-level final predictions
    final_predictions = source_dir / "final_predictions.csv"
    if copy_file_safe(final_predictions, dest_dir / "final_predictions.csv", "final_predictions.csv"):
        print("✓ Copied final_predictions.csv")
    
    # Initialize counters
    total_iterations = 0
    successful_iterations = 0
    total_files_copied = 0
    
    # Find all iteration directories
    iteration_dirs = sorted([d for d in source_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('iteration_')])
    
    if not iteration_dirs:
        print("⚠ Warning: No iteration directories found")
        return
    
    # Process each iteration
    for iteration_dir in iteration_dirs:
        iteration_name = iteration_dir.name
        total_iterations += 1
        
        print(f"Processing {iteration_name}...")
        
        # Create iteration directory structure in destination
        dest_iteration = dest_dir / iteration_name
        model_perf_dest = dest_iteration / "modeling_results" / "model_performance"
        predict_dest = dest_iteration / "model_validation" / "predict_results"
        feature_tables_dest = dest_iteration / "feature_selection" / "filtered_feature_tables"
        
        # Track files copied for this iteration
        files_copied = 0
        
        # Define source paths
        model_perf_src = iteration_dir / "modeling_results" / "model_performance"
        predict_src = iteration_dir / "model_validation" / "predict_results"
        feature_tables_src = iteration_dir / "feature_selection" / "filtered_feature_tables"
        
        # Files to copy from model_performance
        performance_files = [
            "model_performance_metrics.csv",
            "pr_curve.png", 
            "roc_curve.png"
        ]
        
        # Copy model performance files
        for filename in performance_files:
            src_file = model_perf_src / filename
            dest_file = model_perf_dest / filename
            if copy_file_safe(src_file, dest_file, filename):
                files_copied += 1
        
        # Copy predictive_proteins directory (only if not minimal)
        if not minimal:
            predictive_proteins_src = model_perf_src / "predictive_proteins"
            predictive_proteins_dest = model_perf_dest / "predictive_proteins"
            if copy_directory_safe(predictive_proteins_src, predictive_proteins_dest, "predictive_proteins directory"):
                files_copied += 1
        else:
            print("  📦 Skipping predictive_proteins directory (minimal mode)")
        
        # Copy filtered_feature_tables (all or just best cutoff)
        if minimal:
            # Only copy the best cutoff feature table
            metrics_file = model_perf_src / "model_performance_metrics.csv"
            best_cutoff = get_best_cutoff(metrics_file)
            
            if best_cutoff:
                # Extract cutoff number from "cutoff_X" format
                if best_cutoff.startswith('cutoff_'):
                    cutoff_num = best_cutoff.split('_')[-1]
                    best_feature_file = f"select_feature_table_cutoff_{cutoff_num}.csv"
                    
                    src_file = feature_tables_src / best_feature_file
                    dest_file = feature_tables_dest / best_feature_file
                    
                    if copy_file_safe(src_file, dest_file, f"best feature table ({best_feature_file})"):
                        files_copied += 1
                else:
                    print(f"  ⚠ Warning: Unexpected cutoff format: {best_cutoff}")
            else:
                print("  ⚠ Warning: Could not determine best cutoff, skipping feature tables")
        else:
            # Copy entire filtered_feature_tables directory
            if copy_directory_safe(feature_tables_src, feature_tables_dest, "filtered_feature_tables directory"):
                files_copied += 1
        
        # Copy strain median predictions
        strain_pred_src = predict_src / "strain_median_predictions.csv"
        strain_pred_dest = predict_dest / "strain_median_predictions.csv"
        if copy_file_safe(strain_pred_src, strain_pred_dest, "strain_median_predictions.csv"):
            files_copied += 1
        
        # Update counters
        if files_copied > 0:
            successful_iterations += 1
            total_files_copied += files_copied
            print(f"  ✓ Copied {files_copied} items from {iteration_name}")
        else:
            print(f"  ✗ No files found in {iteration_name}")
    
    # Print summary
    print("")
    print("Export complete!")
    print(f"Successfully processed: {successful_iterations}/{total_iterations} iterations")
    print(f"Total files/directories copied: {total_files_copied}")
    
    # Calculate and display total size
    total_size = get_directory_size(dest_dir)
    print(f"Total exported size: {total_size}")
    print(f"Results saved to: {dest_dir}")
    
    # Create manifest file
    create_manifest(dest_dir, source_dir, successful_iterations, total_iterations, minimal)


def create_manifest(dest_dir, source_dir, successful_iterations, total_iterations, minimal=False):
    """Create a manifest file documenting the export."""
    manifest_file = dest_dir / "export_manifest.txt"
    
    mode_text = "MINIMAL" if minimal else "FULL"
    
    manifest_content = f"""Bootstrap Validation Export Manifest ({mode_text} Mode)
=====================================
Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source Directory: {source_dir}
Destination Directory: {dest_dir}
Export Mode: {mode_text}
Total Iterations Processed: {successful_iterations}/{total_iterations}

Files Exported per Iteration:
- modeling_results/model_performance/model_performance_metrics.csv
- modeling_results/model_performance/pr_curve.png
- modeling_results/model_performance/roc_curve.png"""

    if not minimal:
        manifest_content += """
- modeling_results/model_performance/predictive_proteins/ (directory)
- feature_selection/filtered_feature_tables/ (directory - all cutoffs)"""
    else:
        manifest_content += """
- feature_selection/filtered_feature_tables/select_feature_table_cutoff_X.csv (best cutoff only)
  (Note: predictive_proteins directory excluded in minimal mode)"""

    manifest_content += """
- model_validation/predict_results/strain_median_predictions.csv

Top-level Files:
- final_predictions.csv

Usage:
This export contains only the key results and performance metrics from
each bootstrap validation iteration, excluding intermediate files to
minimize storage requirements while preserving essential outputs.
"""
    
    if minimal:
        manifest_content += """
MINIMAL MODE NOTES:
- Large predictive_proteins directories were excluded to save space
- Only the best-performing cutoff feature table was copied per iteration
- This reduces storage requirements while maintaining core results
"""
    
    try:
        with open(manifest_file, 'w') as f:
            f.write(manifest_content)
        print(f"Manifest created: {manifest_file}")
    except Exception as e:
        print(f"⚠ Warning: Could not create manifest file: {e}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Export bootstrap validation results from HPC cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export_bootstrap_results.py -i ./bootstrap_validation -o ./export
  python export_bootstrap_results.py --input /scratch/user/bootstrap_validation --output ~/results --minimal
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input directory containing bootstrap_validation results'
    )
    
    parser.add_argument(
        '-o', '--output', 
        type=str,
        required=True,
        help='Output directory for exported results'
    )
    
    parser.add_argument(
        '--minimal',
        action='store_true',
        help='Minimal export: skip predictive_proteins directory and only copy best cutoff feature tables'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be copied without actually copying files'
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        mode_text = "MINIMAL" if args.minimal else "FULL"
        print(f"DRY RUN MODE ({mode_text}) - No files will be copied")
        print("")
    
    # Convert to Path objects and export
    source_path = Path(args.input).resolve()
    dest_path = Path(args.output).resolve()
    
    if not args.dry_run:
        export_bootstrap_results(source_path, dest_path, minimal=args.minimal)
    else:
        print(f"Would export from: {source_path}")
        print(f"Would export to: {dest_path}")
        print(f"Minimal mode: {args.minimal}")
        print("Run without --dry-run to perform actual export")


if __name__ == "__main__":
    main()