#!/usr/bin/env python3
"""
Script to export bootstrap validation results from HPC cluster.
Extracts key results and performance metrics while avoiding intermediate files.
"""

import argparse
import shutil
import sys
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


def export_bootstrap_results(source_dir, dest_dir):
    """
    Export bootstrap validation results.
    
    Args:
        source_dir (Path): Source directory containing bootstrap_validation
        dest_dir (Path): Destination directory for export
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    
    # Validate source directory exists
    if not source_dir.exists():
        print(f"✗ Error: Source directory '{source_dir}' does not exist")
        sys.exit(1)
    
    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print("Exporting bootstrap validation results...")
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
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
        
        # Copy predictive_proteins directory
        predictive_proteins_src = model_perf_src / "predictive_proteins"
        predictive_proteins_dest = model_perf_dest / "predictive_proteins"
        if copy_directory_safe(predictive_proteins_src, predictive_proteins_dest, "predictive_proteins directory"):
            files_copied += 1
        
        # Copy filtered_feature_tables directory
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
    create_manifest(dest_dir, source_dir, successful_iterations, total_iterations)


def create_manifest(dest_dir, source_dir, successful_iterations, total_iterations):
    """Create a manifest file documenting the export."""
    manifest_file = dest_dir / "export_manifest.txt"
    
    manifest_content = f"""Bootstrap Validation Export Manifest
=====================================
Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source Directory: {source_dir}
Destination Directory: {dest_dir}
Total Iterations Processed: {successful_iterations}/{total_iterations}

Files Exported per Iteration:
- modeling_results/model_performance/model_performance_metrics.csv
- modeling_results/model_performance/pr_curve.png
- modeling_results/model_performance/roc_curve.png
- modeling_results/model_performance/predictive_proteins/ (directory)
- model_validation/predict_results/strain_median_predictions.csv
- feature_selection/filtered_feature_tables/ (directory)

Top-level Files:
- final_predictions.csv

Usage:
This export contains only the key results and performance metrics from
each bootstrap validation iteration, excluding intermediate files to
minimize storage requirements while preserving essential outputs.
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
  python export_bootstrap_results.py --input /scratch/user/bootstrap_validation --output ~/results
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
        '--dry-run',
        action='store_true',
        help='Show what would be copied without actually copying files'
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be copied")
        print("")
    
    # Convert to Path objects and export
    source_path = Path(args.input).resolve()
    dest_path = Path(args.output).resolve()
    
    if not args.dry_run:
        export_bootstrap_results(source_path, dest_path)
    else:
        print(f"Would export from: {source_path}")
        print(f"Would export to: {dest_path}")
        print("Run without --dry-run to perform actual export")


if __name__ == "__main__":
    main()