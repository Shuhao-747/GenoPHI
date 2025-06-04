#!/usr/bin/env python3
"""
Script to export full workflow modeling results from HPC cluster.
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


def export_workflow_results(source_dir, dest_dir):
    """
    Export full workflow modeling results.
    
    Args:
        source_dir (Path): Source directory containing full_workflow_modeling
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
    
    print("Exporting full workflow modeling results...")
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    print("")
    
    total_files_copied = 0
    
    # Copy model performance files
    print("Processing modeling_results/model_performance/...")
    model_perf_src = source_dir / "modeling_results" / "model_performance"
    model_perf_dest = dest_dir / "modeling_results" / "model_performance"
    
    # Files to copy from model_performance
    performance_files = [
        "model_performance_metrics.csv",
        "pr_curve.png", 
        "roc_curve.png"
    ]
    
    files_copied = 0
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
    
    if files_copied > 0:
        print(f"  ✓ Copied {files_copied} items from model_performance")
        total_files_copied += files_copied
    else:
        print("  ✗ No files found in model_performance")
    
    # Copy filtered_feature_tables directory
    print("Processing feature_selection/filtered_feature_tables/...")
    feature_tables_src = source_dir / "feature_selection" / "filtered_feature_tables"
    feature_tables_dest = dest_dir / "feature_selection" / "filtered_feature_tables"
    if copy_directory_safe(feature_tables_src, feature_tables_dest, "filtered_feature_tables directory"):
        print("  ✓ Copied filtered_feature_tables directory")
        total_files_copied += 1
    
    # Copy additional top-level files from modeling_results
    print("Processing modeling_results top-level files...")
    modeling_results_src = source_dir / "modeling_results"
    modeling_results_dest = dest_dir / "modeling_results"
    
    top_level_files = [
        "select_features_model_performance.csv",
        "select_features_model_predictions.csv"
    ]
    
    files_copied = 0
    for filename in top_level_files:
        src_file = modeling_results_src / filename
        dest_file = modeling_results_dest / filename
        if copy_file_safe(src_file, dest_file, filename):
            files_copied += 1
    
    if files_copied > 0:
        print(f"  ✓ Copied {files_copied} top-level files from modeling_results")
        total_files_copied += files_copied
    
    # Process cutoff directories for importance files
    print("Processing cutoff directories for importance files...")
    cutoff_dirs = sorted([d for d in modeling_results_src.iterdir() 
                         if d.is_dir() and d.name.startswith('cutoff_')])
    
    total_cutoffs = len(cutoff_dirs)
    successful_cutoffs = 0
    importance_files_copied = 0
    
    for cutoff_dir in cutoff_dirs:
        cutoff_name = cutoff_dir.name
        print(f"  Processing {cutoff_name}...")
        
        # Find all run directories in this cutoff
        run_dirs = sorted([d for d in cutoff_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('run_')])
        
        cutoff_files_copied = 0
        for run_dir in run_dirs:
            run_name = run_dir.name
            
            # Create destination structure
            dest_run_dir = dest_dir / "modeling_results" / cutoff_name / run_name
            
            # Copy importance files
            importance_files = ["shap_importances.csv", "feature_importances.csv"]
            
            for filename in importance_files:
                src_file = run_dir / filename
                dest_file = dest_run_dir / filename
                if copy_file_safe(src_file, dest_file, f"{cutoff_name}/{run_name}/{filename}"):
                    cutoff_files_copied += 1
                    importance_files_copied += 1
        
        if cutoff_files_copied > 0:
            successful_cutoffs += 1
            print(f"    ✓ Copied {cutoff_files_copied} importance files from {cutoff_name}")
        else:
            print(f"    ✗ No importance files found in {cutoff_name}")
    
    total_files_copied += importance_files_copied
    
    # Print summary
    print("")
    print("Export complete!")
    print(f"Successfully processed: {successful_cutoffs}/{total_cutoffs} cutoff directories")
    print(f"Total files/directories copied: {total_files_copied}")
    print(f"Importance files copied: {importance_files_copied}")
    
    # Calculate and display total size
    total_size = get_directory_size(dest_dir)
    print(f"Total exported size: {total_size}")
    print(f"Results saved to: {dest_dir}")
    
    # Create manifest file
    create_manifest(dest_dir, source_dir, successful_cutoffs, total_cutoffs, importance_files_copied)


def create_manifest(dest_dir, source_dir, successful_cutoffs, total_cutoffs, importance_files_copied):
    """Create a manifest file documenting the export."""
    manifest_file = dest_dir / "export_manifest.txt"
    
    manifest_content = f"""Full Workflow Modeling Export Manifest
==========================================
Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source Directory: {source_dir}
Destination Directory: {dest_dir}
Cutoff Directories Processed: {successful_cutoffs}/{total_cutoffs}
Importance Files Copied: {importance_files_copied}

Files Exported:

Model Performance:
- modeling_results/model_performance/model_performance_metrics.csv
- modeling_results/model_performance/pr_curve.png
- modeling_results/model_performance/roc_curve.png
- modeling_results/model_performance/predictive_proteins/ (directory)

Feature Selection:
- feature_selection/filtered_feature_tables/ (directory)

Top-level Modeling Results:
- modeling_results/select_features_model_performance.csv
- modeling_results/select_features_model_predictions.csv

Cutoff Run Directories:
- modeling_results/cutoff_*/run_*/shap_importances.csv
- modeling_results/cutoff_*/run_*/feature_importances.csv

Usage:
This export contains the key results and performance metrics from
the full workflow modeling, including model performance data, filtered
feature tables, and importance scores from individual model runs.
Excludes intermediate files to minimize storage requirements while
preserving essential outputs for analysis.
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
        description="Export full workflow modeling results from HPC cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export_workflow_results.py -i ./full_workflow_modeling -o ./export
  python export_workflow_results.py --input /scratch/user/full_workflow_modeling --output ~/results
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input directory containing full_workflow_modeling results'
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
        export_workflow_results(source_path, dest_path)
    else:
        print(f"Would export from: {source_path}")
        print(f"Would export to: {dest_path}")
        print("Run without --dry-run to perform actual export")


if __name__ == "__main__":
    main()