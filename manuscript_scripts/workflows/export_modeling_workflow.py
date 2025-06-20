#!/usr/bin/env python3
"""
Script to export full workflow modeling results from HPC cluster.
Extracts key results and performance metrics while avoiding intermediate files.
Handles both main workflow and kmer_modeling workflow, selecting best cutoffs.
Includes phage/strain feature files, cluster files, and kmer_modeling feature tables.
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required but not installed. Please install with: pip install pandas")
    sys.exit(1)


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


def validate_workflow_structure(source_dir, workflow_name):
    """
    Validate that the expected workflow structure exists.
    
    Args:
        source_dir (Path): Source directory to validate
        workflow_name (str): Name of workflow for logging
        
    Returns:
        bool: True if structure is valid, False otherwise
    """
    print(f"Validating {workflow_name} workflow structure...")
    
    if workflow_name == "kmer_modeling":
        required_paths = [
            source_dir / "modeling" / "modeling_results",
            source_dir / "modeling" / "feature_selection"
        ]
        optional_paths = [
            source_dir / "feature_tables",
            source_dir / "full_feature_table.csv"
        ]
    else:  # main workflow
        required_paths = [
            source_dir / "modeling_results",
            source_dir / "feature_selection"
        ]
        optional_paths = [
            source_dir / "phage" / "features",
            source_dir / "strain" / "features"
        ]
    
    # Check required paths
    missing_required = []
    for path in required_paths:
        if not path.exists():
            missing_required.append(str(path))
    
    if missing_required:
        print(f"  ✗ Missing required directories for {workflow_name}:")
        for path in missing_required:
            print(f"    - {path}")
        return False
    
    # Check optional paths (just warn)
    missing_optional = []
    for path in optional_paths:
        if not path.exists():
            missing_optional.append(str(path))
    
    if missing_optional:
        print(f"  ⚠ Optional directories not found (will be skipped):")
        for path in missing_optional:
            print(f"    - {path}")
    
    print(f"  ✓ {workflow_name} workflow structure validated")
    return True


def get_best_cutoff(metrics_file_path):
    """
    Read model performance metrics and return the best cutoff.
    
    Args:
        metrics_file_path (Path): Path to model_performance_metrics.csv
        
    Returns:
        str: Best cutoff value (e.g., "15" from "cutoff_15")
    """
    try:
        if not metrics_file_path.exists():
            print(f"  ⚠ Model performance metrics not found: {metrics_file_path}")
            return None
            
        performance_df = pd.read_csv(metrics_file_path)
        if performance_df.empty:
            print(f"  ⚠ Empty performance metrics file: {metrics_file_path}")
            return None
        
        # Check if required column exists
        if 'cut_off' not in performance_df.columns:
            print(f"  ⚠ 'cut_off' column not found in {metrics_file_path}")
            print(f"      Available columns: {list(performance_df.columns)}")
            return None
            
        # Get the best performing cutoff (first row after sorting)
        cut_off_value = performance_df.iloc[0]['cut_off']
        if pd.isna(cut_off_value):
            print(f"  ⚠ No valid cutoff value in first row of {metrics_file_path}")
            return None
            
        # Extract cutoff number from format like "cutoff_15"
        if isinstance(cut_off_value, str) and '_' in cut_off_value:
            top_cutoff = cut_off_value.split('_')[-1]
        else:
            # Handle case where it might just be the number
            top_cutoff = str(cut_off_value)
            
        print(f"  ✓ Best cutoff identified: {top_cutoff}")
        return top_cutoff
        
    except Exception as e:
        print(f"  ✗ Error reading performance metrics: {e}")
        return None


def export_workflow_results(source_dir, dest_dir, workflow_name="main", relative_path="", dry_run=False):
    """
    Export workflow modeling results for a specific workflow.
    
    Args:
        source_dir (Path): Source directory containing workflow results
        dest_dir (Path): Destination directory for export
        workflow_name (str): Name of the workflow (for logging)
        relative_path (str): Relative path for destination structure
        dry_run (bool): If True, only show what would be copied
        
    Returns:
        dict: Summary of export results
    """
    print(f"\n{'='*60}")
    print(f"Processing {workflow_name} workflow...")
    print(f"Source: {source_dir}")
    print(f"{'='*60}")
    
    total_files_copied = 0
    export_summary = {
        'workflow': workflow_name,
        'files_copied': 0,
        'best_cutoff': None,
        'success': False
    }
    
    # Validate workflow structure first
    if not validate_workflow_structure(source_dir, workflow_name):
        print(f"✗ Invalid workflow structure for {workflow_name}")
        return export_summary
    
    # Determine paths based on workflow structure
    if workflow_name == "kmer_modeling":
        modeling_results_src = source_dir / "modeling" / "modeling_results"
        feature_selection_src = source_dir / "modeling" / "feature_selection"
        dest_base = dest_dir / relative_path if relative_path else dest_dir
    else:
        modeling_results_src = source_dir / "modeling_results"
        feature_selection_src = source_dir / "feature_selection"
        dest_base = dest_dir
    
    # Get best cutoff from model performance metrics
    print("Determining best cutoff...")
    metrics_file = modeling_results_src / "model_performance" / "model_performance_metrics.csv"
    best_cutoff = get_best_cutoff(metrics_file)
    
    if not best_cutoff:
        print(f"  ✗ Cannot determine best cutoff for {workflow_name} workflow")
        return export_summary
    
    export_summary['best_cutoff'] = best_cutoff
    
    if dry_run:
        print(f"\n[DRY RUN] Would export {workflow_name} workflow with best cutoff: {best_cutoff}")
        export_summary['success'] = True
        return export_summary
    
    # Copy model performance files
    print("\nProcessing model_performance/...")
    model_perf_src = modeling_results_src / "model_performance"
    model_perf_dest = dest_base / "modeling_results" / "model_performance"
    
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
    
    # Copy ONLY the best cutoff's filtered feature table
    print(f"Processing filtered_feature_tables for cutoff_{best_cutoff}...")
    filtered_tables_src = feature_selection_src / "filtered_feature_tables"
    filtered_tables_dest = dest_base / "feature_selection" / "filtered_feature_tables"
    
    # Copy the specific feature table for best cutoff
    best_feature_table = f"select_feature_table_cutoff_{best_cutoff}.csv"
    src_file = filtered_tables_src / best_feature_table
    dest_file = filtered_tables_dest / best_feature_table
    
    if copy_file_safe(src_file, dest_file, best_feature_table):
        print(f"  ✓ Copied best cutoff feature table: {best_feature_table}")
        total_files_copied += 1
    
    # Copy additional top-level files from modeling_results
    print("Processing modeling_results top-level files...")
    top_level_files = [
        "select_features_model_performance.csv",
        "select_features_model_predictions.csv"
    ]
    
    files_copied = 0
    for filename in top_level_files:
        src_file = modeling_results_src / filename
        dest_file = dest_base / "modeling_results" / filename
        if copy_file_safe(src_file, dest_file, filename):
            files_copied += 1
    
    if files_copied > 0:
        print(f"  ✓ Copied {files_copied} top-level files from modeling_results")
        total_files_copied += files_copied
    
    # Copy ONLY the best cutoff directory
    print(f"Processing cutoff_{best_cutoff} directory...")
    best_cutoff_src = modeling_results_src / f"cutoff_{best_cutoff}"
    best_cutoff_dest = dest_base / "modeling_results" / f"cutoff_{best_cutoff}"
    
    if not best_cutoff_src.exists():
        print(f"  ✗ Best cutoff directory not found: cutoff_{best_cutoff}")
    else:
        # Find all run directories in the best cutoff
        run_dirs = sorted([d for d in best_cutoff_src.iterdir() 
                          if d.is_dir() and d.name.startswith('run_')])
        
        cutoff_files_copied = 0
        for run_dir in run_dirs:
            run_name = run_dir.name
            dest_run_dir = best_cutoff_dest / run_name
            
            # Copy importance files
            importance_files = ["shap_importances.csv", "feature_importances.csv"]
            
            for filename in importance_files:
                src_file = run_dir / filename
                dest_file = dest_run_dir / filename
                if copy_file_safe(src_file, dest_file, f"cutoff_{best_cutoff}/{run_name}/{filename}"):
                    cutoff_files_copied += 1
        
        if cutoff_files_copied > 0:
            print(f"  ✓ Copied {cutoff_files_copied} importance files from cutoff_{best_cutoff}")
            total_files_copied += cutoff_files_copied
        else:
            print(f"  ✗ No importance files found in cutoff_{best_cutoff}")
    
    # Copy workflow-specific additional files
    if workflow_name == "kmer_modeling":
        print("Processing kmer_modeling specific files...")
        
        # Copy feature_tables directory
        feature_tables_src = source_dir / "feature_tables"
        feature_tables_dest = dest_base / "feature_tables"
        if copy_directory_safe(feature_tables_src, feature_tables_dest, "feature_tables directory"):
            print(f"  ✓ Copied feature_tables directory")
            total_files_copied += 1
        
        # Copy full_feature_table.csv
        full_feature_src = source_dir / "full_feature_table.csv"
        full_feature_dest = dest_base / "full_feature_table.csv"
        if copy_file_safe(full_feature_src, full_feature_dest, "full_feature_table.csv"):
            print(f"  ✓ Copied full_feature_table.csv")
            total_files_copied += 1
        
        # Copy workflow_report.txt
        workflow_report_src = source_dir / "workflow_report.txt"
        workflow_report_dest = dest_base / "workflow_report.txt"
        if copy_file_safe(workflow_report_src, workflow_report_dest, "workflow_report.txt"):
            print(f"  ✓ Copied workflow_report.txt")
            total_files_copied += 1
    
    else:  # Main workflow
        print("Processing main workflow specific files...")
        
        # Copy phage and strain features directories
        for feature_type in ["phage", "strain"]:
            print(f"Processing {feature_type}/features/...")
            features_src = source_dir / feature_type / "features"
            features_dest = dest_base / feature_type / "features"
            
            if features_src.exists():
                # Copy the entire features directory
                if copy_directory_safe(features_src, features_dest, f"{feature_type}/features directory"):
                    print(f"  ✓ Copied {feature_type}/features directory")
                    total_files_copied += 1
            else:
                print(f"  ⚠ {feature_type}/features directory not found")
        
        # Also copy the main feature files that might be useful
        additional_files = [
            "phage/presence_absence_matrix.csv",
            "strain/presence_absence_matrix.csv",
            "merged/full_feature_table.csv",
            "phage/clusters.tsv",
            "strain/clusters.tsv"
        ]
        
        files_copied = 0
        for file_path in additional_files:
            src_file = source_dir / file_path
            dest_file = dest_base / file_path
            if copy_file_safe(src_file, dest_file, file_path):
                files_copied += 1
        
        if files_copied > 0:
            print(f"  ✓ Copied {files_copied} additional feature files")
            total_files_copied += files_copied
    
    export_summary['files_copied'] = total_files_copied
    export_summary['success'] = total_files_copied > 0
    
    print(f"\n{workflow_name} workflow export summary:")
    print(f"  Best cutoff: {best_cutoff}")
    print(f"  Files/directories copied: {total_files_copied}")
    
    return export_summary


def export_all_workflows(source_dir, dest_dir, dry_run=False):
    """
    Export all workflow modeling results.
    
    Args:
        source_dir (Path): Source directory containing workflow results
        dest_dir (Path): Destination directory for export
        dry_run (bool): If True, only show what would be exported
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    
    # Validate source directory exists
    if not source_dir.exists():
        print(f"✗ Error: Source directory '{source_dir}' does not exist")
        sys.exit(1)
    
    if not dry_run:
        # Create destination directory
        dest_dir.mkdir(parents=True, exist_ok=True)
    
    print("Exporting workflow modeling results...")
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    if dry_run:
        print("*** DRY RUN MODE - No files will be copied ***")
    
    export_summaries = []
    
    # Export main workflow
    main_summary = export_workflow_results(source_dir, dest_dir, "main", "", dry_run)
    export_summaries.append(main_summary)
    
    # Export kmer_modeling workflow if it exists
    kmer_modeling_src = source_dir / "kmer_modeling"
    if kmer_modeling_src.exists():
        kmer_summary = export_workflow_results(
            kmer_modeling_src, 
            dest_dir, 
            "kmer_modeling", 
            "kmer_modeling",
            dry_run
        )
        export_summaries.append(kmer_summary)
    else:
        print(f"\n⚠ kmer_modeling directory not found, skipping...")
    
    # Print overall summary
    print(f"\n{'='*60}")
    if dry_run:
        print("DRY RUN COMPLETE!")
    else:
        print("EXPORT COMPLETE!")
    print(f"{'='*60}")
    
    total_files = sum(summary['files_copied'] for summary in export_summaries)
    successful_workflows = sum(1 for summary in export_summaries if summary['success'])
    
    for summary in export_summaries:
        status = "✓" if summary['success'] else "✗"
        if dry_run:
            print(f"{status} {summary['workflow']} workflow: ready to export, best cutoff: {summary['best_cutoff']}")
        else:
            print(f"{status} {summary['workflow']} workflow: {summary['files_copied']} files, best cutoff: {summary['best_cutoff']}")
    
    print(f"\nWorkflows processed: {successful_workflows}/{len(export_summaries)}")
    if not dry_run:
        print(f"Total files/directories copied: {total_files}")
        
        # Calculate and display total size
        total_size = get_directory_size(dest_dir)
        print(f"Total exported size: {total_size}")
        print(f"Results saved to: {dest_dir}")
        
        # Create manifest file
        create_manifest(dest_dir, source_dir, export_summaries)


def create_manifest(dest_dir, source_dir, export_summaries):
    """Create a manifest file documenting the export."""
    manifest_file = dest_dir / "export_manifest.txt"
    
    manifest_content = f"""Full Workflow Modeling Export Manifest
==========================================
Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source Directory: {source_dir}
Destination Directory: {dest_dir}

Workflow Summaries:
"""
    
    for summary in export_summaries:
        status = "SUCCESS" if summary['success'] else "FAILED"
        manifest_content += f"""
{summary['workflow'].upper()} WORKFLOW - {status}:
  Best Cutoff: {summary['best_cutoff']}
  Files Copied: {summary['files_copied']}
"""
    
    manifest_content += f"""

Files Exported Per Workflow:

MAIN WORKFLOW:
- modeling_results/model_performance/ (performance metrics and plots)
- feature_selection/filtered_feature_tables/select_feature_table_cutoff_[BEST].csv
- modeling_results/select_features_model_performance.csv
- modeling_results/select_features_model_predictions.csv
- modeling_results/cutoff_[BEST]/run_*/[shap_importances.csv, feature_importances.csv]
- phage/features/ (feature_assignments.csv, feature_table.csv, selected_features.csv)
- strain/features/ (feature_assignments.csv, feature_table.csv, selected_features.csv)
- phage/presence_absence_matrix.csv
- strain/presence_absence_matrix.csv
- phage/clusters.tsv
- strain/clusters.tsv
- merged/full_feature_table.csv

KMER_MODELING WORKFLOW (if present):
- kmer_modeling/feature_tables/ (all files including selected_features.csv, feature_assignment.csv, etc.)
- kmer_modeling/full_feature_table.csv
- kmer_modeling/workflow_report.txt
- kmer_modeling/modeling/model_performance/ (performance metrics and plots)
- kmer_modeling/modeling/feature_selection/filtered_feature_tables/select_feature_table_cutoff_[BEST].csv
- kmer_modeling/modeling/modeling_results/select_features_model_performance.csv
- kmer_modeling/modeling/modeling_results/select_features_model_predictions.csv
- kmer_modeling/modeling/modeling_results/cutoff_[BEST]/run_*/[shap_importances.csv, feature_importances.csv]

Usage:
This export contains the key results and performance metrics from
both the main workflow and kmer_modeling workflow. Only the best-performing
cutoff results are included for each workflow to minimize storage requirements
while preserving the most important outputs for analysis.

The best cutoffs are determined automatically from model_performance_metrics.csv
in each workflow's model_performance directory.
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
        description="Export workflow modeling results from HPC cluster (main + kmer_modeling)",
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
    
    # Convert to Path objects and export
    source_path = Path(args.input).resolve()
    dest_path = Path(args.output).resolve()
    
    export_all_workflows(source_path, dest_path, args.dry_run)


if __name__ == "__main__":
    main()