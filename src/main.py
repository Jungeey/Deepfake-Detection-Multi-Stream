"""Main entry point for the multi-stream deepfake detection project."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.settings import PROJECT_ROOT
from src.data.data_fetcher import main as fetch_data
from src.preprocessing.preprocessing_pipeline import main as run_preprocessing
from src.utils.mps_utils import benchmark_mps, get_mps_device

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Stream Deepfake Detection Framework"
    )
    
    parser.add_argument(
        '--stage', 
        type=str,
        choices=['setup', 'fetch', 'preprocess', 'benchmark', 'all'],
        default='all',
        help='Pipeline stage to run'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of worker threads'
    )
    
    parser.add_argument(
        '--use-mps',
        action='store_true',
        default=True,
        help='Use MPS acceleration if available'
    )
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_args()
    
    print("=" * 60)
    print("Multi-Stream Deepfake Detection Framework")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"MPS Available: {args.use_mps and get_mps_device().type == 'mps'}")
    print("=" * 60)
    
    if args.stage in ['setup', 'all']:
        print("\n[Stage 1: Environment Setup]")
        print("Environment already configured. Skipping...")
        # Setup is handled by conda environment
    
    if args.stage in ['fetch', 'all']:
        print("\n[Stage 2: Data Acquisition]")
        fetch_data()
    
    if args.stage in ['preprocess', 'all']:
        print("\n[Stage 3: Preprocessing Pipeline]")
        run_preprocessing()
    
    if args.stage in ['benchmark']:
        print("\n[Benchmark: MPS Performance]")
        benchmark_mps()
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()