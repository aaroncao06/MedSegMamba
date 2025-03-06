#bash run_parallel.sh /path/to/input_dir /path/to/base_dir num_jobs Step1_FS_autorecon1_run.sh

INPUT_DIR="$1"  # Directory containing .nii files
BASE_DIR="$2"  # Base directory for output
NUM_JOBS="${3:-18}"  # Number of parallel jobs (default: 18)
SCRIPT_PATH="${4:-Step1_FS_autorecon1_run.sh}"  # Path to Step1_FS_autorecon1.sh

# Check if input directory exists
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory $INPUT_DIR does not exist."
    exit 1
fi

# Run the script in parallel
find "$INPUT_DIR" -maxdepth 1 -name "*.nii" | parallel -j "$NUM_JOBS" bash "$SCRIPT_PATH" {} "$BASE_DIR"
