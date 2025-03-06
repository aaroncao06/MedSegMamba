#bash Step1_FS_autorecon1.sh /path/to/T1.nii /path/to/base_dir

T1="$1"
BASE_DIR="$2"

T1_FILENAME=$(basename "$T1")
SUBJECT_ID="${T1_FILENAME%.*}"  # Removes the file extension

STEP1_DIR="$BASE_DIR/Step1_3DDeepC_T1_FS"
STEP1_PREPROCESSED_DIR="$BASE_DIR/Step1_3DDeepC_T1_FS_Preprocessed"

mkdir -p "$STEP1_DIR"
mkdir -p "$STEP1_PREPROCESSED_DIR"

recon-all -subjid "$SUBJECT_ID" -sd "$STEP1_DIR" -autorecon1 -notal-check -i "$T1"

mri_convert "$STEP1_DIR/$SUBJECT_ID/mri/brainmask.mgz" "$STEP1_PREPROCESSED_DIR/$SUBJECT_ID.nii.gz"