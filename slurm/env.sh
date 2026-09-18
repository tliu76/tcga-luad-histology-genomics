# Sourced by every job. Edit these three lines once for your CARC account.
export SCRATCH_DIR=/scratch1/$USER/mosaic                  # large files: slides (~720 GB), features
export SLIDES=$SCRATCH_DIR/slides
export HF_HOME=$SCRATCH_DIR/hf_cache                         # Phikon weights (download on login node first)
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate mosaic
mkdir -p logs "$SLIDES" "$HF_HOME"
