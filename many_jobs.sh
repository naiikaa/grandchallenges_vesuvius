SAMPLE_SIZES=(16 32 64)
VOLUME_DEPTHS=(8 16 32)

for sample_size in "${SAMPLE_SIZES[@]}"; do
  for volume_depth in "${VOLUME_DEPTHS[@]}"; do
    sbatch training_script.sh "$sample_size" "$volume_depth"
  done
done