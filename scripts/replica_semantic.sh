#!/usr/env/bin bash
set -eo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
scene="$1"
number="$2"
prefix="${script_dir}/../datasets/replica_semantic/${scene}${number}"
mkdir -p "${prefix}/results_frames"
mkdir -p "${prefix}/results_segmentation_maps"
mkdir -p "${prefix}/results_segmentation_labels"
cp ${prefix}/results/frame* ${prefix}/results_frames/
. "${XDG_PREFIX_HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate Tracking-Anything-with-DEVA
cd /mnt/dev-ssd-8T/shuqixiao/dev/projects/semantic-3dgs-slam/third_party/Tracking-Anything-with-DEVA/
python demo/demo_automatic.py --img_path "${prefix}/results_frames" \
	--output "${prefix}/results_segmentation_maps" \
	--temporal_setting semionline --suppress_small_objects --amp --save_all --sam_variant original --size 480 --num_prototypes 256
conda deactivate
cd $script_dir
