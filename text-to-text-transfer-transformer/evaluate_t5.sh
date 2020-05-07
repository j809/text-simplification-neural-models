#!/bin/bash
export MODEL_DIR="/mnt/nfs/work1/cs696e/vbhave/t5/model4"

t5_mesh_transformer  \
	--model_dir="${MODEL_DIR}" \
        --gin_file="${MODEL_DIR}/operative_config.gin" \
        --gin_file="infer.gin" \
        --gin_file="sample_decode.gin" \
        --gin_param="input_filename = '${MODEL_DIR}/test_file'" \
        --gin_param="output_filename = '${MODEL_DIR}/my_output_run'" \
        --gin_param="infer_checkpoint_step = 391200" \
        --gin_param="utils.run.mesh_devices = ['gpu:0']" 
