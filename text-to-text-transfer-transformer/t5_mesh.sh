#!/bin/bash
export MODEL_DIR="/mnt/nfs/work1/cs696e/vbhave/t5/model4"

t5_mesh_transformer  \
	--model_dir="${MODEL_DIR}" \
        --gin_file="dataset.gin" \
        --gin_param="run.train_steps = 1010000" \
        --gin_param="tokens_per_batch=512" \
        --gin_param="utils.run.mesh_shape = 'model:1,batch:1'" \
        --gin_param="utils.run.mesh_devices = ['gpu:0']" \
  	--gin_file="${MODEL_DIR}/pretrained_models_small_operative_config.gin" \
	--gin_param="utils.run.train_dataset_fn = @t5.models.mesh_transformer.tsv_dataset_fn" \
	--gin_param="tsv_dataset_fn.filename = '${MODEL_DIR}/../new_dataset_101k'"
#conda install tensorflow-gpu
#pip install tensorflow-text
