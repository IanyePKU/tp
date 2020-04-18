#!/bin/bash
cd ../
experiment_root=./experiments/mlp_ddtp_findlr
experiment_name=(ddtp_lr_same_0.01 ddtp_lr_same_0.005 ddtp_lr_same_0.001 ddtp_lr_same_0.0005)

for i in {0..3}
do
	python train.py --config ${experiment_root}/${experiment_name[${i}]}/config.yaml \
		| tee ${experiment_root}/${experiment_name[${i}]}/log.txt &
done
