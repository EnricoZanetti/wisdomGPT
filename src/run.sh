#!/bin/bash

ARG_COMPILE=--${2:-'no-compile'}
ARG_BACKEND=--backend=${3:-'inductor'}

if [ "$1" = "vanilla_finetune_without_pretrain" ]; then
	python run.py $ARG_COMPILE $ARG_BACKEND --function=finetune --variant=vanilla --pretrain_corpus_path=./data/wiki.txt --writing_params_path=./core/vanilla.model.params --finetune_corpus_path=./data/birth_places_train.tsv
elif [ "$1" = "vanilla_eval_dev_without_pretrain" ]; then
	if [ -f ./core/vanilla.model.params ]; then
    	python run.py $ARG_COMPILE $ARG_BACKEND --function=evaluate --variant=vanilla --pretrain_corpus_path=./data/wiki.txt --reading_params_path=./core/vanilla.model.params --eval_corpus_path=./data/birth_dev.tsv --outputs_path=./core/vanilla.nopretrain.dev.predictions
	else
		echo "'./core/vanilla.model.params' does not exist. Please run './run.sh vanilla_finetune_without_pretrain' on the VM to create this file."
	fi
elif [ "$1" = "vanilla_eval_test_without_pretrain" ]; then
	if [ -f ./core/vanilla.model.params ]; then
		python run.py $ARG_COMPILE $ARG_BACKEND --function=evaluate --variant=vanilla --pretrain_corpus_path=./data/wiki.txt --reading_params_path=./core/vanilla.model.params --eval_corpus_path=./data/birth_test_inputs.tsv --outputs_path=./core/vanilla.nopretrain.test.predictions
	else
		echo "'./core/vanilla.model.params' does not exist. Please run './run.sh vanilla_finetune_without_pretrain' on the VM to create this file."
	fi
elif [ "$1" = "vanilla_pretrain" ]; then
	echo "Starting Vanilla Pretrain: ~ 2 Hours"
    python run.py $ARG_COMPILE $ARG_BACKEND --function=pretrain --variant=vanilla --pretrain_corpus_path=./data/wiki.txt --writing_params_path=./core/vanilla.pretrain.params
elif [ "$1" = "vanilla_finetune_with_pretrain" ]; then
	if [ -f ./core/vanilla.pretrain.params ]; then
		python run.py $ARG_COMPILE $ARG_BACKEND --function=finetune --variant=vanilla --pretrain_corpus_path=./data/wiki.txt --reading_params_path=./core/vanilla.pretrain.params --writing_params_path=./core/vanilla.finetune.params --finetune_corpus_path=./data/birth_places_train.tsv
	else
		echo "'./core/vanilla.pretrain.params' does not exist. Please run './run.sh vanilla_pretrain' on the VM to create this file. Note: will take around 2 hours."
	fi
elif [ "$1" = "vanilla_eval_dev_with_pretrain" ]; then
	if [ -f ./core/vanilla.finetune.params ]; then
		python run.py $ARG_COMPILE $ARG_BACKEND --function=evaluate --variant=vanilla --pretrain_corpus_path=./data/wiki.txt --reading_params_path=./core/vanilla.finetune.params --eval_corpus_path=./data/birth_dev.tsv --outputs_path=./core/vanilla.pretrain.dev.predictions
	else
		echo "'./core/vanilla.finetune.params' does not exist. Please run './run.sh vanilla_finetune_with_pretrain' on the VM to create this file."
	fi
elif [ "$1" = "vanilla_eval_test_with_pretrain" ]; then
	if [ -f ./core/vanilla.finetune.params ]; then
		python run.py $ARG_COMPILE $ARG_BACKEND --function=evaluate --variant=vanilla --pretrain_corpus_path=./data/wiki.txt --reading_params_path=./core/vanilla.finetune.params --eval_corpus_path=./data/birth_test_inputs.tsv --outputs_path=./core/vanilla.pretrain.test.predictions
	else
		echo "'./core/vanilla.finetune.params' does not exist. Please run './run.sh vanilla_finetune_with_pretrain' on the VM to create this file."
	fi
elif [ "$1" = "perceiver_pretrain" ]; then
	echo "Starting Perceiver Pretrain: ~ 2 Hours"
	python run.py $ARG_COMPILE $ARG_BACKEND --function=pretrain --variant=perceiver --pretrain_corpus_path=./data/wiki.txt --writing_params_path=./core/perceiver.pretrain.params	
elif [ "$1" = "perceiver_finetune_with_pretrain" ]; then
	if [ -f ./core/perceiver.pretrain.params ]; then
		python run.py $ARG_COMPILE $ARG_BACKEND --function=finetune --variant=perceiver --pretrain_corpus_path=./data/wiki.txt --reading_params_path=./core/perceiver.pretrain.params --writing_params_path=./core/perceiver.finetune.params --finetune_corpus_path=./data/birth_places_train.tsv
	else
		echo "'./core/perceiver.pretrain.params' does not exist. Please run './run.sh perceiver_finetune_with_pretrain' on the VM to create this file. Note: will take around 2 hours."
	fi
elif [ "$1" = "perceiver_eval_dev_with_pretrain" ]; then
	if [ -f ./core/perceiver.finetune.params ]; then
		python run.py $ARG_COMPILE $ARG_BACKEND --function=evaluate --variant=perceiver --pretrain_corpus_path=./data/wiki.txt --reading_params_path=./core/perceiver.finetune.params --eval_corpus_path=./data/birth_dev.tsv --outputs_path=./core/perceiver.pretrain.dev.predictions	
	else
		echo "'./core/perceiver.finetune.params' does not exist. Please run './run.sh vanilla_finetune_with_pretrain' on the VM to create this file."
	fi
elif [ "$1" = "perceiver_eval_test_with_pretrain" ]; then
	if [ -f ./core/perceiver.finetune.params ]; then
		python run.py $ARG_COMPILE $ARG_BACKEND --function=evaluate --variant=perceiver --pretrain_corpus_path=./data/wiki.txt --reading_params_path=./core/perceiver.finetune.params --eval_corpus_path=./data/birth_test_inputs.tsv --outputs_path=./core/perceiver.pretrain.test.predictions	
	else
		echo "'./core/perceiver.finetune.params' does not exist. Please run './run.sh vanilla_finetune_with_pretrain' on the VM to create this file."
	fi 
else
	echo "Invalid Option Selected. Only Options Available Are:"
	echo "=============================================================="
	echo "./run.sh vanilla_finetune_without_pretrain"
	echo "./run.sh vanilla_eval_dev_without_pretrain"
	echo "./run.sh vanilla_eval_test_without_pretrain"
	echo "------------------------------------------------------------"
	echo "./run.sh vanilla_pretrain"
	echo "./run.sh vanilla_finetune_with_pretrain"
	echo "./run.sh vanilla_eval_dev_with_pretrain"
	echo "./run.sh vanilla_eval_test_with_pretrain"
	echo "------------------------------------------------------------"
	echo "./run.sh perceiver_pretrain"
	echo "./run.sh perceiver_finetune_with_pretrain"
	echo "./run.sh perceiver_eval_dev_with_pretrain"
	echo "./run.sh perceiver_eval_test_with_pretrain"
	echo "------------------------------------------------------------"
fi