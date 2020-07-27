#!/usr/bin/env bash

#if [ $# -lt 3 ]; then
#    echo "Incorrect arguments"
#    exit 1
#fi

#$1 -> explanation file 1, $2 -> explanation file 2, $3 output path

#PROGRAM_PATH = '/home/aiqy/Project/EGAM/drem-attention/utils/'
#drem_expln_file = '/raid/aiqy/EGAM/output/Lakshimi_data/electronics/min_count5/drem/explanation-output.csv'#sys.argv[1]
#drem_attn_expln_file = '/raid/aiqy/EGAM/output/Lakshimi_data/electronics/min_count5/drem-attention/explanation-output.csv'#sys.argv[2]
#merged_expln_file_path = '/raid/aiqy/EGAM/output/Lakshimi_data/electronics/min_count5/'#sys.argv[3]

python utils/merge_with_replacement.py ./example/AmazonDataset/electronics/drem-explanation-output.csv explanation ./example/AmazonDataset/electronics/merged_data_with_web.csv drem_explanation ./example/AmazonDataset/electronics/original.csv

python utils/merge_with_replacement.py ./example/AmazonDataset/electronics/drem-attention-explanation-output.csv explanation ./example/AmazonDataset/electronics/original.csv drem_attn_explanation ./example/AmazonDataset/electronics/original.csv

python utils/merge_with_replacement.py ./example/AmazonDataset/electronics/drem-attention-explanation-output.csv previous_reviews ./example/AmazonDataset/electronics/original.csv previous_reviews ./example/AmazonDataset/electronics/original.csv

python utils/csv_reorganizer.py ./example/AmazonDataset/electronics/original.csv ./example/AmazonDataset/electronics/mturk-input.csv

python utils/csv_sampler.py ./example/AmazonDataset/electronics/mturk-input.csv ./example/AmazonDataset/electronics/sample.csv 5 5