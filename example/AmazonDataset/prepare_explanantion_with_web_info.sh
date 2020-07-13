#!/usr/bin/env bash

if [ $# -lt 3 ]; then
    echo "Incorrect arguments"
    exit 1
fi

#$1 -> explanation file 1, $2 -> explanation file 2, $3 output path

PROGRAM_PATH = '/home/aiqy/Project/EGAM/drem-attention/utils/'
drem_expln_file = '/raid/aiqy/EGAM/output/Lakshimi_data/electronics/min_count5/drem/explanation-output.csv'#sys.argv[1]
drem_attn_expln_file = '/raid/aiqy/EGAM/output/Lakshimi_data/electronics/min_count5/drem-attention/explanation-output.csv'#sys.argv[2]
merged_expln_file_path = '/raid/aiqy/EGAM/output/Lakshimi_data/electronics/min_count5/'#sys.argv[3]

# merge_explanation
python ./utils/merge_explanations.py $1 $2 $3/merged.csv

# get detailed information from the web
python ./utils/web_scrapper.py $3/merged.csv $3/merged_with_web.csv

# reorganize csv
python ./utils/csv_reorganizer.py $3/merged_with_web.csv $3/mturk-input.csv
