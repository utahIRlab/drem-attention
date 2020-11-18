import os,sys

PROGRAM_PATH = '/home/aiqy/Project/EGAM/drem-attention/utils/'
drem_expln_file = '/raid/aiqy/EGAM/output/Lakshimi_data/electronics/min_count5/drem/explanation-output.csv'#sys.argv[1]
drem_attn_expln_file = '/raid/aiqy/EGAM/output/Lakshimi_data/electronics/min_count5/drem-attention/explanation-output.csv'#sys.argv[2]
merged_expln_file_path = '/raid/aiqy/EGAM/output/Lakshimi_data/electronics/min_count5/'#sys.argv[3]

# merge_explanation
command = ' '.join([
    'python %s/merge_explanations.py' % PROGRAM_PATH,
    drem_expln_file,
    drem_attn_expln_file,
    merged_expln_file_path + 'merged.csv'
])

print(command)
os.system(command)

# get detailed information from the web
command = ' '.join([
    'python %s/web_scrapper.py' % PROGRAM_PATH,
    merged_expln_file_path + 'merged.csv',
    merged_expln_file_path + 'merged_with_web.csv'
])

print(command)
os.system(command)

# reorganize csv
command = ' '.join([
    'python %s/csv_reorganizer.py' % PROGRAM_PATH,
    merged_expln_file_path + 'merged_with_web.csv',
    merged_expln_file_path + 'mturk-input.csv',
])

print(command)
os.system(command)