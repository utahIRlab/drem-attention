import os,sys

PROGRAM_PATH = '/home/aiqy/Project/EGAM/drem-attention/ProductSearch'
DATA_PATH = '/raid/laknaren/datasets/electronics/min_count5/'
MODEL_PATH = '/home/aiqy/data/EGAM/output/Lakshimi_data/electronics/min_count5/drem-attention/'
QUERY_SAMPLE_PATH = '/home/aiqy/data/EGAM/output/Lakshimi_data/electronics/min_count5/sampled_uqi_for_explanation.txt'
EXPLANATION_PATH = '/home/aiqy/data/EGAM/output/Lakshimi_data/electronics/min_count5/drem-attention/'

command = ' '.join([
    'python %s/main.py' % PROGRAM_PATH,
    '--min_count=5 --learning_rate=0.5 --max_train_epoch=20 --embed_size=200 --subsampling_rate=1e-4 --L2_lambda=0.005 --batch_size=64 --window_size=3 --negative_sample=5 --rank_cutoff=100 --similarity_func=bias_product --query_weight=0.5  --net_struct=ZAM -–dynamic_weight=0.1',
    '--data_dir=%s' % DATA_PATH, 
    '--input_train_dir=%s/query_split/' % DATA_PATH,
    '--train_dir=%s' % MODEL_PATH,
    '--decode=True',
    '--explanation_output_dir=%s' % EXPLANATION_PATH,
    '--sample_file_for_explanation=%s' % QUERY_SAMPLE_PATH,
    '--test_mode=explanation_features',
    ''
])

print(command)
os.system(command)