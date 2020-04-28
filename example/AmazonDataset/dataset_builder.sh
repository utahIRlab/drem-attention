#!/usr/bin/env bash

if [ $# -lt 2 ]; then
    echo "Incorrect arguments"
    exit 1
fi

#$1 -> review file name, $2 -> meta file name

source /home/laknaren/workspace/venv/bin/activate

# Download Amazon review dataset “Cell_Phones_and_Accessories” 5-core.
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/$1.json.gz
# Download the meta data from http://jmcauley.ucsd.edu/data/amazon/
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/$2.json.gz
# Stem and remove stop words from the Amazon review datasets if needed. Here, we stem the field of “reviewText” and “summary” without stop words removal.
java -Xmx4g -jar /home/laknaren/workspace/drem-attention/utils/AmazonDataset/jar/AmazonReviewData_preprocess.jar false ./$1.json.gz ./$1.processed.gz
# Index datasets
python /home/laknaren/workspace/drem-attention/utils/AmazonDataset/index_and_filter_review_file.py $1.processed.gz ./tmp_data/ 5
# Match the meta data with the indexed data to extract queries:
java -Xmx16G -jar /home/laknaren/workspace/drem-attention/utils/AmazonDataset/jar/AmazonMetaData_matching.jar false ./$2.json.gz ./tmp_data/min_count5/
# Gather knowledge from meta data:
python /home/laknaren/workspace/drem-attention/utils/AmazonDataset/match_with_meta_knowledge.py ./tmp_data/min_count5/ $2.json.gz
# Randomly split train/test
## The 30% purchases of each user are used as test data
## Also, we randomly sample 20% queries and make them unique in the test set.
python /home/laknaren/workspace/drem-attention/utils/AmazonDataset/random_split_train_test_data.py tmp_data/s/ 0.3 0.3
# Sequentially split train/test
## The last 20% purchases of each user are used as test data
## Also, we manually sample 20% queries and make them unique in the test set.
#python utils/AmazonDataset/sequentially_split_train_test_data.py tmp_data/old_min_count5/ 0.2 0.2
python3 /home/laknaren/workspace/drem-attention/utils/AmazonDataset/collect_time_seq_info.py tmp_data/min_count5 ./$1.json.gz

mv tmp_data/min_count5/random_query_split/ tmp_data/min_count5/query_split/