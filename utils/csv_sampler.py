import os,sys
import csv
import re
import random


#----------------------- MAIN FUNCTION ----------------------------
INPUT_CSV_FILE = sys.argv[1]
OUTPUT_CSV_FILE = sys.argv[2]
repeated_limit = int(sys.argv[3])
split_csv_file_num = int(sys.argv[4])

head_row = None
data_rows = []
with open(INPUT_CSV_FILE) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if head_row == None:
            head_row = row
        else:
            data_rows.append(row)

random.shuffle(data_rows)

user_freq_dict = {}
product_freq_dict = {}
query_freq_dict = {}

head_name_dict = {head_row[i]:i for i in range(len(head_row))}

file_list = [open(OUTPUT_CSV_FILE.replace('.csv', '_%d.csv' % i), 'w') for i in range(split_csv_file_num)]
writer_list = [csv.writer(f, delimiter=',') for f in file_list]
for writer in writer_list:
    writer.writerow(head_row)
# Write csv
count = 0
for row in data_rows:
    user_id = row[head_name_dict['user']].strip()
    product_id = row[head_name_dict['product']].strip()
    query = row[head_name_dict['query']].strip()
    if user_id not in user_freq_dict:
        user_freq_dict[user_id] = 0
    if product_id not in product_freq_dict:
        product_freq_dict[product_id] = 0
    if query not in query_freq_dict:
        query_freq_dict[query] = 0

    if user_freq_dict[user_id] >= repeated_limit or product_freq_dict[product_id] >= repeated_limit or query_freq_dict[query] >= repeated_limit:
        continue

    writer_list[count % split_csv_file_num].writerow(row)
    user_freq_dict[user_id] += 1
    product_freq_dict[product_id] += 1
    query_freq_dict[query] += 1
    count += 1

for f in file_list:
    f.close()

'''
with open(OUTPUT_CSV_FILE, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(head_row)
    for row in data_rows:
        user_id = row[head_name_dict['user']].strip()
        product_id = row[head_name_dict['product']].strip()
        query = row[head_name_dict['query']].strip()
        if user_id not in user_freq_dict:
            user_freq_dict[user_id] = 0
        if product_id not in product_freq_dict:
            product_freq_dict[product_id] = 0
        if query not in query_freq_dict:
            query_freq_dict[query] = 0

        if user_freq_dict[user_id] >= repeated_limit or product_freq_dict[product_id] >= repeated_limit or query_freq_dict[query] >= repeated_limit:
            continue

        writer.writerow(row)
        user_freq_dict[user_id] += 1
        product_freq_dict[product_id] += 1
        query_freq_dict[query] += 1
'''

