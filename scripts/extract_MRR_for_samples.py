import os, sys
import csv, gzip

sample_id_csv = sys.argv[1]
user_id_file = sys.argv[2]
galago_eval_files = sys.argv[3:]

# Read user id
user_ids = []
with gzip.open(user_id_file, 'rt') as fin:
    for line in fin:
        user_ids.append(line.strip())

# Read sample id csv
head_row = None
data_rows = []
with open(sample_id_csv) as csvfile: # the first column must be the sample id
    reader = csv.reader(csvfile, delimiter='\t')
    for row in reader:
        if head_row == None:
            head_row = row
        else:
            data_rows.append(row)

# read performance
perf_dict = None
for file in galago_eval_files:
    eval_results = {}
    with open(file) as fin:
        for line in fin:
            arr = line.split()
            if len(arr) != 2:
                continue
            value = None
            try:
                value = float(arr[1])
            except:
                continue
            eval_results[arr[0]] = value
    if perf_dict == None:
        perf_dict = {k: [eval_results[k]] for k in eval_results}
    else:
        for k in eval_results:
            perf_dict[k].append(eval_results[k])

# match and output performance
head_row.extend([f.split('/')[-1] for f in galago_eval_files])
with open(sample_id_csv + '.updated.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    writer.writerow(head_row)
    for row in data_rows:
        sample_id = row[0]
        arr = sample_id.split('-')
        qid = '_'.join([user_ids[int(arr[0])], arr[2]])
        row.extend(perf_dict[qid])
        writer.writerow(row)



