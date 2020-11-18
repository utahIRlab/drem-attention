import os,sys

baseline_ranklist_path = '/raid/laknaren/drem/output/electronics/test.bias_product.ranklist'
treatment_ranklist_path = '/raid/laknaren/drem-attention/output/electronics/test.bias_product.ranklist'
DATA_PATH = '/raid/laknaren/datasets/electronics/min_count5/'
OUTPUT_PATH = './'


def get_galago_eval_details(file_path, out_path):
    #METRICS = ' --metrics+map --metrics+recip_rank --metrics+ndcg10 --metrics+ndcg50 --metrics+P50'
    METRICS = ' --metrics+recip_rank'

    # run evaluation
    command = 'galago eval'
    command += ' --judgments=' + DATA_PATH + 'query_split/test.qrels'
    command += METRICS
    command += ' --runs+' + file_path
    command += ' --details=true'
    command += ' > ' + out_path
    print(command)
    os.system(command)

    # read eval details
    eval_results = {}
    with open(out_path) as fin:
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
    return eval_results

# Get detail results
baseline_eval_results = get_galago_eval_details(baseline_ranklist_path, './baseline_eval.txt')
treat_eval_results = get_galago_eval_details(treatment_ranklist_path, './treat_eval.txt')

# Sample query
sampled_queries = set()
for qid in baseline_eval_results:
    if qid not in treat_eval_results:
        continue
    if baseline_eval_results[qid] > 0.1 and treat_eval_results[qid] > 0.1:
        sampled_queries.add(qid)
print('find %d valid queries' % len(sampled_queries))

# Find corresponding items
def read_results_with_qids(file_path, qids):
    results = {}
    with open(file_path) as fin:
        for line in fin:
            arr = line.split()
            qid = arr[0]
            rank = int(arr[3])
            did = arr[2]
            if qid in qids and rank <= 10 and rank > 0:
                if qid not in results:
                    results[qid] = set()
                results[qid].add(did)
    return results
baseline_docs = read_results_with_qids(baseline_ranklist_path, sampled_queries)
treat_docs = read_results_with_qids(treatment_ranklist_path, sampled_queries)
qrel_docs = read_results_with_qids(DATA_PATH + 'query_split/test.qrels', sampled_queries)

# Sample query-item pairs
with open(OUTPUT_PATH + 'valid_uqi.txt', 'w') as fout:
    for qid in sampled_queries:
        if qid in baseline_docs and qid in treat_docs and qid in qrel_docs:
            valid_docs = baseline_docs[qid] & treat_docs[qid] & qrel_docs[qid]
            for did in valid_docs:
                fout.write('%s\t%s\n' % (qid, did))




