import os,sys
import csv
import re


#----------------------- PROCESS FUNCTION -------------------------

def process_explanation(orig_str, **kwargs):
    # Add <li>
    arr = orig_str.strip().split('\n')
    new_str = '<li>' + '</li>\n<li>'.join(arr) + '</li>'
    # Replace product id with 'this product'
    product = kwargs['product'].strip()
    new_str = new_str.replace("'%s'" % product, 'this product')
    # Add links to products other than this product
    p = re.compile("'[\w\d]{10}'", re.IGNORECASE)
    other_products = [x.strip("'") for x in p.findall(new_str)]
    for op in other_products:
        new_str = new_str.replace(op, '<a href="https://www.amazon.com/dp/%s">%s</a>' % (op,op))
    return new_str

def process_reviews(orig_str, **kwargs):
    arr = orig_str.strip().split('\n')
    new_str = '<li>' + '</li>\n<li>'.join(arr) + '</li>'
    return new_str

def process_description(orig_str, **kwargs):
    arr = [x.strip().strip("'") for x in orig_str.strip('[]').split("',")]
    if len(''.join(arr)) > 0:
        new_str = '<li>' + '</li>\n<li>'.join(arr) + '</li>'
    else:
        new_str = '<li>None</li>'
    return new_str


PROCESS_FUNC_DICT = {
    'drem_explanation' : process_explanation,
    'drem_attn_explanation' : process_explanation,
    'previous_reviews' : process_reviews,
    'description' : process_description,
}

#----------------------- MAIN FUNCTION ----------------------------
INPUT_CSV_FILE = sys.argv[1]
OUTPUT_CSV_FILE = sys.argv[2]

head_row = None
data_rows = []
with open(INPUT_CSV_FILE) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if head_row == None:
            head_row = row
        else:
            data_rows.append(row)

head_name_dict = {head_row[i]:i for i in range(len(head_row))}
with open(OUTPUT_CSV_FILE, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(head_row)
    for row in data_rows:
        for i in range(len(row)):
            if head_row[i] in PROCESS_FUNC_DICT:
                row[i] = PROCESS_FUNC_DICT[head_row[i]](row[i], product=row[head_name_dict['product']])
        writer.writerow(row)


'''
print(','.join(head_row))
for row in data_rows:
    for i in range(len(row)):
        if head_row[i] in PROCESS_FUNC_DICT:
            row[i] = PROCESS_FUNC_DICT[head_row[i]](row[i])
        row[i] = '"' + row[i].replace('"', '""') + '"'
    print(','.join(row))
'''