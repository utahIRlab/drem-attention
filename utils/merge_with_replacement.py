import csv
import sys

def merge(source_file, source_field, file_to_replace, field_to_replace, output_file):
    review_dict = dict()
    with open(source_file, mode='r') as fin:
        reviews = csv.DictReader(fin)
        for row in reviews:
            review_dict[row['sample_id']] = row

    explanation_dict = dict()
    field_names = None
    with open(file_to_replace, mode='r') as fin:
        reviews = csv.DictReader(fin)
        for row in reviews:
            if row['sample_id'] in review_dict:
                row[field_to_replace] = review_dict[row['sample_id']][source_field]
                explanation_dict[row['sample_id']] = row
                field_names = list(row.keys())

    with open(output_file, mode='w') as merged_expln_file:
        writer = csv.DictWriter(merged_expln_file, fieldnames=field_names)
        writer.writeheader()
        for _, row in explanation_dict.items():
            writer.writerow(row)


if __name__ == "__main__":
    source_file = sys.argv[1]
    source_field = sys.argv[2]
    file_to_replace = sys.argv[3]
    field_to_replace = sys.argv[4]
    output_file = sys.argv[5]
    merge(source_file, source_field, file_to_replace, field_to_replace, output_file)

