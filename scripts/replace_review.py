import csv
import sys

def merge(file_with_reviews, file_to_replace, merged_expln_file):
    review_dict = dict()
    with open(file_with_reviews, mode='r') as fin:
        reviews = csv.DictReader(fin)
        for row in reviews:
            review_dict[row['sample_id']] = row

    explanation_dict = dict()
    field_names = None
    with open(file_to_replace, mode='r') as fin:
        reviews = csv.DictReader(fin)
        for row in reviews:
            if row['sample_id'] in review_dict:
                row['previous_reviews'] = review_dict[row['sample_id']]['previous_reviews']
                explanation_dict[row['sample_id']] = row
                field_names = list(row.keys())

    with open(merged_expln_file, mode='w') as merged_expln_file:
        writer = csv.DictWriter(merged_expln_file, fieldnames=field_names)
        writer.writeheader()
        for _, row in explanation_dict.items():
            writer.writerow(row)


if __name__ == "__main__":
    merge(sys.argv[1], sys.argv[2], sys.argv[3])

