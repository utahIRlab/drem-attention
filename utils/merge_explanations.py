import csv
import sys

def merge(drem_expln_file, drem_attn_expln_file, merged_expln_file):
	drem_dict = dict()
	with open(drem_expln_file, mode='r') as drem_file:
		drem_explanations = csv.DictReader(drem_file)
		for row in drem_explanations:
			drem_dict[row['sample_id']] = row

	explanation_dict = dict()
	field_names = None
	with open(drem_attn_expln_file, mode='r') as drem_attn_file:
		drem_attn_explanations = csv.DictReader(drem_attn_file)
		for row in drem_attn_explanations:
			if row['sample_id'] in drem_dict:
				row['drem_explanation'] = drem_dict[row['sample_id']]['explanation']
				row['drem_attn_explanation'] = row['explanation']
				del row['explanation']
				explanation_dict[row['sample_id']] = row
				field_names = list(row.keys())

	with open(merged_expln_file, mode='w') as merged_expln_file:
		writer = csv.DictWriter(merged_expln_file, fieldnames=field_names)
		writer.writeheader()
		for _, row in explanation_dict.items():
			writer.writerow(row)



if __name__ == "__main__":
	merge(sys.argv[1], sys.argv[2], sys.argv[3])

