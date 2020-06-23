import numpy as np
import json
import random
import gzip
import math

#EXPLANATION_TMPL = "This product was retrieved for this query because the user often buys products related to {entity_type} '{entity_name}' which is also related to the query"
#EXPLANATION_TMPL_DEFAULT = "This product was retrieved for this query because of the product's popularity"

class Tensorflow_data:
	def __init__(self, data_path, input_train_dir, set_name):
		#get product/user/vocabulary information
		self.product_ids = []
		with gzip.open(data_path + 'product.txt.gz', 'rt') as fin:
			for line in fin:
				self.product_ids.append(line.strip())
		self.product_size = len(self.product_ids)
		self.user_ids = []
		with gzip.open(data_path + 'users.txt.gz', 'rt') as fin:
			for line in fin:
				self.user_ids.append(line.strip())
		self.user_size = len(self.user_ids)
		self.words = []
		with gzip.open(data_path + 'vocab.txt.gz', 'rt') as fin:
			for line in fin:
				self.words.append(line.strip())
		self.vocab_size = len(self.words)
		self.query_words = []
		self.query_max_length = 0
		with gzip.open(input_train_dir + 'query.txt.gz', 'rt') as fin:
			for line in fin:
				words = [int(i) for i in line.strip().split(' ')]
				if len(words) > self.query_max_length:
					self.query_max_length = len(words)
				self.query_words.append(words)
		#pad
		for i in range(len(self.query_words)):
			self.query_words[i] = [self.vocab_size for j in range(self.query_max_length-len(self.query_words[i]))] + self.query_words[i]

		# get knowledge
		self.related_product_ids = []
		with gzip.open(data_path + 'related_product.txt.gz', 'rt') as fin:
			for line in fin:
				self.related_product_ids.append(line.strip())
		self.related_product_size = len(self.related_product_ids)
		self.brand_ids = []
		with gzip.open(data_path + 'brand.txt.gz', 'rt') as fin:
			for line in fin:
				self.brand_ids.append(line.strip())
		self.brand_size = len(self.brand_ids)
		self.category_ids = []
		with gzip.open(data_path + 'category.txt.gz', 'rt') as fin:
			for line in fin:
				self.category_ids.append(line.strip())
		self.category_size = len(self.category_ids)

		self.entity_vocab = {
			'user': self.user_ids,
			'word': self.words,
			'product': self.product_ids,
			'related_product': self.related_product_ids,
			'brand': self.brand_ids,
			'categories': self.category_ids
		}
		knowledge_file_dict = {
			'also_bought': data_path + 'also_bought_p_p.txt.gz',
			'also_viewed': data_path + 'also_viewed_p_p.txt.gz',
			'bought_together': data_path + 'bought_together_p_p.txt.gz',
			'brand': data_path + 'brand_p_b.txt.gz',
			'categories': data_path + 'category_p_c.txt.gz'
		}
		knowledge_vocab = {
			'also_bought': self.related_product_ids,
			'also_viewed': self.related_product_ids,
			'bought_together': self.related_product_ids,
			'brand': self.brand_ids,
			'categories': self.category_ids
		}
		self.knowledge = {}
		for name in knowledge_file_dict:
			self.knowledge[name] = {}
			self.knowledge[name]['data'] = []
			self.knowledge[name]['vocab'] = knowledge_vocab[name]
			self.knowledge[name]['distribute'] = np.zeros(len(self.knowledge[name]['vocab']))
			with gzip.open(knowledge_file_dict[name], 'rt') as fin:
				for line in fin:
					knowledge = []
					arr = line.strip().split(' ')
					for x in arr:
						if len(x) > 0:
							x = int(x)
							knowledge.append(x)
							self.knowledge[name]['distribute'][x] += 1
					self.knowledge[name]['data'].append(knowledge)
			self.knowledge[name]['distribute'] = self.knowledge[name]['distribute'].tolist()

		# Read review id
		self.review_id_to_idx = {}
		with gzip.open(data_path + "review_id.txt.gz", 'rt') as fin:
			index = 0
			for line in fin:
				self.review_id_to_idx[line.strip()] = index
				index += 1

		# get flags of whether a review is in the training data
		self.is_train_review = [False for _ in range(len(self.review_id_to_idx))]
		with gzip.open(input_train_dir + 'train_id.txt.gz', 'rt') as fin:
			for line in fin:
				arr = line.strip().split('\t')
				original_review_idx = self.review_id_to_idx[arr[2]]
				self.is_train_review[original_review_idx] = True

		self.user_history_idxs = {
			'review': [],
			'product': [],
			'brand': [],
			'categories': [],
			'also_bought': [],
			'also_viewed': [],
			'bought_together': []
		}
		# get the user history review and user product set
		with gzip.open(data_path + "u_r_seq.txt.gz", 'rt') as fin:
			for line in fin:
				review_ids = line.rstrip().split(" ")
				reviews = [int(review_id_str) for review_id_str in review_ids if str.isnumeric(review_id_str)]
				train_reviews = []
				for review_idx in reviews:
					if self.is_train_review[review_idx]:
						train_reviews.append(review_idx)
				self.user_history_idxs['review'].append(train_reviews)
				self.user_history_idxs['product'].append([-1 for _ in range(len(self.user_history_idxs['review'][-1]))])

		# get user product idx data
		with gzip.open(data_path + "review_u_p.txt.gz", 'rt') as fin:
			review_idx = 0
			for line in fin:
				user_idx, product_idx = int(line.rstrip().split(" ")[0]), int(line.rstrip().split(" ")[1])
				if review_idx in self.user_history_idxs['review'][user_idx]:
					pos = self.user_history_idxs['review'][user_idx].index(review_idx)
					self.user_history_idxs['product'][user_idx][pos] = product_idx
				review_idx += 1

		self.max_history_length = 0
		self.history_length = []

		for product_idxs in self.user_history_idxs['product']:
			brand_idxs = []
			category_idxs = []
			also_bought_idxs = []
			also_viewed_idxs = []
			bought_together_idxs = []

			for product_idx in product_idxs:
				brand_idxs.extend(self.knowledge['brand']['data'][product_idx])
				category_idxs.extend(self.knowledge['categories']['data'][product_idx])
				also_bought_idxs.extend(self.knowledge['also_bought']['data'][product_idx])
				also_viewed_idxs.extend(self.knowledge['also_viewed']['data'][product_idx])
				bought_together_idxs.extend(self.knowledge['bought_together']['data'][product_idx])

			#get unique idxs
			brand_idxs = list(set(brand_idxs))
			category_idxs = list(set(category_idxs))
			also_bought_idxs = list(set(also_bought_idxs))
			also_viewed_idxs = list(set(also_viewed_idxs))
			bought_together_idxs = list(set(bought_together_idxs))

			self.history_length.append(len(product_idxs))
			self.user_history_idxs['brand'].append(brand_idxs)
			self.user_history_idxs['categories'].append(category_idxs)
			self.user_history_idxs['also_bought'].append(also_bought_idxs)
			self.user_history_idxs['also_viewed'].append(also_viewed_idxs)
			self.user_history_idxs['bought_together'].append(bought_together_idxs)

			#history length should be max of all histories
			max_history_length = max(len(product_idxs), len(brand_idxs), len(category_idxs), len(also_bought_idxs), len(also_viewed_idxs), len(bought_together_idxs))
			if self.max_history_length < max_history_length:
				self.max_history_length = max_history_length
		print('Max user history length %d' % self.max_history_length)
		print('Average user history length %.4f' % np.mean(self.history_length))


		#get review sets
		self.word_count = 0
		self.vocab_distribute = np.zeros(self.vocab_size) 
		self.review_info = []
		self.review_text = []
		with gzip.open(input_train_dir + set_name + '.txt.gz', 'rt') as fin:
			for line in fin:
				arr = line.strip().split('\t')
				self.review_info.append([int(arr[0]), int(arr[1])]) # (user_idx, product_idx)
				self.review_text.append([int(i) for i in arr[2].split(' ')])
				for idx in self.review_text[-1]:
					self.vocab_distribute[idx] += 1
				self.word_count += len(self.review_text[-1])
		self.review_size = len(self.review_info)
		self.vocab_distribute = self.vocab_distribute.tolist() 
		self.sub_sampling_rate = None
		self.review_distribute = np.ones(self.review_size).tolist()
		self.product_distribute = np.ones(self.product_size).tolist()

		with gzip.open(input_train_dir + set_name + '_id.txt.gz', 'rt') as fin:
			index = 0
			for line in fin:
				arr = line.strip().split('\t')
				self.review_info[index].append(self.review_id_to_idx[arr[2]]) # (user_idx, product_idx)
				index += 1

		#get product query sets
		self.product_query_idx = []
		with gzip.open(input_train_dir + set_name + '_query_idx.txt.gz', 'rt') as fin:
			for line in fin:
				arr = line.strip().split(' ')
				query_idx = []
				for idx in arr:
					if len(idx) < 1:
						continue
					query_idx.append(int(idx))
				self.product_query_idx.append(query_idx)



		print("Data statistic: vocab %d, review %d, user %d, product %d\n" % (self.vocab_size, 
					self.review_size, self.user_size, self.product_size))

	def sub_sampling(self, subsample_threshold):
		if subsample_threshold == 0.0:
			return
		self.sub_sampling_rate = [1.0 for _ in range(self.vocab_size)]
		threshold = sum(self.vocab_distribute) * subsample_threshold
		count_sub_sample = 0
		for i in range(self.vocab_size):
			#vocab_distribute[i] could be zero if the word does not appear in the training set
			self.sub_sampling_rate[i] = min((np.sqrt(float(self.vocab_distribute[i]) / threshold) + 1) * threshold / float(self.vocab_distribute[i]),
											1.0)
			count_sub_sample += 1

	def read_train_product_ids(self, data_path):
		self.user_train_product_set_list = [set() for i in range(self.user_size)]
		self.train_review_size = 0
		with gzip.open(data_path + 'train.txt.gz', 'rt') as fin:
			for line in fin:
				self.train_review_size += 1
				arr = line.strip().split('\t')
				self.user_train_product_set_list[int(arr[0])].add(int(arr[1]))

	def read_org_product_reviews(self, data_path):
		self.org_review_text = []
		with gzip.open(data_path + 'review_text.txt.gz', 'rt') as fin:
			for line in fin:
				self.org_review_text.append([int(i) for i in line.strip().split(' ')])

	def compute_test_product_ranklist(self, u_idx, original_scores, sorted_product_idxs, rank_cutoff):
		product_rank_list = []
		product_rank_scores = []
		rank = 0
		for product_idx in sorted_product_idxs:
			if product_idx in self.user_train_product_set_list[u_idx] or math.isnan(original_scores[product_idx]):
				continue
			product_rank_list.append(product_idx)
			product_rank_scores.append(original_scores[product_idx])
			rank += 1
			if rank == rank_cutoff:
				break
		return product_rank_list, product_rank_scores

	def output_ranklist(self, user_ranklist_map, user_ranklist_score_map, output_path, similarity_func):
		with open(output_path + 'test.'+similarity_func+'.ranklist', 'w') as rank_fout:
			for uq_pair in user_ranklist_map:
				user_id = self.user_ids[uq_pair[0]]
				rank = 1
				for i in range(len(user_ranklist_map[uq_pair])):
					if user_ranklist_map[uq_pair][i] >= len(self.product_ids):
						continue
					product_id = self.product_ids[user_ranklist_map[uq_pair][i]]
					rank_fout.write(user_id+'_'+str(uq_pair[1]) + ' Q0 ' + product_id + ' ' + str(rank)
							+ ' ' + str(user_ranklist_score_map[uq_pair][i]) + ' ProductSearchEmbedding\n')
					rank += 1


	def output_embedding(self, embeddings, output_file_name):
		with open(output_file_name,'w') as emb_fout:
			try:
				length = len(embeddings)
				if length < 1:
					return
				dimensions = len(embeddings[0])
				emb_fout.write(str(length) + '\n')
				emb_fout.write(str(dimensions) + '\n')
				for i in range(length):
					for j in range(dimensions):
						emb_fout.write(str(embeddings[i][j]) + ' ')
					emb_fout.write('\n')
			except:
				if isinstance(embeddings.tolist(), float):
					emb_fout.write(str(embeddings) + '\n')
				else:
					emb_fout.write(' '.join([str(x) for x in embeddings.tolist()]) + '\n')
				#else:
				#	emb_fout.write(str(embeddings) + '\n')

	def print_entity_list(self, relation_name, entity_name, entity_scores, rank_cut, remove_map):
		if entity_name not in self.entity_vocab:
			print('Cannot find entity %s' % entity_name)
		print('%s list: rank, id, name, score' % relation_name)
		sorted_entity_idxs = sorted(range(len(entity_scores)), 
									key=lambda k: entity_scores[k], reverse=True)
		entity_rank_list = []
		entity_rank_scores = []
		rank = 0
		for entity_idx in sorted_entity_idxs:
			if entity_name in remove_map and entity_idx in remove_map[entity_name]:
				continue
			if math.isnan(entity_scores[entity_idx]):
				continue
			entity_rank_list.append(entity_idx)
			entity_rank_scores.append(entity_scores[entity_idx])
			rank += 1
			if rank >= rank_cut:
				break
		# print results
		for i in range(len(entity_rank_list)):
			print('%d\t%d\t"%s"\t%.4f' % (i, entity_rank_list[i], 
				self.entity_vocab[entity_name][entity_rank_list[i]], entity_rank_scores[i]))
	
	def get_idx(self, input_str, entity_name):
		if entity_name not in self.entity_vocab:
			print('Cannot find entity %s' % entity_name)
			return None
		try:
			return self.entity_vocab[entity_name].index(input_str)
		except:
			return int(input_str)

	def get_user_history_idx(self, user_idx, max_history_length):
		if user_idx < len(self.user_ids):
			user_history_item_idx = self.user_history_idxs['product'][user_idx][-max_history_length:]
			user_history_brand_idx = self.user_history_idxs['brand'][user_idx][-max_history_length:]
			user_history_category_idx = self.user_history_idxs['categories'][user_idx][-max_history_length:]
			user_history_also_bought_idx = self.user_history_idxs['also_bought'][user_idx][-max_history_length:]
			user_history_also_viewed_idx = self.user_history_idxs['also_viewed'][user_idx][-max_history_length:]
			user_history_bought_together_idx = self.user_history_idxs['bought_together'][user_idx][-max_history_length:]
			return {
				'product': user_history_item_idx,
				'brand': user_history_brand_idx,
				'categories': user_history_category_idx,
				'also_bought': user_history_also_bought_idx,
				'also_viewed': user_history_also_viewed_idx,
				'bought_together': user_history_bought_together_idx
			}
		else:
			return {
				'product': [],
				'brand': [],
				'categories': [],
				'also_bought': [],
				'also_viewed': [],
				'bought_together': []
			}


	def get_entity(self, entity_type, index):
		if entity_type == 'product':
			entity_vocab = self.product_ids
		else:
			entity_vocab = self.knowledge[entity_type]['vocab']

		if index < len(entity_vocab):
			return entity_vocab[index]
		else:
			return ''

	def generate_explanation(self, relation_type, att_percentage, entity_type=None, entity_name=None):
		EXPLANATION_TMPL_DEFAULT = "For {percentage}%, this product was retrieved because of its popularity among people who searched for this query."
		EXPLANATION_TMPL_0 = "For {percentage}%, this product was retrieved because of the products previously purchased by the user, especially '<em>{entity_name}</em>'."
		EXPLANATION_TMPL_1 = "For {percentage}%, this product was retrieved because of the <em>{relation_type}</em> of products previously purchased by the user, especially '<em>{entity_name}</em>'."
		EXPLANATION_TMPL_2 = "For {percentage}%, this product was retrieved because of the {entity_type} <em>{relation_type}</em> with the products previously purchased by the user, especially '<em>{entity_name}</em>'."
		
		explanation = None
		if relation_type == 'popularity':
			explanation = EXPLANATION_TMPL_DEFAULT.format(percentage=att_percentage)
		elif relation_type == 'product':
			explanation = EXPLANATION_TMPL_0.format(
					percentage=att_percentage,
					entity_name=entity_name
				)
		elif relation_type == 'brand' or relation_type == 'categories':
			explanation = EXPLANATION_TMPL_1.format(
					percentage=att_percentage,
					relation_type=relation_type, 
					entity_name=entity_name
				)
		else: #'also_bought', 'also_viewed', 'bought_together'
			explanation = EXPLANATION_TMPL_2.format(
					percentage=att_percentage,
					entity_type=entity_type,
					relation_type=relation_type, 
					entity_name=entity_name
				)
		return explanation

	def get_expln_with_max_attn(self, index, attn_score, user_history_dict, user_history_idx_dict,
								attn_distribution_dict):
		explanation = None
		att_percentage = '%.2f' % (100*attn_score)
		# The attn score of zero vec is the last dimension in the attn vector
		# if zero vec has max attn
		if index == len(user_history_dict.keys()):
			explanation = self.generate_explanation('popularity', att_percentage)
		elif att_percentage >= 0.01:
			key = sorted(list(user_history_dict.keys()))[index]
			print('%d, %.2f, %s' % (index, attn_score, key))
			sub_attn_values = np.array(attn_distribution_dict[key])
			print('history content')
			print(user_history_idx_dict[key])
			print('attention')
			print(sub_attn_values)
			max_sub_index = sub_attn_values.argmax()
			hist_id = user_history_idx_dict[key][max_sub_index]
			entity_name = self.get_entity(key, hist_id)
			entity_type = user_history_dict[key]['entity_type']
			if entity_name:
				explanation = self.generate_explanation(
					key, 
					att_percentage,
					entity_type=entity_type, 
					entity_name=entity_name
				)
		return explanation
