"""Training and testing the hierarchical embedding model for personalized product search

See the following papers for more information on the hierarchical embedding model.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import sys
import time, copy
import csv
import operator

from six.moves import xrange	# pylint: disable=redefined-builtin
import tensorflow as tf
import data_util
import numpy as np

from ProductSearchEmbedding import ProductSearchEmbedding_model


tf.app.flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.90,
							"Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
							"Clip gradients to this norm.")
tf.app.flags.DEFINE_float("subsampling_rate", 1e-4,
							"The rate to subsampling.")
tf.app.flags.DEFINE_float("L2_lambda", 0.0,
							"The lambda for L2 regularization.")
tf.app.flags.DEFINE_float("dynamic_weight", 0.5,
							"The weight for the dynamic relationship [0.0, 1.0].")
tf.app.flags.DEFINE_float("query_weight", 0.5,
							"The weight for query.")
tf.app.flags.DEFINE_integer("batch_size", 64,
							"Batch size to use during training.")
#rank list size should be read from data
tf.app.flags.DEFINE_string("data_dir", "./tmp_data/", "Data directory")
tf.app.flags.DEFINE_string("input_train_dir", "", "The directory of training and testing data")
tf.app.flags.DEFINE_string("train_dir", "./tmp/", "Model directory & output directory")
tf.app.flags.DEFINE_string("similarity_func", "bias_product", "Select similarity function, which could be product, cosine and bias_product")
tf.app.flags.DEFINE_string("net_struct", "simplified_fs", "Specify network structure parameters. Please read readme.txt for details.")
tf.app.flags.DEFINE_integer("embed_size", 100, "Size of each embedding.")
tf.app.flags.DEFINE_integer("window_size", 5, "Size of context window.")
tf.app.flags.DEFINE_integer("max_train_epoch", 5,
							"Limit on the epochs of training (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
							"How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("seconds_per_checkpoint", 3600,
							"How many seconds to wait before storing embeddings.")
tf.app.flags.DEFINE_integer("negative_sample", 5,
							"How many samples to generate for negative sampling.")
tf.app.flags.DEFINE_boolean("decode", False,
							"Set to True for testing.")
tf.app.flags.DEFINE_string("test_mode", "product_scores", "Test modes: product_scores -> output ranking results and ranking scores; output_embedding -> output embedding representations for users, items and words. (default is product_scores)")
tf.app.flags.DEFINE_integer("rank_cutoff", 100,
							"Rank cutoff for output ranklists.")
tf.app.flags.DEFINE_string("explanation_output_file", "./utils/explanation-output.csv",
							"Output CSV file where generated explanations will be written")
tf.app.flags.DEFINE_integer("max_history_length", 20,
							"How many history purchases used for user profiling.")


FLAGS = tf.app.flags.FLAGS

EXPLANATION_TMPL = "This product was retrieved for this query because the user often buys products related to {entity_type} '{entity_name}' which is also related to the query"


def create_model(session, forward_only, data_set, review_size):
	"""Create translation model and initialize or load parameters in session."""
	model = ProductSearchEmbedding_model(
			data_set,
			FLAGS.window_size, FLAGS.embed_size, FLAGS.max_gradient_norm, FLAGS.batch_size,
			FLAGS.learning_rate, FLAGS.L2_lambda, FLAGS.dynamic_weight, FLAGS.query_weight, 
			FLAGS.net_struct, FLAGS.similarity_func, FLAGS.max_history_length, forward_only, FLAGS.negative_sample)
	ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
	if ckpt:
		print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		print("Created model with fresh parameters.")
		session.run(tf.global_variables_initializer())
	return model


def train():
	# Prepare data.
	print("Reading data in %s" % FLAGS.data_dir)
	
	data_set = data_util.Tensorflow_data(FLAGS.data_dir, FLAGS.input_train_dir, 'train')
	data_set.sub_sampling(FLAGS.subsampling_rate)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	# config.log_device_placement=True

	with tf.Session(config=config) as sess:
		# Create model.
		print("Creating model")
		model = create_model(sess, False, data_set, data_set.review_size)

		print('Start training')
		words_to_train = float(FLAGS.max_train_epoch * data_set.word_count) + 1
		current_words = 0.0
		previous_words = 0.0
		start_time = time.time()
		last_check_point_time = time.time()
		step_time, loss = 0.0, 0.0
		current_epoch = 0
		current_step = 0
		get_batch_time = 0.0
		training_seq = [i for i in xrange(data_set.review_size)]
		model.setup_data_set(data_set, words_to_train)
		while True:
			random.shuffle(training_seq)
			model.intialize_epoch(training_seq)
			has_next = True
			while has_next:
				time_flag = time.time()
				input_feed, has_next = model.get_train_batch()
				get_batch_time += time.time() - time_flag

				if len(input_feed[model.relation_dict['word']['idxs'].name]) > 0:
					time_flag = time.time()
					step_loss, _ = model.step(sess, input_feed, False)
					#step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
					loss += step_loss / FLAGS.steps_per_checkpoint
					current_step += 1
					step_time += time.time() - time_flag

				# Once in a while, we print statistics.
				if current_step % FLAGS.steps_per_checkpoint == 0:
					print("Epoch %d Words %d/%d: lr = %5.3f loss = %6.2f words/sec = %5.2f prepare_time %.2f step_time %.2f\r" %
            				(current_epoch, model.finished_word_num, model.words_to_train, input_feed[model.learning_rate.name], loss, 
            					(model.finished_word_num- previous_words)/(time.time() - start_time), get_batch_time, step_time), end="")
					step_time, loss = 0.0, 0.0
					current_step = 1
					get_batch_time = 0.0
					sys.stdout.flush()
					previous_words = model.finished_word_num
					start_time = time.time()
					#print('time: ' + str(time.time() - last_check_point_time))
					#if time.time() - last_check_point_time > FLAGS.seconds_per_checkpoint:
					#	checkpoint_path_best = os.path.join(FLAGS.train_dir, "ProductSearchEmbedding.ckpt")
					#	model.saver.save(sess, checkpoint_path_best, global_step=model.global_step)

			current_epoch += 1
			#checkpoint_path_best = os.path.join(FLAGS.train_dir, "ProductSearchEmbedding.ckpt")
			#model.saver.save(sess, checkpoint_path_best, global_step=model.global_step)
			if current_epoch >= FLAGS.max_train_epoch:	
				break
		checkpoint_path_best = os.path.join(FLAGS.train_dir, "ProductSearchEmbedding.ckpt")
		model.saver.save(sess, checkpoint_path_best, global_step=model.global_step)


def get_product_scores():
	# Prepare data.
	print("Reading data in %s" % FLAGS.data_dir)
	
	data_set = data_util.Tensorflow_data(FLAGS.data_dir, FLAGS.input_train_dir, 'test')
	data_set.read_train_product_ids(FLAGS.input_train_dir)
	current_step = 0
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		# Create model.
		print("Read model")
		model = create_model(sess, True, data_set, data_set.train_review_size)
		user_ranklist_map = {}
		user_ranklist_score_map = {}
		print('Start Testing')
		words_to_train = float(FLAGS.max_train_epoch * data_set.word_count) + 1
		test_seq = [i for i in xrange(data_set.review_size)]
		model.setup_data_set(data_set, words_to_train)
		model.intialize_epoch(test_seq)
		model.prepare_test_epoch()
		has_next = True
		while has_next:
			input_feed, has_next, uqr_pairs = model.get_test_batch()

			if len(uqr_pairs) > 0:
				user_product_scores, _ = model.step(sess, input_feed, True)
				current_step += 1

			# record the results
			for i in xrange(len(uqr_pairs)):
				u_idx, p_idx, q_idx, r_idx = uqr_pairs[i]
				sorted_product_idxs = sorted(range(len(user_product_scores[i])), 
									key=lambda k: user_product_scores[i][k], reverse=True)
				user_ranklist_map[(u_idx, q_idx)],user_ranklist_score_map[(u_idx, q_idx)] = data_set.compute_test_product_ranklist(u_idx,
												user_product_scores[i], sorted_product_idxs, FLAGS.rank_cutoff) #(product name, rank)
			if current_step % FLAGS.steps_per_checkpoint == 0:
				print("Finish test review %d/%d\r" %
            			(model.cur_uqr_i, len(model.test_seq)), end="")

	data_set.output_ranklist(user_ranklist_map, user_ranklist_score_map, FLAGS.train_dir, FLAGS.similarity_func)
	return

def output_embedding():
	# Prepare data.
	print("Reading data in %s" % FLAGS.data_dir)
	
	data_set = data_util.Tensorflow_data(FLAGS.data_dir, FLAGS.input_train_dir, 'test')
	data_set.read_train_product_ids(FLAGS.input_train_dir)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		# Create model.
		print("Read model")
		model = create_model(sess, True, data_set, data_set.train_review_size)
		user_ranklist_map = {}
		print('Start Testing')
		words_to_train = float(FLAGS.max_train_epoch * data_set.word_count) + 1
		test_seq = [i for i in xrange(data_set.review_size)]
		model.setup_data_set(data_set, words_to_train)
		model.intialize_epoch(test_seq)
		model.prepare_test_epoch()
		has_next = True
		input_feed, has_next, uqr_pairs = model.get_test_batch()

		if len(uqr_pairs) > 0:
			embeddings , keys = model.step(sess, input_feed, True, FLAGS.test_mode)

			# record the results
			for i in xrange(len(keys)):
				data_set.output_embedding(embeddings[i], FLAGS.train_dir + '%s.txt' % keys[i])
			
	return

def interactive_explain_mode():
	# Prepare data.
	print("Reading data in %s" % FLAGS.data_dir)
	FLAGS.batch_size = 1
	
	data_set = data_util.Tensorflow_data(FLAGS.data_dir, FLAGS.input_train_dir, 'test')
	data_set.read_train_product_ids(FLAGS.input_train_dir)
	current_step = 0
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		# Create model.
		print("Read model")
		model = create_model(sess, True, data_set, data_set.train_review_size)
		user_ranklist_map = {}
		user_ranklist_score_map = {}
		print('Start Interactive Process')
		words_to_train = float(FLAGS.max_train_epoch * data_set.word_count) + 1
		test_seq = [i for i in xrange(data_set.review_size)]
		model.setup_data_set(data_set, words_to_train)
		model.intialize_epoch(test_seq)
		model.prepare_test_epoch()
		has_next = True
		input_feed, has_next, uqr_pairs = model.get_test_batch()
		while True:
			# read information from stdin
			mode, user_idx, query_idx, product_idx = None, None, None, None
			test_feed = copy.deepcopy(input_feed)
			print('Enter rank cut:')
			rank_cut = int(sys.stdin.readline().strip())
			print('Enter mode:')
			mode = sys.stdin.readline().strip()
			# Output user+query or product?
			if mode == 'product': # product
				print('Enter product idx or name:')
				product_idx = data_set.get_idx(sys.stdin.readline().strip(), 'product')
				test_feed[model.product_idxs.name] = [product_idx]
				p_entity_list, _ = model.step(sess, test_feed, True, 'explain_product')
				# output results
				print('Product %d %s' % (product_idx, data_set.product_ids[product_idx]))
				for relation_name, entity_name, entity_scores in p_entity_list:
					data_set.print_entity_list(relation_name, entity_name, entity_scores[0], rank_cut, {})
			else: # user + query
				print('Enter user idx or name:')
				user_idx = data_set.get_idx(sys.stdin.readline().strip(), 'user')
				user_history_idx_dict =  data_set.get_user_history_idx(user_idx, model.max_history_length)
				print('Enter query idx:')
				query_idx = int(sys.stdin.readline().strip())
				query_word_idx = model.data_set.query_words[query_idx]

				for key in user_history_idx_dict:
					test_feed[model.user_history_dict[key]['idxs'].name] = user_history_idx_dict[key]

				test_feed[model.query_word_idxs.name] = [query_word_idx]
				uq_entity_list, _ = model.step(sess, test_feed, True, 'explain_user_query')
				remove_map = {
					'product' : data_set.user_train_product_set_list[user_idx]
				}
				print('User %d %s' % (user_idx, data_set.user_ids[user_idx]))
				print('Query %d %s' % (query_idx, '_'.join([data_set.words[x] for x in query_word_idx if x < len(data_set.words)])))
				# output results
				for relation_name, entity_name, entity_scores in uq_entity_list:
					data_set.print_entity_list(relation_name, entity_name, entity_scores[0], rank_cut, remove_map)

	return


def find_explanation_path():
	print("Reading data in %s" % FLAGS.data_dir)

	data_set = data_util.Tensorflow_data(FLAGS.data_dir, FLAGS.input_train_dir, 'test')
	data_set.read_train_product_ids(FLAGS.input_train_dir)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		# Create model.
		print("Read model")
		model = create_model(sess, True, data_set, data_set.train_review_size)
		print('Start Interactive Process')
		words_to_train = float(FLAGS.max_train_epoch * data_set.word_count) + 1
		test_seq = [i for i in xrange(data_set.review_size)]
		model.setup_data_set(data_set, words_to_train)
		model.intialize_epoch(test_seq)
		model.prepare_test_epoch()
		input_feed, has_next, uqr_pairs = model.get_test_batch()

		test_feed = copy.deepcopy(input_feed)


		with open(FLAGS.explanation_output_file, mode='w') as write_csv_file:
			csv_writer = csv.writer(write_csv_file, delimiter=',')
			csv_writer.writerow(['user','query','product','explanation', 'previous_reviews'])

			for (user_idx, product_idx, query_idx, review_idx) in uqr_pairs:
				user_history_idx_dict, user_hist_len_dict =  model.get_history_and_length_dicts(review_idx)
				for key in user_history_idx_dict:
					test_feed[model.user_history_dict[key]['idxs'].name] = [user_history_idx_dict[key]]
					test_feed[model.user_history_dict[key]['length'].name] = [user_hist_len_dict[key]]

				test_feed[model.product_idxs.name] = [product_idx]
				query_word_idx = model.data_set.query_words[query_idx]
				test_feed[model.query_word_idxs.name] = [query_word_idx]

				attn_distribution_dict, _ = model.step(sess, test_feed, True, 'explanation_path')
				user = data_set.user_ids[user_idx]
				product = data_set.product_ids[product_idx]
				query = ' '.join([data_set.words[x] for x in query_word_idx if x < len(data_set.words)])
				review_idxs = [idx for idx, review in enumerate(data_set.review_info) if review[0] == user_idx]
				review_word_idxs = [data_set.review_text[idx] for idx in review_idxs]
				reviews = []
				for idx, review_word_idx in enumerate(review_word_idxs):
					review_txt = ' '.join([data_set.words[idx] for idx in review_word_idx if idx < len(data_set.words)])
					reviews.append(str(idx+1) + ') ' + review_txt)

				print('User %d %s' % (user_idx, user))
				print('Product %d %s' % (product_idx, product))
				attn_values = np.array(attn_distribution_dict['master'])
				max_index = attn_values.argmax()
				#if zero vec has max attn
				if max_index < len(model.user_history_dict.keys()):
					key = sorted(list(model.user_history_dict.keys()))[max_index]
					sub_attn_values = np.array(attn_distribution_dict[key])
					max_sub_index = sub_attn_values.argmax()
					hist_id = user_history_idx_dict[key][max_sub_index]
					entity_name = data_set.get_entity(key, hist_id)
					entity_type = model.user_history_dict[key]['entity_type']
					if entity_name:
						explanation = EXPLANATION_TMPL.format(entity_type=entity_type, entity_name=entity_name)
						csv_writer.writerow([user, query, product, explanation, '\n'.join(reviews)])


def main(_):
	if FLAGS.input_train_dir == "":
		FLAGS.input_train_dir = FLAGS.data_dir

	if FLAGS.decode:
		if FLAGS.test_mode == 'output_embedding':
			output_embedding()
		elif 'explain' in FLAGS.test_mode:
			interactive_explain_mode()
		elif FLAGS.test_mode == 'explanation_path':
			find_explanation_path()
		else:
			get_product_scores()
	else:
		train()

if __name__ == "__main__":
	tf.app.run()
