# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements data process strategies.
"""

import os
import json
import logging
import numpy as np
from collections import Counter
import sys
reload(sys)
sys.setdefaultencoding('utf8')

class BRCDataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """
    def __init__(self, max_p_num, max_p_len, max_q_len,
                 train_files=[], dev_files=[], test_files=[], vocab = None):
        self.logger = logging.getLogger("brc")
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        self.vocab = vocab

        self.train_set, self.dev_set, self.test_set = [], [], []
        if train_files:
            for train_file in train_files:
                self.train_set += self._load_dataset(train_file, train=True)
            self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))

        if dev_files:
            for dev_file in dev_files:
                self.dev_set += self._load_dataset(dev_file, train = True)
            self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))

        if test_files:
            for test_file in test_files:
                self.test_set += self._load_dataset(test_file)
            self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))

    def list2string(self, lis):
        string = ''
        for l in lis:
            string += l
        return string

    def _load_dataset(self, data_path, train=False):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        with open(data_path) as fin:
            data_set = []
            for lidx, line in enumerate(fin):
                sample = json.loads(line.strip())
                # if train:
                #     #print sample['answer_spans']
                #     if len(sample['answer_spans']) == 0:
                #         continue
                #     if sample['answer_spans'][0][1] >= self.max_p_len:
                #         continue

                if 'answer_docs' in sample:
                    sample['answer_passages'] = sample['answer_docs']

                sample['question_tokens'] = sample['segmented_question']

                sample['passages'] = []

                for d_idx, doc in enumerate(sample['documents']):
                    #get sentences
                    punc = "！。，；;:！,？,｡！＃＄％＆＊＋，／：；＜＝＞＼＾＿｀｛｜｝=～〃〟‘’‛„…‧﹏"
                    punc_unicode = punc.decode("utf-8")

                    if train:
                        sen_list = []

                        #print '___________doc__________________ '
                        most_related_para = doc['most_related_para']
                        passage_token = []

                        for p_id, para in enumerate(doc['segmented_paragraphs'],0):
                            passage_token += para
                            start_idx = 0
                            # print self.list2string(para)
                            # print '++++++++++++++++++++++++++++s'
                            for idx, word in enumerate(para, 1):
                                if idx < len(para):
                                    if (word in punc_unicode) and not (para[idx] in punc_unicode):
                                        senten = para[start_idx:idx]
                                        #print self.list2string(senten)
                                        if not len(senten) > 100:
                                            sen_list.append(senten)
                                        start_idx = idx
                                else:
                                    if word in punc_unicode:
                                        senten = para[start_idx:idx]
                                        #print self.list2string(senten)
                                        if not len(senten) > 100:
                                            sen_list.append(senten)
                                        start_idx = idx
                            if not start_idx == len(para):
                                senten = para[start_idx:]
                                #print self.list2string(senten)
                                # over 50 word continue
                                if not len(senten) > 100:
                                    sen_list.append(senten)
                        sample['passages'].append(
                            {'passage_tokens': passage_token,
                             'sentence_tokens_list': sen_list,
                             'title_tokens': doc['segmented_title'],
                             'is_selected': doc['is_selected']}
                        )
                    else:
                        para = []
                        sen_list = []
                        for para_tokens in doc['segmented_paragraphs']:
                            para += para_tokens
                            start_idx = 0
                            for idx, word in enumerate(para_tokens, 1):
                                if idx < len(para_tokens):
                                    if (word in punc_unicode) and not (para_tokens[idx] in punc_unicode):
                                        senten = para_tokens[start_idx:idx]
                                        # print self.list2string(senten)
                                        if not len(senten) > 50:
                                            sen_list.append(senten)
                                        start_idx = idx
                                else:
                                    if word in punc_unicode:
                                        senten = para_tokens[start_idx:idx]
                                        # print self.list2string(senten)
                                        if not len(senten) > 50:
                                            sen_list.append(senten)
                                        start_idx = idx
                            if not start_idx == len(para_tokens):
                                senten = para_tokens[start_idx:]
                                # print self.list2string(senten)
                                if not len(senten) > 50:
                                    sen_list.append(senten)
                        sample['passages'].append({'passage_tokens': para,
                                                   'sentence_tokens_list': sen_list,
                                                   'title_tokens': doc['segmented_title'],
                                                   })
                data_set.append(sample)
        return data_set

    def _one_mini_batch(self, data, indices, pad_id):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                      'question_ids':[],
                      'question_token_ids': [],
                      'question_length': [],
                      'padded_p_len':0,
                      'passage_token_ids': [],
                      'passage_length': [],
                      'start_id': [],
                      'end_id': []}
        max_passage_num = max([len(sample['passages']) for sample in batch_data['raw_data']])
        max_passage_num = min(self.max_p_num, max_passage_num)
        for sidx, sample in enumerate(batch_data['raw_data']):
            for pidx in range(max_passage_num):
                if pidx < len(sample['passages']):
                    #print pidx
                    #print 'hahahhahah'+ str(sample['question_id'])
                    batch_data['question_ids'].append(sample['question_id'])
                    batch_data['question_token_ids'].append(sample['question_token_ids'])
                    #print 'question_token_ids: ' + str(sample['question_token_ids'])
                    batch_data['question_length'].append(len(sample['question_token_ids']))
                    #print 'question_token_ids_len: ' + str(len(sample['question_token_ids']))
                    passage_token_ids = sample['passages'][pidx]['passage_token_ids']
                    #print 'passage_token_ids' + str(passage_token_ids)
                    #print 'len(passage_token_ids)' + str(len(passage_token_ids))
                    #print 'max_p_len' + str(self.max_p_len)
                    batch_data['passage_token_ids'].append(passage_token_ids)
                    batch_data['passage_length'].append(min(len(passage_token_ids), self.max_p_len))
                else:
                    batch_data['question_token_ids'].append([])
                    batch_data['question_length'].append(0)
                    batch_data['passage_token_ids'].append([])
                    batch_data['passage_length'].append(0)
        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data, pad_id)
        batch_data['padded_p_len'] = padded_p_len
        for sample in batch_data['raw_data']:
            if 'answer_passages' in sample and len(sample['answer_passages']):
                #print 'answer_passages'
                #print sample['answer_passages']
                #print sample['answer_passages'][0]
                #print padded_p_len
                #gold_passage_offset = 0
                #for p in xrange(0,sample['answer_passages'][0]):
                    #gold_passage_offset +=batch_data['passage_length'][p]
                gold_passage_offset = padded_p_len * sample['answer_passages'][0]
                #print 'answer_passages: '+ str(sample['answer_passages'])
                batch_data['start_id'].append(gold_passage_offset + sample['answer_spans'][0][0])
                batch_data['end_id'].append(gold_passage_offset + sample['answer_spans'][0][1])
                #print 'gold_passage_offset: ' + str(gold_passage_offset)
                #print 'start id: ' + str(batch_data['start_id'])
                #print 'end id: ' + str(batch_data['end_id'])
            else:
                # fake span for some samples, only valid for testing
                batch_data['start_id'].append(0)
                batch_data['end_id'].append(0)
        return batch_data

    def _dynamic_padding(self, batch_data, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        #print 'dynamic _padding...'
        #print 'pad_id' + str(pad_id)
        pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))+1
        #print 'pad_p_len' + str(pad_p_len)
        pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        #print 'pad_q_len' + str(pad_q_len)
        #for ids in batch_data['passage_token_ids'] :
            #print 'padding: '
            #print (ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_token_ids']]
        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids']]
        return batch_data, pad_p_len, pad_q_len

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['question_tokens']:
                    yield token
                for passage in sample['passages']:
                    for token in passage['passage_tokens']:
                        yield token
                    for token in passage['title_tokens']:
                        yield token

    def convert_to_ids(self, vocab):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['question_token_ids'] = vocab.convert_to_ids(sample['question_tokens'])
                for passage in sample['passages']:
                    passage['title_tokens_ids'] = vocab.convert_to_ids(passage['title_tokens'])
                    passage['passage_token_ids'] = vocab.convert_to_ids(passage['passage_tokens'])
                    passage['sentence_tokens_id_list'] = []
                    for sen in passage['sentence_tokens_list']:
                        id_list = vocab.convert_to_ids(sen)
                        passage['sentence_tokens_id_list'].append(id_list)

    def gen_mini_batches(self, set_name, batch_size, pad_id, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices, pad_id)

    def gen_batches(self, set_name, batch_size, pad_id, shuffle = True ):
        """
            Generate data batches for a specific dataset (train/dev/test)
            Args:
                set_name: train/dev/test to indicate the set
                batch_size: number of samples in one batch
                pad_id: pad id
                shuffle: if set to be true, the data is shuffled.
            Returns:
                a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            if set_name == 'test':
                yield self._get_test_mini_batch(data, batch_indices, pad_id)
            else:
                yield self._get_mini_batch(data, batch_indices, pad_id)

    def _get_mini_batch(self, data, indices, pad_id):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                      'question_ids':[],
                      'question_types':[],
                      'question_token_ids': [],
                      'question_length': [],
                      'passage_token_ids_list':[],
                      'passage_title_token_ids_list':[],
                      'passage_sentence_token_ids_list': [],
                      'passage_length_list': [],
                      'passage_title_length_list': [],
                      'passage_sen_length_list': [],
                      'passage_is_selected_list': [],
                      'passage_sen_num': [],
                      'padded_p_len':[],
                      'fake_answers':[],
                      'test_answers': [],
                      'segmented_answers':[],
                      'ref_answers':[],
                      'ref_answers_length':[]
                      }
        for sidx, sample in enumerate(batch_data['raw_data']):
            batch_data['question_ids'].append(sample['question_id'])
            batch_data['question_types'].append((sample['question_type']))
            batch_data['question_token_ids'].append(sample['question_token_ids'])
            batch_data['question_length'].append(min(len(sample['question_token_ids']), self.max_p_len))
            batch_data['ref_answers'].append(sample['answers'])
            batch_data['ref_answers_length'].append(len(sample['answers']))
            batch_data['segmented_answers'].append(sample['segmented_answers'])
            batch_data['fake_answers'].append(sample['fake_answers'])
            #print '-------------------------------------------'
            passage_token_ids_list, passage_length_list= [], []
            passage_title_token_ids_list, passage_title_length_list = [], []
            passage_sentence_token_ids_list, passage_sen_length_list, padded_p_len_list = [], [], []
            passage_is_selected_list = []

            for d_idx, passage in enumerate(sample['passages']):
                #if passage['is_selected'] == True:
                    #print 'passage ' + str(d_idx) + 'is selected'
                passage_is_selected_list.append(passage['is_selected'])

                passage_token_ids_list.append(passage['passage_token_ids'])
                passage_length_list.append(len(passage['passage_token_ids']))
                passage_title_token_ids_list.append(passage['title_tokens_ids'])
                passage_title_length_list.append(len(passage['title_tokens_ids']))

                tmp_sen_list, tmp_sen_lth_list = [], []
                for i, sentence in enumerate(passage['sentence_tokens_id_list'], 0):
                    #print self.list2string( passage['sentence_tokens_list'][i])
                    tmp_sen_list.append(sentence)
                    # print sentence
                    # print self.list2string(self.vocab.recover_from_ids(sentence))
                    # print ('len of sentence ', len(sentence))
                    tmp_sen_lth_list.append(len(sentence))
                    tmp_sen_list_padded, padded_p_len, tmp_sen_lth_list_padded = self._dynamic_padding_new(
                        tmp_sen_list, tmp_sen_lth_list, pad_id)
                    passage_sentence_token_ids_list.append(tmp_sen_list_padded)
                    passage_sen_length_list.append(tmp_sen_lth_list_padded)
                    padded_p_len_list.append(padded_p_len)

                #print len(passage['passage_token_ids'])
            #print ('len of passage_token_ids', len(passage_token_ids))
            # print ('question_length', len(sample['question_tokens']))
            # print 'question: '
            #print self.list2string(sample['question_tokens'])
            batch_data['padded_p_len'].append(padded_p_len_list)

            batch_data['passage_token_ids_list'].append(passage_token_ids_list)
            batch_data['passage_length_list'].append(passage_length_list)

            batch_data['passage_title_token_ids_list'].append(passage_title_token_ids_list)
            batch_data['passage_title_length_list'].append(passage_title_length_list)

            batch_data['passage_sentence_token_ids_list'].append(passage_sentence_token_ids_list)
            batch_data['passage_sen_num'].append(len(passage_sentence_token_ids_list))
            batch_data['passage_sen_length_list'].append(passage_sen_length_list)
            batch_data['passage_is_selected_list'].append(passage_is_selected_list)


        # batch_size = 0, no padding


        return batch_data

    def _get_test_mini_batch(self, data, indices, pad_id):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                      'question_ids': [],
                      'question_types': [],
                      'question_token_ids': [],
                      'question_length': [],
                      'passage_token_ids_list': [],
                      'passage_title_token_ids_list': [],
                      'passage_sentence_token_ids_list': [],
                      'passage_length_list': [],
                      'passage_title_length_list': [],
                      'passage_sen_length_list': [],
                      'passage_sen_num': [],
                      'passage_is_selected_list': [],
                      'padded_p_len': [],
                      }
        for sidx, sample in enumerate(batch_data['raw_data']):
            batch_data['question_ids'].append(sample['question_id'])
            batch_data['question_types'].append((sample['question_type']))
            batch_data['question_token_ids'].append(sample['question_token_ids'])
            batch_data['question_length'].append(min(len(sample['question_token_ids']), self.max_p_len))

            print '-------------------------------------------'
            passage_token_ids_list, passage_length_list= [], []
            passage_title_token_ids_list, passage_title_length_list = [], []
            passage_sentence_token_ids_list, passage_sen_length_list, padded_p_len_list = [], [], []

            for d_idx, passage in enumerate(sample['passages']):
                # if passage['is_selected'] == True:
                # print 'passage ' + str(d_idx) + 'is selected'
                passage_token_ids_list.append(passage['passage_token_ids'])
                passage_length_list.append(len(passage['passage_token_ids']))
                passage_title_token_ids_list.append(passage['title_tokens_ids'])
                passage_title_length_list.append(len(passage['title_tokens_ids']))
                tmp_sen_list, tmp_sen_lth_list = [], []
                for i, sentence in enumerate(passage['sentence_tokens_id_list'], 0):
                    # print self.list2string( passage['sentence_tokens_list'][i])
                    tmp_sen_list.append(sentence)
                    # print sentence
                    #print self.list2string(self.vocab.recover_from_ids(sentence))
                    # print ('len of sentence ', len(sentence))
                    tmp_sen_lth_list.append(len(sentence))
                    tmp_sen_list_padded, padded_p_len, tmp_sen_lth_list_padded = self._dynamic_padding_new(
                        tmp_sen_list, tmp_sen_lth_list, pad_id)
                    passage_sentence_token_ids_list.append(tmp_sen_list_padded)
                    passage_sen_length_list.append(tmp_sen_lth_list_padded)
                    padded_p_len_list.append(padded_p_len)

            batch_data['padded_p_len'].append(padded_p_len_list)
            batch_data['passage_token_ids_list'].append(passage_token_ids_list)
            batch_data['passage_length_list'].append(passage_length_list)

            batch_data['passage_title_token_ids_list'].append(passage_title_token_ids_list)
            batch_data['passage_title_length_list'].append(passage_title_length_list)

            batch_data['passage_sentence_token_ids_list'].append(passage_sentence_token_ids_list)
            batch_data['passage_sen_num'].append(len(passage_sentence_token_ids_list))
            batch_data['passage_sen_length_list'].append(passage_sen_length_list)

        # batch_size = 0, no padding


        return batch_data

    def _dynamic_padding_new(self, passage_token_ids, passage_sen_length, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        #print 'dynamic _padding...'
        #print 'pad_id' + str(pad_id)
        for leng in passage_sen_length:
            if(leng > self.max_p_len ):
                leng = self.max_p_len
        pad_p_len = min(self.max_p_len, max(passage_sen_length))
            #print (ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
        passage_token_ids_padded = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in passage_token_ids]
        return passage_token_ids_padded, pad_p_len, passage_sen_length