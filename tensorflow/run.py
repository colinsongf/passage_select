# -*- coding:utf8 -*-
"""
This module prepares and runs the whole system.

python run.py --prepare --train_files ../data/preprocessed/trainset/search/search.train.json --dev_files  ../data/preprocessed/devset/search.dev.json --test_files ../data/seg/search/test/search.test.json 

python run.py --prepare --train_files ../data/demo/trainset/search.train.json --dev_files  ../data/demo/devset/search.dev.json --test_files ../data/demo/testset/search.test.json

nohup python -u run.py --train --algo BIDAF --epochs 300  --gpu 0 --train_files ../data/demo/trainset/search.train.json --dev_files  ../data/demo/devset/search.dev.json --test_files ../data/demo/test/search.test.json >BIDAF_log.txt 2>&1 &

nohup python -u run.py --train --algo MLSTM --epochs 3000  --train_files ../data/preprocessed/trainset/search/search.train.json --dev_files  ../data/preprocessed/devset/search.dev.json --test_files ../data/seg/search/test/search.test.json >MLSTM_log.txt 2>&1 &

########test MCTS ##

python run.py --prepare --train_files ../data/preprocessed/trainset/test_search.train.json --dev_files  ../data/preprocessed/devset/search.dev.json --test_files ../data/seg/search.test.json

nohup python -u run.py --train --algo MCST --epochs 1  --train_files ../data/preprocessed/trainset/search/test.search.train.json --dev_files  ../data/preprocessed/devset/search.dev.json --test_files ../data/seg/search/test/search.test.json >test_log.txt 2>&1 &
python -u run.py --train --algo MCST --epochs 2  --hidden_size 150 --batch_size 1 --train_files ../data/demo/trainset/test --dev_files  ../data/demo/devset/search.dev.json --test_files ../data/demo/testset/search.test.json
python -u run.py --prepare --train_files ../data/demo/trainset/test --dev_files  ../data/demo/devset/search.dev.json --test_files ../data/demo/testset/search.test.json


shiyan 1
python -u run.py --prepare --train_files ../data/demo/trainset/search.train.json --dev_files  ../data/demo/devset/search.dev.json --test_files ../data/demo/testset/search.test.json

nohup python -u run.py --train --algo MCST --epochs 20  --hidden_size 150 --batch_size 1 --gpu 0 --train_files ../data/demo/trainset/search.train.json --dev_files  ../data/demo/devset/search.dev.json --test_files ../data/demo/testset/search.test.json >exp_20_3000_10_log.txt 2>&1 &

nohup python -u run.py --train --algo MCST --epochs 30  --hidden_size 150 --batch_size 1 --gpu 1 --train_files ../data/demo/trainset/search.train.json --dev_files  ../data/demo/devset/search.dev.json --test_files ../data/demo/testset/search.test.json >exp_30_3000_20_log.txt 2>&1 &

nohup python -u run.py --train --algo MCST --epochs 50  --hidden_size 150 --batch_size 1 --gpu 2 --train_files ../data/demo/trainset/search.train.json --dev_files  ../data/demo/devset/search.dev.json --test_files ../data/demo/testset/search.test.json >exp_50_500_10_log.txt 2>&1 &

nohup python -u run.py --train --algo MCST --epochs 20  --hidden_size 150 --batch_size 1 --gpu 3 --train_files ../data/demo/trainset/search.train.json --dev_files  ../data/demo/devset/search.dev.json --test_files ../data/demo/testset/search.test.json >exp_20_100_20_log.txt 2>&1 &



base line :
# demo :

python run.py --prepare  

nohup python -u run.py --train --algo BIDAF --epochs 30  >BIDAF_demo_log.txt 2>&1 &

nohup python -u run.py --train --algo MLSTM --epochs 30  >MLSTM_demo_log.txt 2>&1 &

python run.py --evaluate --algo BIDAF

python run.py --predict --algo BIDAF --test_files ../data/demo/devset/search.dev.json 

#raw data
python run.py --prepare --train_files ../data/preprocessed/trainset/search.train.json --dev_files  ../data/preprocessed/devset/search.dev.json --test_files ../data/preprocessed/testset/search.test.json

python run.py --prepare --train_files /home/xujun/lizhaohui/DuReader/data/preprocessed/trainset/search/search.train.json --dev_files  /home/xujun/lizhaohui/DuReader/data/preprocessed/devset/search.dev.json
nohup python -u run.py --train --algo BIDAF --epochs 100 --gpu 0 ../data/preprocessed/trainset/search.train.json --dev_files  ../data/preprocessed/devset/search.dev.json --test_files ../data/preprocessed/testset/search.test.json >BIDAF_log.txt 2>&1 &
nohup python -u run.py --train --algo MLSTM --epochs 100 --gpu 0 ../data/preprocessed/trainset/search.train.json --dev_files  ../data/preprocessed/devset/search.dev.json --test_files ../data/preprocessed/testset/search.test.json >MLSTM_log.txt 2>&1 &

nohup python -u run.py --train --algo BIDAF --epochs 30 --train_files /home/xujun/lizhaohui/DuReader/data/preprocessed/trainset/search/parta1 --dev_files  /home/xujun/lizhaohui/DuReader/data/preprocessed/devset/dev_part500  >all_BIDAF_log.txt 2>&1 &

"""

import sys
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.append('..')
sys.setrecursionlimit(100000)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import argparse
import logging
from dataset import BRCDataset
from vocab import Vocab
from mcst_model import MCSTmodel
from rc_model import RCModel


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Reading Comprehension on BaiduRC dataset')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='sgd',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.0001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=1,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=32,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10,
                                help='train epochs')
    train_settings.add_argument('--min_cnt', type=int, default=1,
                                help='size of the embeddings')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['BIDAF', 'MLSTM','MCST'], default='MCST',
                                help='choose the algorithm to use')

    model_settings.add_argument('--embed_size', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=150,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_p_num', type=int, default=5,
                                help='max passage num in one sample')
    model_settings.add_argument('--max_p_len', type=int, default=1500,
                                help='max length of passage')
    model_settings.add_argument('--max_q_len', type=int, default=60,
                                help='max length of question')
    model_settings.add_argument('--max_a_len', type=int, default=20,
                                help='max length of answer')
    model_settings.add_argument('--search_time', type=int, default=3000,
                                help='search time of mcts')
    model_settings.add_argument('--beta', type=float, default=100,
                                help=' parameter that balances the loss part')
    model_settings.add_argument('--beta1', type=float, default=1,
                                help=' parameter that balances loss part')
    model_settings.add_argument('--beta2', type=float, default=1,
                                help=' parameter that balances loss part')
    model_settings.add_argument('--Bleu4', type=int, default=1,
                                help='weight of Bleu-4')
    model_settings.add_argument('--RougeL', type=int, default=1,
                                help='weigh of Rouge-L')
    model_settings.add_argument('--Bleu1', type=int, default=1,
                                help='weight of Bleu-1')
    model_settings.add_argument('--Bleu2', type=int, default=1,
                                help='weight of Bleu-2')
    model_settings.add_argument('--Bleu3', type=int, default=1,
                                help='weight of Bleu-3')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--emb_files', nargs='+',
                               default=['../data/vectors.txt'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--train_files', nargs='+',
                               default=['../data/demo/trainset/search.train.json'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['../data/demo/devset/search.dev.json'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['../data/demo/testset/search.test.json'],
                               help='list of files that contain the preprocessed test data')
    path_settings.add_argument('--brc_dir', default='../data/baidu',
                               help='the dir with preprocessed baidu reading comprehension data')
    path_settings.add_argument('--vocab_dir', default='../data/vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='../data/models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='../data/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='../data/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    path_settings.add_argument('--draw_path', default='.log/',
                               help='tensorboard')
    return parser.parse_args()


def prepare(args):
    """
    checks data, creates the directories, prepare the vocabulary and embeddings
    """
    logger = logging.getLogger("brc")
    logger.info('Checking the data files...')
    for data_path in args.train_files + args.dev_files + args.test_files:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    logger.info('Preparing the directories...')
    for dir_path in [args.vocab_dir, args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info('Building vocabulary...')
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.train_files, args.dev_files, args.test_files)
    logger.info('Building dateset success')
    vocab = Vocab(lower=True)


    for word in brc_data.word_iter('train'):
        vocab.add(word)

    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt= args.min_cnt)
    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                                                                            vocab.size()))

    logger.info('Assigning embeddings...')
    vocab.randomly_init_embeddings(args.embed_size)
    #vocab.load_pretrained_embeddings(args.emb_files[0])

    logger.info('Saving vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as fout:
        pickle.dump(vocab, fout)

    logger.info('Done with preparing!')

def train(args):
    """
    trains the reading comprehension model
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.train_files, args.dev_files, vocab=vocab)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Initialize the model...')
    if args.algo == 'MCST':
        logger.info('Use MCST Model to train...')
        rc_model = MCSTmodel(vocab, args)
        logger.info('Training MCST model...')
        rc_model.train(brc_data, args.epochs, args.batch_size, save_dir=args.model_dir,
                       save_prefix=args.algo,
                       dropout_keep_prob=args.dropout_keep_prob)
    else:
        rc_model = RCModel(vocab, args)
        logger.info('Training the model...')
        rc_model.train(brc_data, args.epochs, args.batch_size, save_dir=args.model_dir,
                    save_prefix=args.algo,
                    dropout_keep_prob=args.dropout_keep_prob)
    logger.info('Done with model training!')


def evaluate(args):
    """
    evaluate the trained model on dev files
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.dev_files) > 0, 'No dev files are provided.'
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len, dev_files=args.dev_files)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Restoring the model...')
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    logger.info('Evaluating the model on dev set...')
    dev_batches = brc_data.gen_mini_batches('dev', args.batch_size,
                                            pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    dev_loss, dev_bleu_rouge = rc_model.evaluate(
        dev_batches, result_dir=args.result_dir, result_prefix='dev.predicted')
    logger.info('Loss on dev set: {}'.format(dev_loss))
    logger.info('Result on dev set: {}'.format(dev_bleu_rouge))
    logger.info('Predicted answers are saved to {}'.format(os.path.join(args.result_dir)))


def predict(args):
    """
    predicts answers for test files
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.test_files) > 0, 'No test files are provided.'
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          test_files=args.test_files)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Restoring the model...')
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    logger.info('Predicting answers for test set...')
    test_batches = brc_data.gen_mini_batches('test', args.batch_size,
                                             pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    rc_model.evaluate(test_batches,
                      result_dir=args.result_dir, result_prefix='test.predicted')


def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()

    logger = logging.getLogger("brc")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
    if args.predict:
        predict(args)

if __name__ == '__main__':
    run()