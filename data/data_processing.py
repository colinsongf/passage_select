# -*- coding:utf8 -*-
"""
Data processing file
1. divide dataset ：
    python data_processing.py --div DEYN --div_dir ./demo/
2 anaylze datast : 
    python data_processing.py -a --analysis_files ./preprocessed/trainset/search --result_dir ./raw/results/
3. prepare baseline data set, add 'segmented_question'(list[]), 'segmented_title'(list[]), 'segmented_paragraphs'(list[][])，"segmented_answers" (list[][]) to the json
    python data_processing.py --pre_base --seg_dir ./seg/ --target_files ./raw/trainset/search.train.json.part100
    python data_processing.py --pre_base --seg_dir ./seg/search/dev/ --target_files ./raw/devset/search.dev.json.part 
4. cut the train data set :
    python data_processing.py --cut --target_files ./raw/trainset/search.train.json --size 1000
5. glove train prepare
    python data_processing.py --seg_trainfile  --target_files ./demo/
6. analysis title 
    python data_processing.py -a is_selected --analysis_files ./data/dataset/search.train.json
    python data_processing.py -a is_selected --analysis_files ./preprocessed/trainset/search
data 2019.1.10
author zh lee
"""

import sys
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.append('..')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import argparse
import logging
import json
import thulac
import datetime


start_time = ""
end_time = ""

train_search_file = "search.train.json"
train_zhidao_file = "zhidao.train.json"
dev_search_file = "search.dev.json"
dev_zhidao_file = "zhidao.dev.json"
test_search_file = "search.test.json"
test_zhidao_file = "zhidao.test.json"

train_file = "trainset/"
dev_file = "devset/"
test_file = "testset/"

all_dic = {}
all_q_dic = {}
all_a_dic = {}


num_q = 0
num_a = 0
a_in_para = []
a_in_doc = []
a_in_dic = []

def list2string(lis):
    strin = ''
    for l in lis:
        strin += l
    return strin

def a_dic_add(word):
    if word not in all_a_dic:
        all_a_dic.append(word)
    if word not in all_dic:
        all_dic.append(word)

def q_dic_add(word):
    if not(all_q_dic.has_key(word[0])):
        all_q_dic[word[0]] = word[1]

def add_list(word,list):
    if word not in list:
        list.append(word)

def build_dic(args):
    logger = logging.getLogger("read")
    logger.info("Build dictionary " + args.analysis_files[0])
    f = open(args.analysis_files[0])
    line = f.readline()
    seg = thulac.thulac(seg_only=True)
    n_line = 1
    while line:
        #logger.info("\033[0;30;46m process line : " +str(n_line)+ "\033[0m")
        data = json.loads(line)
        for key, value in data.items():
            if key == "documents":
                for pair in value:
                    for (k, v) in pair.items():
                        n_para = 0
                        if k == "title":
                            #未处理
                            ans_str = v.encode('utf-8').strip()
                            words = seg.cut(ans_str)
                            # print "\033[1;36;40m ---- title: \033[0m" + " ".join(" ".join(str(x) for x in word) for word in words)
                        elif k == "paragraphs":
                            if isinstance(v, list):
                                for vv in v:
                                    n_para = n_para + 1
                                    if isinstance(vv, list):
                                        ss = str(vv)
                                    else:
                                        para_str = vv.encode('utf-8').strip()
                                        words = seg.cut(para_str)
                                        # print "\033[1;36;40m ----Number " + " paragraph: \033[0m" + " ".join(" ".join(str(x) for x in word) for word in words)
                                        for word in words:
                                            q_dic_add(word)
                        else:
                            logger.info("unknown key" + k)

        line = f.readline()
        n_line += 1
    logger.info("Build dictionary success which has words: " + str(len(all_q_dic)))
    f.close()

# def analysis_data(args):
#     """
#     Analysis data in data set
#     :return:
#     """
#     logger = logging.getLogger("read")
#     #build_dic(args)
#     logger.info("analysis_data "+ args.analysis_files[0])
#     #of = open(args.result_dir + args.analysis_files[0].split('/')[3]+ ".result",'w')
#     #of.truncate()
#     f = open(args.analysis_files[0])
#     #seg = thulac.thulac(seg_only=True)
#
#     n_line = 1
#     for line in f:
#         n_line += 1
#     print 'Question set =' + str(n_line)
#     f.close()
#     #of.close()

def is_selected(args):
    """
    Analysis data in data set
    :return: 
    """
    logger = logging.getLogger("read")
    #build_dic(args)
    logger.info("analysis_data "+ args.analysis_files[0])
    #of = open(args.result_dir + args.analysis_files[0].split('/')[3]+ ".result",'w')
    #of.truncate()
    f = open(args.analysis_files[0])
    #seg = thulac.thulac(seg_only=True)
    num_question = 0
    num_passage = 0
    num_ques_has_selected_answ = 0
    num_ques_has_no_selected_answ = 0
    num_ques_has_selected = 0
    num_ques_has_no_selected = 0
    num_no_answer = 0
    num_no_answer_no_selected = 0
    for line in f:
        num_question += 1
        sample = json.loads(line)
        answer_list = sample['answers']
        answer_doc_id = sample['answer_docs']
        is_select = False
        if len(answer_list) == 0:
            num_no_answer += 1
        for doc_id in answer_doc_id:
            if sample['documents'][doc_id]['is_selected'] == True:
                is_select = True
        if is_select:
            num_ques_has_selected_answ += 1
        else:
            num_ques_has_no_selected_answ += 1
        has_selected = False
        for d_idx, doc in enumerate(sample['documents']):
            if doc['is_selected'] == True:
                has_selected = True
        if has_selected:
            num_ques_has_selected += 1
        else:
            if len(answer_list) == 0:
                num_no_answer_no_selected += 1
            print list2string(answer_list)
            num_ques_has_no_selected += 1

    print 'Question set = '
    print str(num_question)
    print 'Num of have selected information = '
    print str(num_ques_has_selected_answ)
    print 'Num of do not have selected information = '
    print str(num_ques_has_no_selected_answ)
    print 'Num of have selected doc = ' + str(num_ques_has_selected)
    print 'Num of do not have selected doc = ' + str(num_ques_has_no_selected)
    print 'Num of have no answer = ' + str(num_no_answer)
    print 'Num of no answer and no selected = ' + str(num_no_answer_no_selected)
    f.close()
    #of.close()

def unicode(str):
    return str.encode('utf-8')

def pre_segment(args):
    logger = logging.getLogger("data_process")
    logger.info('Checking the data files...')
    logger.info('Preparing the directories...')
    for dir_path in [args.seg_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    if args.target_files:
        seg = thulac.thulac(seg_only=True)
        files = os.listdir(args.target_files[0])
        for target_file in files:
            target_file = args.target_files[0] + '/' + target_file
            logger.info('Target file: {} questions.'.format(str(target_file)))
            f_name = target_file.split('/')
            f_name = f_name[len(f_name)-1]
            out_file = open(args.seg_dir + f_name, 'w')
            out_file.truncate()
            logger.info('Write to : {} file.'.format(str(out_file)))
            with open(target_file) as fin:
                n = 1
                for line in fin:
                    #print n
                    n +=1
                    sample = json.loads(line)
                    sample['segmented_question'] = []
                    #question
                    #print sample['question']
                    words = seg.cut(sample['question'])
                    for word in words:
                        sample['segmented_question'].append(word[0])
                    #print sample['segmented_question']
                    # doc
                    for d_idx, doc in enumerate(sample['documents']):
                        #print "doc "+ str(d_idx)
                        doc['segmented_title'] = []
                    # title
                        #print doc['title']
                        words = seg.cut(doc['title'])
                        for word in words:
                            doc['segmented_title'].append(word[0])
                        #print doc['segmented_title']
                    # paragraphs
                        doc['segmented_paragraphs'] = []
                        for p_idx, para in enumerate(doc['paragraphs']):
                            segment_para = []
                            #print "para" + str(p_idx)
                            words = seg.cut(para)
                            for word in words:
                                segment_para.append(word[0])
                            doc['segmented_paragraphs'].append(segment_para)
                        #print doc['segmented_paragraphs']
                    # answer
                    if sample.has_key('answers'):
                        sample['segmented_answers'] = []
                        for a_idx, ans in enumerate(sample['answers']):
                            seg_ans = []
                            #print "answer: " +str(a_idx)
                            words = seg.cut(ans)
                            for word in words:
                                seg_ans.append(word[0])
                            sample['segmented_answers'].append(seg_ans)
                    #print sample['segmented_answers']
                    out_file.write(str(json.dumps(sample, encoding='utf8', ensure_ascii=False))+'\n')
                    out_file.flush()
            out_file.close()
            logger.info('segment success')

def output_trainfile(args):
    logger = logging.getLogger("data_process")
    logger.info('Checking the data files...')
    logger.info('Preparing the directories...')
    # for dir_path in [args.seg_dir, args.result_dir, args.summary_dir]:
    #     if not os.path.exists(dir_path):
    #         os.makedirs(dir_path)

    if args.target_files:
        seg = thulac.thulac(seg_only=True)
        files = os.listdir(args.target_files[0])
        seg_file_path = './seg_text'
        seg_file = open(seg_file_path,'w')
        seg_file.truncate()
        num_words = 0
        logger.info('Write to : {} file.'.format(str(seg_file)))
        for target_file in files:
            target_file = args.target_files[0] + '/' + target_file
            logger.info('Target file: {} questions.'.format(target_file))
            with open(target_file) as fin:
                n = 1
                for line in fin:
                    if n%50 == 0:
                        print n
                    n += 1
                    sample = json.loads(line)
                    seg_str = ''
                    #words = seg.cut(sample['question'])
                    for word in sample['segmented_question']:
                        seg_str += ' ' + word
                        num_words += 1
                    #print sample['segmented_question']
                    # doc
                    for d_idx, doc in enumerate(sample['documents']):
                        #print "doc "+ str(d_idx)
                        for word in doc['segmented_title']:
                            seg_str += ' ' + word
                            num_words += 1
                        #print doc['segmented_title']
                    # paragraphs
                        for p_idx, para in enumerate(doc['segmented_paragraphs']):
                            for word in para:
                                seg_str += ' ' + word
                                num_words += 1
                    # answer
                    if sample.has_key('answers'):
                        sample['segmented_answers'] = []
                        for a_idx, ans in enumerate(sample['answers']):
                            seg_ans = []
                            #print "answer: " +str(a_idx)
                            words = seg.cut(ans)
                            for word in words:
                                seg_ans.append(word[0])
                                seg_str += ' ' + word[0]
                                num_words += 1
                            sample['segmented_answers'].append(seg_ans)
                    #print sample['segmented_answers']
                    seg_file.write(seg_str + '\n')
                    seg_file.flush()
                print ('num_of_words', num_words)

        seg_file.close()
        logger.info('glove success')



def divide_DEYN(args):
    """
    Divide dataset by Description or Entity or Yes/No
    """
    logger = logging.getLogger("data_process")
    in_file_list = [train_file+train_search_file,train_file+ train_zhidao_file,dev_file + dev_search_file,
     dev_file + dev_zhidao_file,test_file + test_search_file,test_file + test_zhidao_file]
    #in_file_list = [train_file+train_search_file,  dev_file + dev_search_file, test_file + test_search_file]
    for file in in_file_list:
        p = file.split('/')
        DES_file_path = args.div_dir + p[0] + "/description."+p[1]
        ENT_file_path = args.div_dir + p[0] + "/entity." + p[1]
        YN_file_path = args.div_dir + p[0] + "/yesno." + p[1]


        ofile_des = open(DES_file_path,'w')
        ofile_des.truncate()
        ofile_ent = open(ENT_file_path,'w')
        ofile_ent.truncate()
        ofile_yn = open(YN_file_path,'w')
        ofile_yn.truncate()
        path = args.div_dir+file
        logger.info("Open file: "+str(path))
        line_n = 1
        with open(path) as fin:
            f = fin.readline()
            while f:
                data = json.loads(f)
                for (key,value) in data.items():
                    if key == "question_type":
                        if value == "DESCRIPTION":
                            print "DESCRIPTION " + str(line_n)
                            ofile_des.write(f)
                            ofile_des.flush()
                        elif value == "ENTITY":
                            print "ENTITY " + str(line_n)
                            ofile_ent.write(f)
                            ofile_ent.flush()
                        elif value == "YES_NO":
                            print "YES_NO " + str(line_n)
                            ofile_yn.write(f)
                            ofile_yn.flush()
                        else: print "unknown "+ str(value) + " " +str(line_n)
                f = fin.readline()
                line_n += 1
        ofile_des.close()
        ofile_ent.close()
        ofile_yn.close()

def divide_FO(args):
    """
    Divide dataset by Fact and Opinion 
    :return: 
    """
    logger = logging.getLogger("data_process")
    in_file_list = [train_file+train_search_file,train_file+ train_zhidao_file,dev_file + dev_search_file,
     dev_file + dev_zhidao_file,test_file + test_search_file,test_file + test_zhidao_file]
    #in_file_list = [train_file + train_search_file, dev_file + dev_search_file, test_file + test_search_file]
    for file in in_file_list:
        p = file.split('/')
        opn_file_path = args.div_dir + p[0] + "/opinion." + p[1]
        fact_file_path = args.div_dir + p[0] + "/fact." + p[1]

        opn_file = open(opn_file_path, 'w')
        opn_file.truncate()
        fact_file = open(fact_file_path, 'w')
        fact_file.truncate()
        path = args.div_dir + file
        logger.info("Open file: " + str(path))
        line_n = 1
        with open(path) as fin:
            f = fin.readline()
            while f:
                data = json.loads(f)
                for (key, value) in data.items():
                    if key == "fact_or_opinion":
                        if value == "FACT":
                            print "FACT " + str(line_n)
                            fact_file.write(f)
                            fact_file.flush()
                        elif value == "OPINION":
                            print "OPINION " + str(line_n)
                            opn_file.write(f)
                            opn_file.flush()
                        else:
                            print "unknown " + str(value) + " " + str(line_n)
                f = fin.readline()
                line_n += 1
        opn_file.close()
        fact_file.close()

def _parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser('Pre_procession of raw data set')
    parser.add_argument('-a', '--analysis',choices=['is_selected','analysis'], default= None,
                                help=" analysis ")
    parser.add_argument('-b', '--pre_base',
                        help="pre_process for base line system",
                        action='store_true')
    parser.add_argument('-s', '--seg_trainfile',
                        help="pre_process for glove train_file",
                        action='store_true')
    parser.add_argument('--is_selected',
                        help="analyze data set",
                        action='store_true')
    parser.add_argument('--seg', choices=['label', 'nolabel'], default= None,
                        help='if use segment model')
    parser.add_argument('--div', choices=['DEYN', 'FO'], default= None,
                                help="if divide data set into sub set, "+
                                     "choose DEYN: divide data use Answer type: Description / Entity / Yes_No or"+
                                     "choose FO: divide data use Question type: Fact / Opinion")

    model_settings = parser.add_argument_group('parameter settings')
    model_settings.add_argument('--size', type=int, default=100,
                                help='size of the line')
    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--analysis_files', nargs='+',
                               default=['./demo/trainset/search.train.json'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--target_files', nargs='+',
                               default=['./draw/trainset/search.train.json'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--train_files', nargs='+',
                               default=['./draw/trainset/search.train.json'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['./demo/devset/search.dev.json'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['./demo/testset/search.test.json'],
                               help='list of files that contain the preprocessed test data')
    path_settings.add_argument('--seg_dir', default=['./seg/'],
                               help='the dir to write segment results ')
    path_settings.add_argument('--div_dir', default='./demo/',
                               help='the dir with data set preprocessed to be divide')
    path_settings.add_argument('--result_dir', default='../data/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='../data/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()

def run():
    args = _parse_args()
    start_time = str(datetime.datetime.now())
    logger = logging.getLogger("data_process")
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
    if args.analysis == 'is_selected':
        is_selected(args)
    if args.div == "DEYN":
        divide_DEYN(args)
    if args.div == "FO":
        divide_FO(args)
    if args.pre_base:
        pre_segment(args)
    if args.seg_trainfile:
        output_trainfile(args)
    end_time = str(datetime.datetime.now())
    logger.info('Start at ' + start_time)
    logger.info('End at '+ end_time)




if __name__ == '__main__':
    run()