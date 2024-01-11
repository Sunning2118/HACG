
import json
import pickle
import re

from gensim.models import Word2Vec
from javalang.ast import Node


def get_token(node):
   
    token = ''
    if isinstance(node, str):  # 是一段字符串
        token = node
    elif isinstance(node, set):  # 是一段集合  {'public', 'private'}
        token = 'Modifier'
    elif isinstance(node, Node):  # 是一个节点
        token = node.__class__.__name__
    return token


def get_child(root):

    if isinstance(root, Node):  
        children = root.children
    elif isinstance(root, set):  
        children = list(root)  
    else:
        children = []  

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    # print(sub_item)
                    yield sub_item
            elif item:
                # print(item)
                yield item

    return list(expand(children))


def get_sequence(node, sequence):

    token, children = get_token(node), get_child(node)
    sequence.append(token)
    for child in children:
        get_sequence(child, sequence)


def split_long_word(word):
    pattern = r'(?<!^)(?=[A-Z])|(?<!\d)(?=\d)'
    if word.find('-') != -1:
        pattern = r'-'
    if word.find('_') != -1:
        pattern = r'_'
    decomposition_word = re.split(pattern, word)
    # decomposition_word = [word.lower() for word in decomposition_word]
    return decomposition_word


if __name__ == '__main__':
    alltokens = []  # 遍历得到的所有的token，有重复,
   

    alltokens_Hump_decomposition = []
    

    set_tokens = []
   

    tokens_set_Hump_decomposition = []
    

    ForW2V_tokens = []
   

    ForW2V_tokens_Hump_decomposition = []
    
    df_codeid_classid_ast_all_path = "../data/java250/DataFrame_pkl/df_codeid_classid_ast_all.pkl"
    with open(df_codeid_classid_ast_all_path, 'rb') as fp:
        df_codeid_fid_ast = pickle.load(fp)
        num_of_row = df_codeid_fid_ast.shape[0]
        num_of_column = df_codeid_fid_ast.shape[1]
        for i in range(num_of_row):
            print(i)
            tmp_tokens = [] 
            AST = df_codeid_fid_ast.loc[i, 'AST']
            Codeid = df_codeid_fid_ast.loc[i, 'Codeid']
            Classid = df_codeid_fid_ast.loc[i, 'Classid']
            get_sequence(AST, tmp_tokens)
            ForW2V_tokens.append(tmp_tokens)
            tmp_tokens_ = []  
            for tokens in tmp_tokens:
                alltokens.append(tokens)
                for tokens_Hump_decomposition in split_long_word(tokens):
                    tmp_tokens_.append(tokens_Hump_decomposition)
                    alltokens_Hump_decomposition.append(tokens_Hump_decomposition)
            ForW2V_tokens_Hump_decomposition.append(tmp_tokens_)


    with open("../data/java250/generate_pkl/alltokens.pkl", 'wb') as f:
        pickle.dump(alltokens, f)
    print('len(alltokens): ', len(alltokens))  # len(alltokens):

    with open("../data/java250/generate_pkl/alltokens_Hump_decomposition.pkl", 'wb') as f:
        pickle.dump(alltokens_Hump_decomposition, f)
    print('len(alltokens_Hump_decomposition): ',
          len(alltokens_Hump_decomposition))  

   
    set_tokens = list(set(alltokens)) 
    with open("../data/java250/generate_pkl/set_tokens.pkl", 'wb') as f:
        pickle.dump(set_tokens, f)
    print('len(set_tokens): ', len(set_tokens))  # len(set_tokens):


    tokens_set_Hump_decomposition = list(set(alltokens_Hump_decomposition))
    with open("../data/java250/generate_pkl/tokens_set_Hump_decomposition.pkl", 'wb') as f:
        pickle.dump(tokens_set_Hump_decomposition, f)
    print('len(tokens_set_Hump_decomposition): ',
          len(tokens_set_Hump_decomposition))  # len(tokens_set_Hump_decomposition):  57767


    with open("../data/java250/generate_pkl/ForW2V_tokens.pkl", 'wb') as f:
        pickle.dump(ForW2V_tokens, f)
    print('len(ForW2V_tokens): ', len(ForW2V_tokens))  # len(ForW2V_tokens):  9989


    with open("../data/java250/generate_pkl/ForW2V_tokens_Hump_decomposition.pkl", 'wb') as f:
        pickle.dump(ForW2V_tokens_Hump_decomposition, f)
    print('len(ForW2V_tokens_Hump_decomposition): ',
          len(ForW2V_tokens_Hump_decomposition))  # len(ForW2V_tokens_Hump_decomposition):  9989


    set_token_size = len(set_tokens)  # tokens的种类数
    set_tokenids = range(set_token_size)  # 给token编码
    dict_token_id = dict(zip(set_tokens, set_tokenids))  # token的字典编码
    with open('../data/java250/generate_json/dict_token_id.json', 'w') as fp:
        json.dump(dict_token_id, fp)
    dict_id_token = dict(zip(set_tokenids, set_tokens))  # token的字典编码
    with open('../data/java250/generate_json/dict_id_token.json', 'w') as fp:
        json.dump(dict_id_token, fp)

    # token驼峰分解后字典
    tokens_set_Hump_decomposition_size = len(tokens_set_Hump_decomposition)
    token_set_Hump_decomposition_ids = range(tokens_set_Hump_decomposition_size)  # 给token编码
    dict_token_id_Hump_decomposition = dict(
        zip(tokens_set_Hump_decomposition, token_set_Hump_decomposition_ids))  # token的字典编码
    with open('../data/java250/generate_json/dict_token_id_Hump_decomposition.json', 'w') as fp:
        json.dump(dict_token_id_Hump_decomposition, fp)
    dict_id_token_Hump_decomposition = dict(
        zip(token_set_Hump_decomposition_ids, tokens_set_Hump_decomposition))  # token的字典编码
    with open('../data/java250/generate_json/dict_id_token_Hump_decomposition.json', 'w') as fp:
        json.dump(dict_id_token_Hump_decomposition, fp)

    # 训练word2vec的模型
    w2v = Word2Vec(sentences=ForW2V_tokens,  # 语料库，series可迭代
                   vector_size=128,  # 向量的尺寸128
                   workers=16,  # 并行
                   sg=1,  # 是0， 则是CBOW模型，是1则是Skip-Gram模型，默认是0即CBOW模型。
                   min_count=1
                   )
    w2v.save("./model/token2vec_alltoken_128.model")  # word2vec

    w2v = Word2Vec(sentences=ForW2V_tokens_Hump_decomposition,  # 语料库，series可迭代
                   vector_size=128,  # 向量的尺寸128
                   workers=16,  # 并行
                   sg=1,  # 是0， 则是CBOW模型，是1则是Skip-Gram模型，默认是0即CBOW模型。
                   min_count=1
                   )
    w2v.save("./model/token2vec_alltoken_Hump_decomposition_128.model")
