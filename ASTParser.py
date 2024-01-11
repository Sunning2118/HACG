"""
 将每段代码生成一棵AST树，构造DataFrame(Fid, Code，AST)
"""
import pickle

import javalang
import pandas as pd


def generate_df_class_id_ast():
    df_codeid_class_id_ast_all = pd.DataFrame(columns=['Codeid', 'Classid', 'AST'])
    count = 0
    with open(df_codeid_classid_code_all_path, 'rb') as fp:
        df_class_id_code_all = pickle.load(fp)
        num_of_row = df_class_id_code_all.shape[0]
        num_of_column = df_class_id_code_all.shape[1]
        for i in range(num_of_row):
            Codeid = df_class_id_code_all.loc[i, 'Codeid']
            Classid = df_class_id_code_all.loc[i, 'Classid']
            Code = df_class_id_code_all.loc[i, 'Code']
            try:
                AST = javalang.parse.parse(Code)  
            except javalang.parser.JavaSyntaxError as e:

                print('Java syntax error:', e)
                print(Codeid)
            except javalang.tokenizer.LexerError as e:
                print('Java lexer error:', e)
                print(Codeid)
            else:
                count += 1
                print(count)
                dict_tmp = {"Codeid": Codeid, "Classid": Classid, "AST": AST}
                df_codeid_class_id_ast_all.loc[len(df_codeid_class_id_ast_all)] = dict_tmp
    df_codeid_class_id_ast_all.to_pickle(df_codeid_classid_ast_all_path)


if __name__ == '__main__':
    df_codeid_classid_code_all_path = "data/java250/DataFrame_pkl/df_codeid_classid_code_all.pkl"
    df_codeid_classid_ast_all_path = "../data/java250/DataFrame_pkl/df_codeid_classid_ast_all.pkl"
    generate_df_class_id_ast()
