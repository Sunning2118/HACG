
import os

import pandas as pd


def generate_df_class_id_code():
    df_codeid_classid_code_all = pd.DataFrame(columns=['Codeid', 'Classid', 'Code'])
    original_data_folder_name = os.listdir(original_data_path)
    original_data_folder_name_int = [int(x) for x in original_data_folder_name]
    count = 0
    for Classid in original_data_folder_name_int:
        
        folder_path = os.path.join(original_data_path, str(Classid))

        items = os.listdir(folder_path)

        for javafile_name in items:
            if javafile_name.endswith(".java"):
                Codeid = javafile_name[:-5]
                java_path = os.path.join(folder_path, javafile_name)
                count += 1
                print(count)
                # 指定文件编码为 UTF-8
                with open(java_path, "r", encoding="utf-8") as f:
                    Code = f.read()
                dict_tmp = {"Codeid": Codeid, "Classid": Classid, "Code": Code}
                # print(Codeid, Classid)
                df_codeid_classid_code_all.loc[len(df_codeid_classid_code_all)] = dict_tmp
    df_codeid_classid_code_all.to_pickle(df_codeid_classid_code_all_path)


if __name__ == '__main__':
    original_data_path = "data/java250/OriginalData/java250_noimport_nocomment"
    df_codeid_classid_code_all_path = "data/java250/DataFrame_pkl/df_codeid_classid_code_all.pkl"
    generate_df_class_id_code()  # 生成DataFrame：Classid_Code；写入DataFrame_pkl下的df_classid_code_all.pkl
