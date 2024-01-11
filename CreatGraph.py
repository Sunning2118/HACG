"""

"""
import copy
import json
import pickle
import sys
from collections import deque

import dgl
import pandas as pd
import torch
from anytree import AnyNode
from gensim.models import Word2Vec

from CS03_Dict import get_token, get_child


def createtree(root_ct, node_ct, parent_ct):

    token_ct, children_ct = get_token(node_ct), get_child(node_ct)
    newnode_ct = AnyNode(token=None, data=None, parent=None, ifRoot=None)
    if parent_ct is None:  # 根节点不用AnyNode声明，仅需要进行赋值
        root_ct.token = token_ct
        root_ct.data = node_ct  # 元数据
        root_ct.parent = None  #
        root_ct.ifRoot = True

    else:
        newnode_ct = AnyNode(token=None, data=None, parent=None, ifRoot=False)
        newnode_ct.token = token_ct
        newnode_ct.data = node_ct  # 元数据
        newnode_ct.parent = parent_ct  #
        newnode_ct.ifRoot = False

    for child_ct in children_ct:
        if get_token(child_ct):
            if parent_ct is None:
                createtree(root_ct, child_ct, root_ct)
            else:
                createtree(root_ct, child_ct, newnode_ct)


def anytree_node_id_order(ASTroot, dict_token_id, dict_node_id_tokenid):

    queue = deque([(ASTroot, 1)])  
    order = 0  
    while queue:
        level_size = len(queue)
        for _ in range(level_size):
            node, _ = queue.popleft()
            node.id = order  
            dict_node_id_tokenid[order] = dict_token_id[node.token]
            order += 1
            for child in node.children:
                queue.append((child, order))  


def getnodeandedge(node, src, tgt):

    for child in node.children:  # 对齐
        src.append(node.id)
        tgt.append(child.id)
        getnodeandedge(child, src, tgt)


def getnodeandedge_StateFlow(root):
    src = []
    tgt = []

    statement_name_list = ['LocalVariableDeclaration',
                           'StatementExpression',
                           'IfStatement',
                           'SwitchStatement',
                           'ForStatement',
                           'WhileStatement',
                           'DoStatement',
                           'ReturnStatement',
                           'TryStatement',
                           'BlockStatement',
                           'Statement',
                           'AssertStatement',
                           'BreakStatement',
                           'ContinueStatement',
                           'ThrowStatement'
                           ]
    MethodDeclaration_name = 'MethodDeclaration'
    BlockStatement_name = 'BlockStatement'
    for descendant in root.descendants:
        if descendant.token == MethodDeclaration_name or descendant.token == BlockStatement_name:
            MethodDeclaration_Node = descendant
            count = 0
            pass_count = 0  # 防止mathod和block下面只有一个有用的语句
            min_count = sys.maxsize
            max_count = -1
            for child in MethodDeclaration_Node.children:  # 对齐
                if child.token in statement_name_list:
                    pass_count = pass_count + 1
                    if count > max_count:
                        max_count = count
                    if count < min_count:
                        min_count = count
                count = count + 1
            if pass_count <= 1:
                continue
            count = 0
            for child in MethodDeclaration_Node.children:  # 对齐
                if child.token in statement_name_list:
                    if count == min_count:
                        src.append(child.id)
                    elif count == max_count:
                        tgt.append(child.id)
                    else:
                        src.append(child.id)
                        tgt.append(child.id)
                count = count + 1
    return src, tgt


def getnodeandedge_IfStatement(root):
    statement_name_list = ['LocalVariableDeclaration',
                           'StatementExpression',
                           'IfStatement',
                           'SwitchStatement',
                           'ForStatement',
                           'WhileStatement',
                           'DoStatement',
                           'ReturnStatement',
                           'TryStatement',
                           'BlockStatement',
                           'Statement',
                           'AssertStatement',
                           'BreakStatement',
                           'ContinueStatement',
                           'ThrowStatement'
                           ]
    src = []
    tgt = []
    if root.parent.token == 'MethodDeclaration' or root.parent.token == 'BlockStatement':  # 只关心纯一级IfStatement的节点
        root_parent = root.parent
    else:
        return src, tgt

    siblings = root_parent.children  # 寻找纯一级的IfStatement节点的右侧兄弟节点
    root_index = siblings.index(root)
    root_right_sibling_index = root_index + 1
    root_right_sibling = None

    while root_right_sibling_index < len(siblings):  # 如果右边有兄弟节点
        root_right_sibling = siblings[root_right_sibling_index]
        if root_right_sibling.token in statement_name_list:
            break
        root_right_sibling_index = root_right_sibling_index + 1

    # 上下级：if->block；if->if
    BlockStatement_id_under_IfStatement_list = []
    for root_child in root.children:  # 纯一级的IfStatement节点的源节点、目的节点
        if root_child.token == 'BlockStatement' or root_child.token == 'IfStatement':
            src.append(root.id)
            tgt.append(root_child.id)

    for root_descendant in root.descendants:
        if root_descendant.token == 'BlockStatement' and root_descendant.parent.token == 'IfStatement':  # Block节点
            BlockStatement_id_under_IfStatement_list.append(root_descendant.id)
        if root_descendant.token == 'IfStatement':
            for root_descendant_child in root_descendant.children:
                if root_descendant_child.token == 'BlockStatement' or root_descendant_child.token == 'IfStatement':
                    src.append(root_descendant.id)
                    tgt.append(root_descendant_child.id)

    if root_right_sibling:
        for BlockStatement_id in BlockStatement_id_under_IfStatement_list:
            src.append(BlockStatement_id)
            tgt.append(root_right_sibling.id)
    return src, tgt


def getnodeandedge_SwitchStatement(root):
    statement_name_list = ['LocalVariableDeclaration',
                           'StatementExpression',
                           'IfStatement',
                           'SwitchStatement',
                           'ForStatement',
                           'WhileStatement',
                           'DoStatement',
                           'ReturnStatement',
                           'TryStatement',
                           'BlockStatement',
                           'Statement',
                           'AssertStatement',
                           'BreakStatement',
                           'ContinueStatement',
                           'ThrowStatement'
                           ]
    src = []
    tgt = []
    if root.parent.token == 'MethodDeclaration' or root.parent.token == 'BlockStatement':  # 只关心纯一级SwitchStatement的节点
        root_parent = root.parent
    else:
        return src, tgt

    siblings = root_parent.children 
    root_index = siblings.index(root)
    root_right_sibling_index = root_index + 1
    root_right_sibling = None

    while root_right_sibling_index < len(siblings):  
        root_right_sibling = siblings[root_right_sibling_index]
        if root_right_sibling.token in statement_name_list:
            break
        root_right_sibling_index = root_right_sibling_index + 1


    SwitchStatementCase_id_under_SwitchStatement_list = []
    for root_child in root.children:  
        if root_child.token == 'SwitchStatementCase':
            src.append(root.id)
            tgt.append(root_child.id)
            SwitchStatementCase_id_under_SwitchStatement_list.append(root_child.id)

    if root_right_sibling:
        for SwitchStatementCase_id in SwitchStatementCase_id_under_SwitchStatement_list:
            src.append(SwitchStatementCase_id)
            tgt.append(root_right_sibling.id)
    return src, tgt


def getnodeandedge_ForStatement(root):
    statement_name_list = ['LocalVariableDeclaration',
                           'StatementExpression',
                           'IfStatement',
                           'SwitchStatement',
                           'ForStatement',
                           'WhileStatement',
                           'DoStatement',
                           'ReturnStatement',
                           'TryStatement',
                           'BlockStatement',
                           'Statement',
                           'AssertStatement',
                           'BreakStatement',
                           'ContinueStatement',
                           'ThrowStatement'
                           ]
    src = []
    tgt = []
    if root.parent.token == 'MethodDeclaration' or root.parent.token == 'BlockStatement':  # 只关心纯一级SwitchStatement的节点
        root_parent = root.parent
    else:
        return src, tgt

    siblings = root_parent.children  # 寻找纯一级的IfStatement节点的右侧兄弟节点
    root_index = siblings.index(root)
    root_right_sibling_index = root_index + 1
    root_right_sibling = None

    while root_right_sibling_index < len(siblings):  # 如果右边有兄弟节点
        root_right_sibling = siblings[root_right_sibling_index]
        if root_right_sibling.token in statement_name_list:
            break
        root_right_sibling_index = root_right_sibling_index + 1


    For_BlockStatement_Node = None
    for root_child in root.children:  
        if root_child.token == 'BlockStatement':
            For_BlockStatement_Node = root_child
            if For_BlockStatement_Node:
                src.append(root.id)
                tgt.append(For_BlockStatement_Node.id)
                src.append(For_BlockStatement_Node.id)
                tgt.append(root.id)

    if root_right_sibling:
        src.append(root.id)
        tgt.append(root_right_sibling.id)
        if For_BlockStatement_Node:
            src.append(For_BlockStatement_Node.id)
            tgt.append(root_right_sibling.id)
    return src, tgt


def getnodeandedge_WhileStatement(root):
    statement_name_list = ['LocalVariableDeclaration',
                           'StatementExpression',
                           'IfStatement',
                           'SwitchStatement',
                           'ForStatement',
                           'WhileStatement',
                           'DoStatement',
                           'ReturnStatement',
                           'TryStatement',
                           'BlockStatement',
                           'Statement',
                           'AssertStatement',
                           'BreakStatement',
                           'ContinueStatement',
                           'ThrowStatement'
                           ]
    src = []
    tgt = []
    if root.parent.token == 'MethodDeclaration' or root.parent.token == 'BlockStatement':  # 只关心纯一级SwitchStatement的节点
        root_parent = root.parent
    else:
        return src, tgt

    siblings = root_parent.children  # 寻找纯一级的IfStatement节点的右侧兄弟节点
    root_index = siblings.index(root)
    root_right_sibling_index = root_index + 1
    root_right_sibling = None

    while root_right_sibling_index < len(siblings):  # 如果右边有兄弟节点
        root_right_sibling = siblings[root_right_sibling_index]
        if root_right_sibling.token in statement_name_list:
            break
        root_right_sibling_index = root_right_sibling_index + 1

    # 上下级：WhileStatement<->BlockStatement
    While_BlockStatement_Node = None
    for root_child in root.children:  # 纯一级的WhileStatement节点的源节点、目的节点
        if root_child.token == 'BlockStatement':
            While_BlockStatement_Node = root_child
            if While_BlockStatement_Node:
                src.append(root.id)
                tgt.append(While_BlockStatement_Node.id)
                src.append(While_BlockStatement_Node.id)
                tgt.append(root.id)

    if root_right_sibling:
        src.append(root.id)
        tgt.append(root_right_sibling.id)
        if While_BlockStatement_Node:
            src.append(While_BlockStatement_Node.id)
            tgt.append(root_right_sibling.id)

    return src, tgt


def getnodeandedge_DoStatement(root):
    statement_name_list = ['LocalVariableDeclaration',
                           'StatementExpression',
                           'IfStatement',
                           'SwitchStatement',
                           'ForStatement',
                           'WhileStatement',
                           'DoStatement',
                           'ReturnStatement',
                           'TryStatement',
                           'BlockStatement',
                           'Statement',
                           'AssertStatement',
                           'BreakStatement',
                           'ContinueStatement',
                           'ThrowStatement'
                           ]
    src = []
    tgt = []
    if root.parent.token == 'MethodDeclaration' or root.parent.token == 'BlockStatement':  # 只关心纯一级SwitchStatement的节点
        root_parent = root.parent
    else:
        return src, tgt

    siblings = root_parent.children  # 寻找纯一级的IfStatement节点的右侧兄弟节点
    root_index = siblings.index(root)
    root_right_sibling_index = root_index + 1
    root_right_sibling = None

    while root_right_sibling_index < len(siblings):  # 如果右边有兄弟节点
        root_right_sibling = siblings[root_right_sibling_index]
        if root_right_sibling.token in statement_name_list:
            break
        root_right_sibling_index = root_right_sibling_index + 1

    # 上下级：DoStatement<->BlockStatement
    Do_BlockStatement_Node = None
    for root_child in root.children:  # 纯一级的WhileStatement节点的源节点、目的节点
        if root_child.token == 'BlockStatement':
            Do_BlockStatement_Node = root_child
            if Do_BlockStatement_Node:
                src.append(root.id)
                tgt.append(Do_BlockStatement_Node.id)
                src.append(Do_BlockStatement_Node.id)
                tgt.append(root.id)

    if root_right_sibling:
        if Do_BlockStatement_Node:
            src.append(Do_BlockStatement_Node.id)
            tgt.append(root_right_sibling.id)

    return src, tgt


def getnodeandedge_ControlFlow(root):
    src = []
    tgt = []
    statement_name_list = [
        'IfStatement',
        'SwitchStatement',
        'ForStatement',
        'WhileStatement',
        'DoStatement',
    ]
    for descendant in root.descendants:
        if descendant.token in statement_name_list:
            if descendant.token == 'IfStatement':
                src_tmp, tgt_tmp = getnodeandedge_IfStatement(descendant)
            elif descendant.token == 'SwitchStatement':
                src_tmp, tgt_tmp = getnodeandedge_SwitchStatement(descendant)
            elif descendant.token == 'ForStatement':
                src_tmp, tgt_tmp = getnodeandedge_ForStatement(descendant)
            elif descendant.token == 'WhileStatement':
                src_tmp, tgt_tmp = getnodeandedge_WhileStatement(descendant)
            else:
                src_tmp, tgt_tmp = getnodeandedge_DoStatement(descendant)
            src = src + src_tmp
            tgt = tgt + tgt_tmp
    return src, tgt


def getnodeandedge_DefUseFlow(root):
    src = []
    tgt = []
    var_token_path = {}  
    for descendant in root.descendants:  
       
        if descendant.token == 'FormalParameter':
            if_basic_type = 0
            FormalParameter_node = descendant
            FormalParameter_node_children_len = len(FormalParameter_node.children)
            for FormalParameter_child_node in FormalParameter_node.children:
                if FormalParameter_child_node.token is 'BasicType':  # 只看基础类型的
                    if_basic_type = 1
                    break
            if if_basic_type == 1:
                var_node = FormalParameter_node.children[FormalParameter_node_children_len - 1]
                var_name = var_node.token
                if var_name in var_token_path:  # 如果已经存在于字典中
                    var_path = var_token_path[var_name]
                    if len(var_path) > 1:  # 有def和use
                        src_tmp = copy.deepcopy(var_path)
                        tgt_tmp = copy.deepcopy(var_path)
                        src_tmp.pop()
                        tgt_tmp.pop(0)
                        src = src + src_tmp
                        tgt = tgt + tgt_tmp
                        var_token_path[var_name] = [var_node.id]
                    else:
                        var_token_path[var_name] = [var_node.id]
                else:  # 如果尚未放入字典中
                    var_token_path[var_name] = [var_node.id]


        elif descendant.token == 'VariableDeclarator':  # 声明，VariableDeclarator最多只有两个孩子节点
            if_basic_type = 0
            VariableDeclarator_node = descendant
            VariableDeclarator_child_node = VariableDeclarator_node.children[0]
            VariableDeclarator_parent_node = VariableDeclarator_node.parent
            for VariableDeclarator_parent_child in VariableDeclarator_parent_node.children:
                if VariableDeclarator_parent_child.token is 'BasicType':  # 只看基础类型的
                    if_basic_type = 1
                    break
            if if_basic_type == 1:  # 如果是基础类型的，引用类型则忽略
                var_node = VariableDeclarator_child_node
                var_name = var_node.token
                if var_name in var_token_path:  # 如果已经存在于字典中
                    var_path = var_token_path[var_name]
                    if len(var_path) > 1:
                        src_tmp = copy.deepcopy(var_path)
                        tgt_tmp = copy.deepcopy(var_path)
                        src_tmp.pop()
                        tgt_tmp.pop(0)
                        src = src + src_tmp
                        tgt = tgt + tgt_tmp
                        var_token_path[var_name] = [var_node.id]
                    else:
                        var_token_path[var_name] = [var_node.id]
                else:  # 如果尚未放入字典中
                    var_token_path[var_name] = [var_node.id]

        elif descendant.token == 'MemberReference':  
            MemberReference_node = descendant
            MemberReference_node_children_len = len(MemberReference_node.children)
            var_node = MemberReference_node.children[MemberReference_node_children_len - 1]
            var_name = var_node.token
            if var_name in var_token_path:  
                var_token_path[var_name].append(var_node.id)
    for var_name in var_token_path.keys():
        var_path = var_token_path[var_name]
        if len(var_path) > 1:
            src_tmp = copy.deepcopy(var_path)
            tgt_tmp = copy.deepcopy(var_path)
            src_tmp.pop()
            tgt_tmp.pop(0)
            src = src + src_tmp
            tgt = tgt + tgt_tmp

    return src, tgt


if __name__ == '__main__':
    with open("../data/java250/generate_json/dict_token_id.json", 'rb') as fp:  # 字典 用于编号
        dict_token_id = json.load(fp)
    with open("../data/java250/generate_json/dict_id_token.json", 'rb') as fp:
        dict_id_token = json.load(fp)
    w2v = Word2Vec.load("./model/token2vec_alltoken_128.model")

    df_codeid_classid_ast_all_path = "../data/java250/DataFrame_pkl/df_codeid_classid_ast_all.pkl"
    df_Codeid_Classid_GraphDS_all_path = "../data/java250/DataFrame_pkl/df_codeid_classid_heteroGraphDS_sota.pkl"

    df_Codeid_Classid_GraphDS_all = pd.DataFrame(columns=['Codeid', 'Classid', 'Graph', 'etype', 'Node_feature'])

    statement_name_list = ['LocalVariableDeclaration',
                           'StatementExpression',
                           'IfStatement',
                           'SwitchStatement',
                           'ForStatement',
                           'WhileStatement',
                           'DoStatement',
                           'ReturnStatement',
                           'TryStatement',
                           'BlockStatement',
                           'Statement',
                           'AssertStatement',
                           'BreakStatement',
                           'ContinueStatement',
                           'ThrowStatement'
                           ]

    with open(df_codeid_classid_ast_all_path, 'rb') as fp:
        df_codeid_classid_ast_all = pickle.load(fp)
        num_of_row = df_codeid_classid_ast_all.shape[0]
        num_of_column = df_codeid_classid_ast_all.shape[1]
        for i in range(num_of_row):  # 对于每个树来说
            # if i % 1000 == 0:
            #     print(i)
            print(i)

            Codeid = df_codeid_classid_ast_all.loc[i, 'Codeid']
            Classid = df_codeid_classid_ast_all.loc[i, 'Classid']
            AST = df_codeid_classid_ast_all.loc[i, 'AST']

            print(Codeid)

            newtree = AnyNode(token=None, data=None, parent=None, ifRoot=True)
            createtree(newtree, AST, parent_ct=None)

            dict_node_id_tokenid = {} 
            anytree_node_id_order(newtree, dict_token_id, dict_node_id_tokenid)

            ##########################################################################################
            etype = []
            src = []
            tgt = []

            ##########################################################################################
            # 结构边
            ##########################################################################################
            edgesrc = []
            edgetgt = []
            getnodeandedge(newtree, edgesrc, edgetgt)

            # 双向的结构边
            edgesrc_new = edgesrc + edgetgt
            edgetgt_new = edgetgt + edgesrc

            for _ in range(len(edgesrc_new)):
                etype.append(0)
            src = src + edgesrc_new
            tgt = tgt + edgetgt_new

            ##########################################################################################
            # StateFlow边
            ##########################################################################################

            edgesrc_StateFlow, edgetgt_StateFlow = getnodeandedge_StateFlow(newtree)
            for _ in range(len(edgesrc_StateFlow)):
                etype.append(1)
            src = src + edgesrc_StateFlow
            tgt = tgt + edgetgt_StateFlow
            ##########################################################################################
            # ControlFlow边
            ##########################################################################################

            edgesrc_ControlFlow, edgetgt_ControlFlow = getnodeandedge_ControlFlow(newtree)
            for _ in range(len(edgesrc_ControlFlow)):
                etype.append(2)
            src = src + edgesrc_ControlFlow
            tgt = tgt + edgetgt_ControlFlow

            ##########################################################################################
            # DefUseFlow边
            ##########################################################################################
            edgesrc_DefUseFlow, edgetgt_DefUseFlow = getnodeandedge_DefUseFlow(newtree)
            for _ in range(len(edgesrc_DefUseFlow)):
                etype.append(3)
            src = src + edgesrc_DefUseFlow
            tgt = tgt + edgetgt_DefUseFlow
            ##########################################################################################
            Graph = dgl.graph((src, tgt))

            # Word2Vec
            Node_src_tgt = set(edgesrc + edgetgt)
            w2v_list = []
            for Node_id in range(len(Node_src_tgt)):
                token_id = dict_node_id_tokenid[Node_id]
                np_tmp = w2v.wv[dict_id_token[str(token_id)]]
                np_tmp_ = pd.np.copy(np_tmp)
                np_tmp_ = torch.from_numpy(np_tmp_).unsqueeze(0)
                w2v_list.append(np_tmp_)
            feature = torch.cat(w2v_list, dim=0)



            dict_tmp = {"Codeid": Codeid, "Classid": Classid, "Graph": Graph, "etype": etype, "Node_feature": feature}

            df_Codeid_Classid_GraphDS_all.loc[len(df_Codeid_Classid_GraphDS_all)] = dict_tmp

        df_Codeid_Classid_GraphDS_all.to_pickle(df_Codeid_Classid_GraphDS_all_path)
        print('*' * 20)
