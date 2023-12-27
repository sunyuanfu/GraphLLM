import random
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd
import json
import numpy as np
import os
import time
import json
from llamaapi import LlamaAPI
import threading
FILE = 'dataset/ogbn_products_orig/ogbn-products.csv'

arxiv_natural_lang_mapping = {
    'cs.AI': 'Artificial Intelligence',
    'cs.CL': 'Computation and Language',
    'cs.CC': 'Computational Complexity',
    'cs.CE': 'Computational Engineering, Finance, and Science',
    'cs.CG': 'Computational Geometry',
    'cs.GT': 'Computer Science and Game Theory',
    'cs.CV': 'Computer Vision and Pattern Recognition',
    'cs.CY': 'Computers and Society',
    'cs.CR': 'Cryptography and Security',
    'cs.DS': 'Data Structures and Algorithms',
    'cs.DB': 'Databases',
    'cs.DL': 'Digital Libraries',
    'cs.DM': 'Discrete Mathematics',
    'cs.DC': 'Distributed, Parallel, and Cluster Computing',
    'cs.ET': 'Emerging Technologies',
    'cs.FL': 'Formal Languages and Automata Theory',
    'cs.GL': 'General Literature',
    'cs.GR': 'Graphics',
    'cs.AR': 'Hardware Architecture',
    'cs.HC': 'Human-Computer Interaction',
    'cs.IR': 'Information Retrieval',
    'cs.IT': 'Information Theory',
    'cs.LO': 'Logic in Computer Science',
    'cs.LG': 'Machine Learning',
    'cs.MS': 'Mathematical Software',
    'cs.MA': 'Multiagent Systems',
    'cs.MM': 'Multimedia',
    'cs.NI': 'Networking and Internet Architecture',
    'cs.NE': 'Neural and Evolutionary Computing',
    'cs.NA': 'Numerical Analysis',
    'cs.OS': 'Operating Systems',
    'cs.OH': 'Other Computer Science',
    'cs.PF': 'Performance',
    'cs.PL': 'Programming Languages',
    'cs.RO': 'Robotics',
    'cs.SI': 'Social and Information Networks',
    'cs.SE': 'Software Engineering',
    'cs.SD': 'Sound',
    'cs.SC': 'Symbolic Computation',
    'cs.SY': 'Systems and Control'
}

original_dict = {
    'cs.DC': 5,
    'cs.RO': 27,
    'cs.SD': 25,
    'cs.MA': 11,
    'cs.CV': 16,
    'cs.CC': 9,
    'cs.AR': 15,
    'cs.CL': 30,
    'cs.LG': 24,
    'cs.IT': 28,
    'cs.GT': 36,
    'cs.AI': 10,
    'cs.CR': 4,
    'cs.NI': 8,
    'cs.PL': 22,
    'cs.DS': 34,
    'cs.SE': 23,
    'cs.OH': 21,
    'cs.ET': 18,
    'cs.IR': 31,
    'cs.HC': 6,
    'cs.GR': 17,
    'cs.DB': 37,
    'cs.CY': 3,
    'cs.NE': 13,
    'cs.CG': 20,
    'cs.MM': 1,
    'cs.DL': 38,
    'cs.LO': 2,
    'cs.SI': 26,
    'cs.CE': 7,
    'cs.DM': 39,
    'cs.FL': 33,
    'cs.PF': 29,
    'cs.OS': 35,
    'cs.MS': 32,
    'cs.SC': 14,
    'cs.GL': 12
}

def _process():
    if os.path.isfile(FILE):
        return

    print("Processing raw text...")

    data = []
    files = ['dataset/ogbn_products/Amazon-3M.raw/trn.json',
             'dataset/ogbn_products/Amazon-3M.raw/tst.json']
    for file in files:
        with open(file) as f:
            for line in f:
                data.append(json.loads(line))

    df = pd.DataFrame(data)
    df.set_index('uid', inplace=True)

    nodeidx2asin = pd.read_csv(
        'dataset/ogbn_products/mapping/nodeidx2asin.csv.gz', compression='gzip')

    dataset = PygNodePropPredDataset(
        name='ogbn-products', transform=T.ToSparseTensor())
    graph = dataset[0]
    graph.n_id = np.arange(graph.num_nodes)
    graph.n_asin = nodeidx2asin.loc[graph.n_id]['asin'].values

    graph_df = df.loc[graph.n_asin]
    graph_df['nid'] = graph.n_id
    graph_df.reset_index(inplace=True)

    if not os.path.isdir('dataset/ogbn_products_orig'):
        os.mkdir('dataset/ogbn_products_orig')
    pd.DataFrame.to_csv(graph_df, FILE,
                        index=False, columns=['uid', 'nid', 'title', 'content'])


import dgl
from torch.utils.data import Dataset as TorchDataset

# convert PyG dataset to DGL dataset


class CustomDGLDataset(TorchDataset):
    def __init__(self, name, pyg_data):
        self.name = name
        self.pyg_data = pyg_data

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        data = self.pyg_data
        g = dgl.DGLGraph()
        if self.name == 'ogbn-arxiv' or self.name == 'ogbn-products':
            edge_index = data.edge_index.to_torch_sparse_coo_tensor().coalesce().indices()
        else:
            edge_index = data.edge_index
        g.add_nodes(data.num_nodes)
        g.add_edges(edge_index[0], edge_index[1])

        if data.edge_attr is not None:
            g.edata['feat'] = torch.FloatTensor(data.edge_attr)
        if self.name == 'ogbn-arxiv' or self.name == 'ogbn-products':
            g = dgl.to_bidirected(g)
            print(
                f"Using GAT based methods,total edges before adding self-loop {g.number_of_edges()}")
            g = g.remove_self_loop().add_self_loop()
            print(f"Total edges after adding self-loop {g.number_of_edges()}")
        if data.x is not None:
            g.ndata['feat'] = torch.FloatTensor(data.x)
        g.ndata['label'] = torch.LongTensor(data.y)
        return g

    @property
    def train_mask(self):
        return self.pyg_data.train_mask

    @property
    def val_mask(self):
        return self.pyg_data.val_mask

    @property
    def test_mask(self):
        return self.pyg_data.test_mask


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['node_id'] = idx
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
    
def get_raw_text_products(use_text=False, seed=0):
    data = torch.load('/gpfsnyu/scratch/ys6310/TAPE/dataset/ogbn_products/ogbn-products_subset.pt')
    text = pd.read_csv('/gpfsnyu/scratch/ys6310/TAPE/dataset/ogbn_products_orig/ogbn-products_subset.csv')
    text = [f'Product:{ti}; Description: {cont}\n'for ti,
            cont in zip(text['title'], text['content'])]

    data.edge_index = data.adj_t.to_symmetric()

    if not use_text:
        return data, None

    return data, text

def get_raw_text_arxiv_2023(use_text=False, seed=0):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    data = torch.load('/gpfsnyu/scratch/ys6310/TAPE/dataset/arxiv_2023/graph.pt')

    # split data
    data.num_nodes = len(data.y)
    num_nodes = data.num_nodes
    node_id = np.arange(num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(num_nodes * 0.6):int(num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(num_nodes * 0.8):int(num_nodes * 0.8)+100])

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(num_nodes)])

    # data.edge_index = data.adj_t.to_symmetric()
    if not use_text:
        return data, None

    df = pd.read_csv('/gpfsnyu/scratch/ys6310/TAPE/dataset/arxiv_2023_orig/paper_info.csv')
    text = []
    for ti, ab in zip(df['title'], df['abstract']):
        text.append(f'Title: {ti}\nAbstract: {ab}')
        # text.append((ti, ab))
    return data, text

def load_data(dataset, use_dgl=False, use_text=False, seed=0):
    if dataset == 'arxiv_2023':
        num_classes = 40
    else:
        exit(f'Error: Dataset {dataset} not supported')

    if not use_text:
        data, _ = get_raw_text_arxiv_2023(use_text=False, seed=seed)
        if use_dgl:
            data = CustomDGLDataset(dataset, data)
        return data, num_classes

    else:
        data, text = get_raw_text_arxiv_2023(use_text=True, seed=seed)

    return data, num_classes, text

def get_llama_predictions(llama, text_data):
    llama_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f'"{text_data} Give the most likely (only one) arXiv CS sub-category of this paper directly. Response to me in the form "cs.XX"."'},
    ]
    llama_messages[1]["content"] += "Your answer should be chosen from {}.".format(', '.join(['"{}"'.format(arxiv_natural_lang_mapping[key]) for key in arxiv_natural_lang_mapping.keys()]))
    # print(llama_messages[1]["content"])
    api_request_json = {"model": "llama-13b-chat", "messages": llama_messages}
    response = llama.run(api_request_json)
    response_data = response.json()
    prediction = response_data['choices'][0]['message']['content']
    return prediction

def get_category_index(category_abbr, original_dict):

    return original_dict.get(category_abbr, None)

def extract_category_abbreviation(response_text):
    prefix = 'cs.'
    start_index = response_text.find(prefix)

    if start_index != -1 and len(response_text) > start_index + len(prefix) + 2:
        # 提取 cs. 后面两个字符
        category_abbreviation = response_text[start_index + len(prefix):start_index + len(prefix) + 2]
        return prefix + category_abbreviation

    return None

def evaluate_llama_predictions(llama, data, text):
    correct_predictions = 0
    total_samples = len(data.test_id)

    for node_index in data.test_id:
        text_data = text[node_index]
        response = get_llama_predictions(llama, text_data)
        # Assuming data.y contains ground truth labels
        ground_truth_label = data.y[node_index].item()
        prediction = extract_category_abbreviation(response)
        if prediction is not None:
         # Check if the prediction is correct
           y_pred = get_category_index(prediction,original_dict)
           if y_pred is not None:
              if y_pred == ground_truth_label:
               correct_predictions += 1

    accuracy = correct_predictions / total_samples
    return accuracy

def evaluate_llama_predictions_multi_thread(llama, data, text):
    # 存储结果的全局变量
    results = []

    # 锁用于在多线程中同步对共享资源的访问
    results_lock = threading.Lock()

    def process_node(node_index, text_data):
        response = get_llama_predictions(llama, text_data)
        ground_truth_label = data.y[node_index].item()
        prediction = extract_category_abbreviation(response)

        if prediction is not None:
            y_pred = get_category_index(prediction, original_dict)
            if y_pred is not None and y_pred == ground_truth_label:
                with results_lock:
                    results.append(True)
            else:
                with results_lock:
                    results.append(False)

    # 通过多线程并行处理
    threads = []
    for node_index in data.test_id:
        text_data = text[node_index]
        thread = threading.Thread(target=process_node, args=(node_index, text_data))
        threads.append(thread)
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    # 计算准确率
    accuracy = sum(results) / len(results)
    return accuracy

if __name__ == '__main__':
    llama = LlamaAPI('LL-4AQUhX62d88zD7OrnrKactz1Q6HBnH3YQ6HiE3yeEAMidnntqGi8V62nnIli6nbF')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    data, num_classes, text = load_data(dataset='arxiv_2023',use_text=True)
    print(data)
    # # response = get_llama_predictions(llama,text[2])
    # # prediction = extract_category_abbreviation(response)
    # # print(response)
    # # print(prediction)
    # print(get_category_index('cs.DC',original_dict))
    accuracy = evaluate_llama_predictions_multi_thread(llama, data, text)
    print(accuracy)
 