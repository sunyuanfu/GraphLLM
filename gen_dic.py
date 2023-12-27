# import pandas as pd

# # 读取 CSV 文件
# csv_file_path = '/gpfsnyu/scratch/ys6310/TAPE/dataset/arxiv_2023_orig/paper_info.csv'
# df = pd.read_csv(csv_file_path)

# # 获取唯一的类别和标签
# unique_categories = df['category'].unique()
# unique_labels = df['label'].unique()

# # 创建类别到标签的映射字典
# category_to_label_mapping = {category: label for category, label in zip(unique_categories, unique_labels)}

# # 创建标签到类别的映射字典
# label_to_category_mapping = {label: category for category, label in zip(unique_categories, unique_labels)}

# # 打印字典元素
# print(category_to_label_mapping)

# # 如果需要按照指定格式输出
# formatted_mapping = {f'cs.{label}': category for label, category in label_to_category_mapping.items()}
# print(formatted_mapping)
# 已知的字典
original_dict = {
    'Distributed, Parallel, and Cluster Computing (cs.DC)': 5,
    'Robotics (cs.RO)': 27,
    'Sound (cs.SD)': 25,
    'Multiagent Systems (cs.MA)': 11,
    'Computer Vision and Pattern Recognition (cs.CV)': 16,
    'Computational Complexity (cs.CC)': 9,
    'Hardware Architecture (cs.AR)': 15,
    'Computation and Language (cs.CL)': 30,
    'Machine Learning (cs.LG)': 24,
    'Information Theory (cs.IT)': 28,
    'Computer Science and Game Theory (cs.GT)': 36,
    'Artificial Intelligence (cs.AI)': 10,
    'Cryptography and Security (cs.CR)': 4,
    'Networking and Internet Architecture (cs.NI)': 8,
    'Programming Languages (cs.PL)': 22,
    'Data Structures and Algorithms (cs.DS)': 34,
    'Software Engineering (cs.SE)': 23,
    'Other Computer Science (cs.OH)': 21,
    'Emerging Technologies (cs.ET)': 18,
    'Information Retrieval (cs.IR)': 31,
    'Human-Computer Interaction (cs.HC)': 6,
    'Graphics (cs.GR)': 17,
    'Databases (cs.DB)': 37,
    'Computers and Society (cs.CY)': 3,
    'Neural and Evolutionary Computing (cs.NE)': 13,
    'Computational Geometry (cs.CG)': 20,
    'Multimedia (cs.MM)': 1,
    'Digital Libraries (cs.DL)': 38,
    'Logic in Computer Science (cs.LO)': 2,
    'Social and Information Networks (cs.SI)': 26,
    'Computational Engineering, Finance, and Science (cs.CE)': 7,
    'Discrete Mathematics (cs.DM)': 39,
    'Formal Languages and Automata Theory (cs.FL)': 33,
    'Performance (cs.PF)': 29,
    'Operating Systems (cs.OS)': 35,
    'Mathematical Software (cs.MS)': 32,
    'Symbolic Computation (cs.SC)': 14,
    'General Literature (cs.GL)': 12
}

# 按照种类编号由小到大重新排列
sorted_dict = dict(sorted(original_dict.items(), key=lambda x: x[1]))

# 生成新的字典按'cs.AI': 'Artificial Intelligence'的形式
new_dict = {f'cs.{label}': category for category, label in sorted_dict.items()}

# 打印新字典
print(new_dict)
