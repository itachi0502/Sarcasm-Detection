from transformers import AutoTokenizer
import shap
import torch

def read_conll(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split('\t')
            id_, label, text = parts
            data.append((id_, int(label), text))
    return data

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-2021-124m-irony")

# 读取数据
data = read_conll("/home/wzl/project/kbner/kb/datasets/iron_context/test_data_processed.txt")

# 将文本数据转换为模型输入
inputs = tokenizer([text for _, _, text in data], return_tensors="pt", padding=True, truncation=True)

# 选择参考数据集和输入数据集
reference_dataset = [inputs["input_ids"][:100], inputs["attention_mask"][:100]]  # 假设我们使用前100个数据点作为参考数据集
input_dataset = [inputs["input_ids"][100:200], inputs["attention_mask"][100:200]]  # 假设我们想要解释第100到200个数据点

# 加载模型
model = torch.load("/home/wzl/project/bert4torch/output1/best-model.pt")

# 创建解释器
explainer = shap.DeepExplainer(model, reference_dataset)

# 计算SHAP值
shap_values = explainer.shap_values(input_dataset)

# 可视化SHAP值
shap.summary_plot(shap_values, input_dataset)
