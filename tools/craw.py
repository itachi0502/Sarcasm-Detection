import requests
import json
import os
from bert_score import score
from transformers import BertTokenizer
import torch

# 调用 New York Times API 搜索文章
def search_articles(query, api_key):
    base_url = 'https://api.nytimes.com/svc/search/v2/articlesearch.json'
    params = {'q': query, 'api-key': api_key}
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        return json.loads(response.text)['response']['docs']
    else:
        return []

# 获取文章的前十句句子
def get_sentences(article):
    if 'lead_paragraph' in article:
        return article['lead_paragraph'].split('. ')[:10]
    else:
        return []

# 计算 BertScore
def calculate_bert_score(original_text, sentences, tokenizer, device):
    scores = []
    for sentence in sentences:
        P, R, F1 = score([original_text], [sentence], lang="en", model_type="bert-base-uncased", device=device)
        scores.append(F1.item())

    return scores

# 主函数
def main():
    api_key = '1EAsLuGJb1HZkQyGnAGCGPlwUCZNTQie'
    data_path = '/home/wzl/project/kbner/kb/datasets/iron/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for filename in ['new_train.txt', 'new_val.txt', 'test.txt']:
        with open(os.path.join(data_path, filename), 'r') as file:
            lines = file.readlines()

        new_lines = []
        for line in lines:
            id, label, text = line.strip().split('\t')
            articles = search_articles(text, api_key)

            new_line = [id, label, text]
            for article in articles:
                sentences = get_sentences(article)
                scores = calculate_bert_score(text, sentences, tokenizer, device)

                for sentence, score in zip(sentences, scores):
                    new_line.append(sentence + '+' + str(score))
                    print(new_line)
            new_lines.append('\n'.join(new_line) + '\n\n')

        with open(os.path.join(data_path, 'new_' + filename), 'w') as file:
            file.writelines(new_lines)

if __name__ == '__main__':
    main()
