import os


import re

# Step 1: 读取数据文件
def read_files(text_file, ann_file):
    # 读取文本文件
    with open(text_file, "r", encoding="utf-8") as tf:
        text = tf.read()
    
    # 读取注解文件
    annotations = []
    with open(ann_file, "r", encoding="utf-8") as af:

            # print("ann_file: ", ann_file)
            for line in af:
                parts = line.strip().split("\t")
                # print(parts)
                if len(parts) >= 3:
                    entity_type, span_range = parts[1].split()[0], parts[1].split()[1:]
                    # print(span_range)
                    if len(span_range) >2 and ';' in span_range[1]:
                        start = span_range[0]
                        for i in range(len(span_range)-2):
                            if i == 0:
                                continue
                            span_range_multi = span_range[i].split(';')
                            span_start = int(start)
                            span_end = int(span_range_multi[0])
                            annotations.append({"type": entity_type, "start": span_start, "end": span_end})
                            start = span_range_multi[1]
                        end = span_range[-1]
                        annotations.append({"type": entity_type, "start": int(start), "end": int(end)})
                    elif len(span_range) > 1:
                        span_start, span_end = map(int, span_range)
                        annotations.append({"type": entity_type, "start": span_start, "end": span_end})
                
    return text, annotations

# Step 2: 将文本划分为词并记录偏移
def tokenize_text(text):
    tokens = []
    offsets = []
    for match in re.finditer(r'\S+', text):  # 匹配每个非空格单词
        tokens.append(match.group())
        offsets.append((match.start(), match.end()))
    # print("Tokens:", tokens)
    # print("Offsets:", offsets)
    return tokens, offsets

# Step 3: 根据注解生成标签
def assign_labels(tokens, offsets, annotations):
    labels = ["O"] * len(tokens)  # 初始化为非实体标签 "O"
    for annotation in annotations:
        for idx, (start, end) in enumerate(offsets):
            if start >= annotation["start"] and end <= annotation["end"]:  # 如果 token 在实体范围内
                if start == annotation["start"]:  # 实体的第一个 token
                    labels[idx] = f"B-{annotation['type']}"
                else:  # 实体内的 token
                    labels[idx] = f"I-{annotation['type']}"
    return labels

# 主处理流程
if __name__ == "__main__":
    # 输入文件路径

    text_folder_path = 'cadec/text'
    ann_folder_path = 'cadec/original'

    # with open('train.txt', 'w') as f:

    dataset = []
    for file in os.listdir(text_folder_path):
        text_file = os.path.join(text_folder_path, file)
        ann_file = os.path.join(ann_folder_path, file.replace('.txt', '.ann'))
        print(text_file, ann_file)
        # 读取文本和注解
        text, annotations = read_files(text_file, ann_file)

        # 生成 token 和其偏移
        tokens, offsets = tokenize_text(text)

        # 分配标签
        labels = assign_labels(tokens, offsets, annotations)

        # 输出结果
        # print("Tokens:", tokens)
        # print("Labels:", labels)

        dataset.append((tokens, labels))
        print(len(dataset))
    
    # with open('train.txt', 'w') as f:
    #     for tokens, labels in dataset:
    #         for token, label in zip(tokens, labels):
    #             f.write(token + '\t' + label + '\n')
    #         f.write('\n')

    with open('train_small300.txt', 'w') as f:
        for tokens, labels in dataset[:300]:
            for token, label in zip(tokens, labels):
                f.write(token + '\t' + label + '\n')
            f.write('\n')
    
    with open('train_small500.txt', 'w') as f:
        for tokens, labels in dataset[:500]:
            for token, label in zip(tokens, labels):
                f.write(token + '\t' + label + '\n')
            f.write('\n')

    # with open('train.txt', 'w') as f:
    #     for tokens, labels in dataset[:int(len(dataset)*0.72)]:
    #         for token, label in zip(tokens, labels):
    #             f.write(token + '\t' + label + '\n')
    #         f.write('\n')

    # with open('val.txt', 'w') as f:
    #     for tokens, labels in dataset[int(len(dataset)*0.72):int(len(dataset)*0.8)]:
    #         for token, label in zip(tokens, labels):
    #             f.write(token + '\t' + label + '\n')
    #         f.write('\n')
    
    # with open('test.txt', 'w') as f:
    #     for tokens, labels in dataset[int(len(dataset)*0.8):]:
    #         for token, label in zip(tokens, labels):
    #             f.write(token + '\t' + label + '\n')
    #         f.write('\n')