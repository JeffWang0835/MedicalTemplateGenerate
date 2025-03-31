import json
import pandas as pd

data_path = [
    "./data/Chinese-medical-dialogue-data/Data_数据/IM_内科/内科5000-33000.csv",
    "./data/Chinese-medical-dialogue-data/Data_数据/Oncology_肿瘤科/肿瘤科5-10000.csv",
    "./data/Chinese-medical-dialogue-data/Data_数据/Pediatric_儿科/儿科5-14000.csv",
    "./data/Chinese-medical-dialogue-data/Data_数据/Surgical_外科/外科5-14000.csv",
]

train_json_path = "./data/train.json"
val_json_path = "./data/val.json"
# 每个数据取 10000 条作为训练
train_size = 10000
# 每个数据取 2000 条作为验证
val_size = 2000


def qa_process():
    train_f = open(train_json_path, "a", encoding='utf-8')
    val_f = open(val_json_path, "a", encoding='utf-8')
    for path in data_path:
        data = pd.read_csv(path, encoding='ANSI')
        train_count = 0
        val_count = 0
        for index, row in data.iterrows():
            question = row["ask"]
            answer = row["answer"]
            line = {
                "question": question,
                "answer": answer
            }
            line = json.dumps(line, ensure_ascii=False)
            if train_count < train_size:
                train_f.write(line + "\n")
                train_count = train_count + 1
            elif val_count < val_size:
                val_f.write(line + "\n")
                val_count = val_count + 1
            else:
                break
    print("数据处理完毕！")
    train_f.close()
    val_f.close()


if __name__ == '__main__':
    qa_process()


