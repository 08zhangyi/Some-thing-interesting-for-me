import json

def split(row):
    '''split 25 ||| 28 ||| 李嬷嬷and return
    istart, iend, speaker'''
    tokens = row.split('|||')
    istart = int(tokens[0])
    iend = int(tokens[1])
    ## squad requires (-1, -1) if there is no answer
    if istart == -1: iend = -1
    speaker = tokens[2].strip()
    return [istart, iend, speaker]

def read_samples(context_file, result_file, title="红楼梦"):
    res = dict()
    res["title"] = title
    res["paragraphs"] = []
    with open(result_file, "r") as fin:
        lines = fin.readlines()
        labels = [split(line) for line in lines]
    with open(context_file, "r") as fin:
        contexts = fin.readlines()
    for i in range(len(contexts)):
        answer = {"answer_start":labels[i][0], "text":labels[i][2]}
        answers = [answer]
        para_entry = dict()
        para_entry["context"] = contexts[i]
        qas = [{"answers": answers, 
            "question": "说下一句话的人是谁？",
            "id": i}]
        para_entry["qas"] = qas
        res["paragraphs"].append(para_entry)
    return res


def training_example():
    res = read_samples("training_sentence.csv", "training_labels.csv")
    out = {"data":[res], "version":"chinese_squad_v1.0"}
    with open("chinese_speaker_squad.json", "w") as fout:
        fout.write(json.dumps(out, ensure_ascii=False))

def test_example():
    #res = read_samples("training_sentence.csv", "training_labels.csv")
    res = dict()
    res["title"] = "hongloumeng"
    res["paragraphs"] = []
    from honglou import talks as contexts
    for i in range(len(contexts)):
        answer = {"answer_start":-1, "text":""}
        answers = [answer]
        para_entry = dict()
        para_entry["context"] = contexts[i]['context']
        qas = [{"answers": answers, 
            "question": "说下一句话的人是谁？",
            "id": i}]
        para_entry["qas"] = qas
        res["paragraphs"].append(para_entry)
    out = {"data":[res], "version":"chinese_squad_v1.0"}
    with open("chinese_speaker_squad_valid.json", "w") as fout:
        fout.write(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    #test_example()
    training_example()

