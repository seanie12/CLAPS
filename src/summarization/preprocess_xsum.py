import json
import os

from tqdm import tqdm


def extract_doc_sum(lines):
    is_summary = False
    is_document = False
    summary = None
    document = list()
    for line in lines:
        line = line.strip()
        if "FIRST-SENTENCE" in line:
            is_summary = True
            continue
        elif "RESTBODY" in line:
            is_document = True
            continue

        if is_summary:
            summary = line
            is_summary = False
        elif is_document:
            document.append(line)
    assert summary is not None
    assert len(document) > 0

    document = " ".join(document)

    return document, summary

def preprocess(id_file, output_dir):
    with open(id_file) as f:
        all_ids = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    for split in all_ids.keys():
        file_dir = os.path.join(output_dir, split)
    
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        
        ids = all_ids[split]
        output_file = os.path.join(output_dir, "{}.jsonl".format(split))
        art_path = os.path.join(file_dir, "articles.txt")
        abs_path = os.path.join(file_dir, "references.txt")
        
        fw = open(output_file, "w")
        abs_file = open(abs_path, "w")
        art_file = open(art_path, "w")
        
        for file_name in tqdm(os.listdir("./data/bbc-summary-data")):

            with open(os.path.join("./data/bbc-summary-data", file_name)) as f:
                file_id = file_name.split(".")[0]

                if file_id not in ids:
                    continue
                lines = f.readlines()
                document, summary = extract_doc_sum(lines)
                json_dict = {"id": file_id,
                            "article": document,
                            "abstract": summary}
                fw.write(json.dumps(json_dict) + "\n")
                fw.flush()
                
                art_file.write(document.strip() + "\n")
                abs_file.write(summary.strip() + "\n")
        fw.close()

def get_all_articles(input_dir):
    json_dict = dict()
    for file_name in os.listdir(input_dir):
        with open(os.path.join(input_dir, file_name)) as f:
            for line in f:
                items = json.loads(line)
                json_dict[items["id"]] = items["article"]
    
    return json_dict
    

if __name__ == "__main__":
    id_file = "./data/ids.json"   
    output_dir = "./data/xsum"
    preprocess(id_file, output_dir)
