import pandas as pd
from sklearn.utils import shuffle
from random_seed import set_random_seed
set_random_seed(123)

class TxtToJson():
    def __init__(self, src_dir, target_dir, max_tok_seq_len, json_path="./dataset/train.json"):
        self.src_dir = src_dir
        self.target_dir = target_dir
        self.max_tok_seq_len = max_tok_seq_len
        self.json_path = json_path
        
        self.preprocess()
        self.toJson()
    
    def preprocess(self):
        src_data = open(self.src_dir, encoding="utf-8").read().split('\n')
        tgt_data = open(self.target_dir, encoding="utf-8").read().split('\n')
        print(f"[+] Finished Reading {self.src_dir} and {self.target_dir} Length of data: ", len(src_data))
        
        
        src_text_toc = [line.split() for line in src_data]
        src_tok_concat = [" ".join(tok[0:max_tok_seq_len]) for tok in src_text_toc]
        
        print(f"[+] Finished Tokenizing Source")
        
        tgt_text_toc = [line.split() for line in tgt_data]
        tgt_tok_concat = [" ".join(tok[0:max_tok_seq_len]) for tok in tgt_text_toc]
        
        print(f"[+] Finished Tokenizing Target")
        
        raw_data = {'Code': [line for line in src_tok_concat],
                    'Text': [line for line in tgt_tok_concat]}
        df = pd.DataFrame(raw_data, columns=["Code", "Text"])
        
        self.df = shuffle(df)
        print(self.df)
    
    def toJson(self):
        self.df.to_json(self.json_path, orient="records", lines=True)
        print(f"[+] Finished Saving to {self.json_path}")

if __name__=="__main__":
    # train_src_dir = "./dataset/train_source.txt"
    # train_target_dir = "./dataset/train_target.txt"
    # max_tok_seq_len = 300
    
    # TxtToJson(train_src_dir, train_target_dir, max_tok_seq_len, json_path="./dataset/train.json")
    
    src_dir = "./dataset/test_bpe_source.txt"
    target_dir = "./dataset/test_bpe_target.txt"
    max_tok_seq_len = 300
    
    TxtToJson(src_dir, target_dir, max_tok_seq_len, json_path="./dataset/test.json")