import torch
import model.asmdepictor.Models as Asmdepictor
import torch.nn as nn
import json
from tqdm import tqdm
from model.asmdepictor.Translator import Translator

class Inference():
    def __init__(self, model, code_vocab, text_vocab, tokenize, max_token_seq_len, device, test_src, test_tgt):
        self.model = model
        self.code_vocab = code_vocab
        self.text_vocab = text_vocab
        self.tokenize = tokenize
        self.max_token_seq_len = max_token_seq_len
        self.device = device
        self.test_src = test_src
        self.test_tgt = test_tgt
        
        
        self.hypothesis_list = list()
        self.reference_list =  test_tgt
        
        self.build_hypothesis_list()
        
        self.save_hypothesis_list()
    
    def build_hypothesis_list(self):
        print("Building hypothesis list...")
        
        for src in tqdm(self.test_src):
            hypothesis = self.make_a_hypothesis_transformer(src)
            self.hypothesis_list.append(hypothesis)
    
    def make_a_hypothesis_transformer(self, src):
        input_tensor = self.sentence_to_tensor(src)
        
        translator = Translator(
            model=self.model,
            beam_size=5,
            max_seq_len=self.max_token_seq_len+3,
            src_pad_idx=self.code_vocab.stoi['<pad>'],
            trg_pad_idx=self.text_vocab.stoi['<pad>'],
            trg_bos_idx=self.text_vocab.stoi['<sos>'],
            trg_eos_idx=self.text_vocab.stoi['<eos>']).to(self.device)
        
        output_tensor = translator.translate_sentence(input_tensor)
        predict_sentence = ' '.join(self.text_vocab.itos[idx] for idx in output_tensor)
        predict_sentence = predict_sentence.replace('<sos> ', '').replace(' <eos>', '')
        return predict_sentence
        
    def sentence_to_tensor(self, sentence):
        sentence = self.tokenize(sentence)
        unk_idx = self.code_vocab.stoi['<unk>']
        pad_idx = self.code_vocab.stoi['<pad>']
        sentence_idx = [self.code_vocab.stoi.get(i, unk_idx) for i in sentence]
        
        for i in range(self.max_token_seq_len-len(sentence_idx)):
            sentence_idx.append(self.code_vocab.stoi.get(i, pad_idx))
        
        sentence_tensor = torch.tensor(sentence_idx).to(self.device)
        sentence_tensor = sentence_tensor.unsqueeze(0)
        return sentence_tensor
    
    def save_hypothesis_list(self):
        print("Saving hypothesis list...")
        with open("./predicted_output/prediction.txt", mode='w', encoding='utf-8') as f:
            for hypothesis in self.hypothesis_list:
                f.write(hypothesis + '\n')
        
        print("Saving ground ...")
        with open("./predicted_output/ground_truth.txt", mode='w', encoding='utf-8') as f:
            for reference in self.reference_list:
                f.write(reference + '\n')
        
        print("Done!")
        
        
        

if __name__ == "__main__":
    code_vocab_path = "./dataset/code_vocab.pt"
    text_vocab_path = "./dataset/text_vocab.pt"
    
    code_vocab = torch.load(code_vocab_path)
    text_vocab = torch.load(text_vocab_path)
    
    # print(code_vocab.stoi)
    model_path = "./dataset/asmdepictor_pretrained.param"
    tokenize = lambda x : x.split()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_token_seq_len = 300
    print(device)
    
    src_pad_idx = code_vocab.stoi['<pad>']
    src_vocab_size = len(code_vocab.stoi)
    trg_pad_idx = text_vocab.stoi['<pad>']
    trg_vocab_size = len(text_vocab.stoi)
    
    proj_share_weight = True
    embs_share_weight = True
    d_k = 64
    d_v = 64
    d_model = 512
    d_word_vec = 512
    d_inner_hid = 2048
    n_layers = 3
    n_head = 8
    dropout = 0.1
    scale_emb_or_prj = 'emb'
    
    model = Asmdepictor.Asmdepictor(src_vocab_size,
                                    trg_vocab_size,
                                    src_pad_idx=src_pad_idx,
                                    trg_pad_idx=trg_pad_idx,
                                    trg_emb_prj_weight_sharing=proj_share_weight,
                                    emb_src_trg_weight_sharing=embs_share_weight,
                                    d_k=d_k,
                                    d_v=d_v,
                                    d_model=d_model,
                                    d_word_vec=d_word_vec,
                                    d_inner=d_inner_hid,
                                    n_layers=n_layers,
                                    n_head=n_head,
                                    dropout=dropout,
                                    scale_emb_or_prj=scale_emb_or_prj,
                                    n_position=max_token_seq_len+3).to(device)
    
    model = nn.DataParallel(model)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    
    test_data = list()
    for line in open("./dataset/test.json", mode='r', encoding='utf-8'):
        test_data.append(json.loads(line))
        
    test_src = list()
    test_tgt = list()
    for d in test_data:
        test_src.append(d['Code'].lower())
        test_tgt.append(d['Text'].lower())
    
    Inference = Inference(model, code_vocab, text_vocab, tokenize, max_token_seq_len, device, test_src, test_tgt)