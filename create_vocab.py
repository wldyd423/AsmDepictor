from torchtext.data import Field, TabularDataset
import torch

if __name__ == "__main__":
    max_token_seq_len = 300
    
    global tokenize
    tokenize = lambda x : x.split()
    
    train_json_dir = "./dataset/train.json"
    test_json_dir = "./dataset/test.json"
    
    code = Field(sequential=True,
                 use_vocab=True,
                 tokenize=tokenize,
                 lower=True,
                 pad_token='<pad>',
                 fix_length=max_token_seq_len)
    
    text = Field(sequential=True,
                 use_vocab=True,
                 tokenize=tokenize,
                 lower=True,
                 init_token='<sos>',
                 eos_token='<eos>',
                 pad_token='<pad>',
                 fix_length=max_token_seq_len)
    
    fields = {'Code' : ('code', code), 'Text' : ('text', text)}
    train_data, test_data = TabularDataset.splits(path='',
                                                  train=train_json_dir,
                                                  test=test_json_dir,
                                                  format='json',
                                                  fields=fields)
    
    code.build_vocab(train_data.code, train_data.text, min_freq=2)
    text.build_vocab(train_data.code, train_data.text, min_freq=0)
    
    # print(code.vocab)
    
    torch.save(code.vocab, './dataset/code_vocab.pt')
    torch.save(text.vocab, './dataset/text_vocab.pt')