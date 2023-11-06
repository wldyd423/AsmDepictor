import os
import util
import unit
from capstone import *
from tqdm import tqdm

class Dump():
    def __init__(self, binary_name, dmp_path):
        self.dmp_path = dmp_path
        self.binary_name = binary_name
        
        self.funcs = util.load_from_dmp(self.dmp_path)
        print(binary_name)
        # print(self.funcs)
        
        self.md = Cs(CS_ARCH_X86, CS_MODE_64)
        self.func_dict = dict()
        self.build_text()
        
   
    def build_text(self):
        f_addrs = sorted(self.funcs.keys())
        # print(f_addrs)
        
        # bar = util.ProgressBar(len(f_addrs), name="Creating Plain Text...:")
        for f_idx, f_addr in enumerate(f_addrs):
            F = self.funcs[f_addr]
            # print(F)
            self.func_dict[F.name] = []
            if not F.is_linker_func:
                bb_addrs = sorted(F.bbls.keys())
                for bb_addr in bb_addrs:
                    BB = F.bbls[bb_addr]
                    instn_addrs = sorted(BB.instns.keys())
                    
                    for instn_addr in instn_addrs:
                        I = BB.instns[instn_addr]
                        # print(I.cs_instr)
                        # print([x for x in self.md.disasm(I.raw_bytes, 0x0)][0])
                        cs_instr = [x for x in self.md.disasm(I.raw_bytes, 0x0)][0]
                        if not cs_instr or cs_instr.mnemonic == 'nop':
                            continue
                        
                        opcode = cs_instr.mnemonic
                        operands = cs_instr.op_str
                        if operands == '':
                            instruction = opcode
                        else:
                            instruction = opcode + ' ' + operands
                        instruction = instruction.replace(' + ', '+')
                        instruction = instruction.replace(' - ', '-')
                        instruction = instruction.replace(',', '')
                        instruction = instruction.replace(' ', '_')
                        # print(instruction)
                        self.func_dict[F.name].append(instruction)
            # bar += 1
        with open("./dataset/plain_source.txt", mode='a', encoding='utf-8') as f:
            for func in tqdm(self.func_dict):
                f.write(', '.join(self.func_dict[func]) + '\n')
        
        with open("./dataset/test_target.txt", mode='a', encoding='utf-8') as f:
            for func in tqdm(self.func_dict):
                name = func.replace("__", "")
                name = name.replace("_", " ")
                # if first char is space, remove it
                if name[0] == ' ':
                    name = name[1:]
                f.write(name + '\n')
            
            
                        



if __name__ == "__main__":
    # delete ./dataset/test_source.txt and ./dataset/test_target.txt at start
    if os.path.exists("./dataset/plain_source.txt"):
        os.remove("./dataset/plain_source.txt")
    if os.path.exists("./dataset/test_source.txt"):
        os.remove("./dataset/test_source.txt")
    if os.path.exists("./dataset/test_target.txt"):
        os.remove("./dataset/test_target.txt")
    binary_dir = "./target_binary"
    
    for root, dirs, files in os.walk(binary_dir):
        for file in files:
            
            # if ends with .dmp.gz
            binname, ext = os.path.splitext(file)
            if ext == '.gz':
                binname, _ = os.path.splitext(binname)
                print(binname, file)
                Dump(binname, os.path.join(root, file))
                
    ## run bpe
    # cmd: subword-nmt apply-bpe --codes ../vocab/pretrained_bpe_voca.voc --input ./dataset/plain_source.txt --output ./dataset/test_source.txt
    
    # os.system("subword-nmt apply-bpe --codes ../vocab/pretrained_bpe_voca.voc --input ./dataset/plain_source.txt --output ./dataset/test_source.txt")
    os.system("subword-nmt apply-bpe --codes ./vocab/pretrained_bpe_voca.voc --input ./ida_preprocessing/dataset/plain_source.txt --output ./dataset/test_source.txt")
                