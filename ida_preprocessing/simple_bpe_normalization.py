import os
import util
import unit
from capstone import *
from tqdm import tqdm

class Dump():
    def __init__(self, binary_name, dmp_path, bin_info):
        self.dmp_path = dmp_path
        self.binary_name = binary_name
        
        self.funcs = util.load_from_dmp(self.dmp_path)
        self.bin_info = bin_info
        # print(tmp)
        
        self.md = Cs(CS_ARCH_X86, CS_MODE_64)
        self.func_dict = dict()
        self.build_text()
        
   
    def build_text(self):
        f_addrs = sorted(self.funcs.keys())
        F_prev = None
        
        bar = util.ProgressBar(len(f_addrs), name="Creating Plain Text...: " + self.bin_info.bin_name)
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
            bar += 1
        with open("./dataset/test_source.txt", mode='w', encoding='utf-8') as f:
            for func in tqdm(self.func_dict):
                f.write(','.join(self.func_dict[func]) + '\n')
        
        with open("./dataset/test_target.txt", mode='w', encoding='utf-8') as f:
            for func in tqdm(self.func_dict):
                name = func.replace("__", "")
                name = name.replace("_", " ")
                # if first char is space, remove it
                if name[0] == ' ':
                    name = name[1:]
                f.write(name + '\n')
            
            
                        

def parse(binname):
    # print(binname)
    tmp, _, _, _ = binname.split('__')
    compiler, optlevel = tmp.split('-')
    # print(compiler, optlevel)
    return compiler, optlevel

if __name__ == "__main__":
    binary_dir = "./sample_binary"
    
    for root, dirs, files in os.walk(binary_dir):
        for file in files:
            
            # if ends with .dmp.gz
            binname, ext = os.path.splitext(file)
            if ext == '.gz':
                binname, _ = os.path.splitext(binname)
                # print(binname, file)
                compiler, optlevel = parse(binname)
                bin_info = unit.Binary_Info(os.path.join(root, binname))
                bin_info.compiler_info_label = compiler
                bin_info.opt_level_label = optlevel
                Dump(binname, os.path.join(root, file), bin_info)
                
                
                