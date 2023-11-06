### This code doesn't work


import os
from tqdm import tqdm
import subprocess

target_dir = "./binaries_to_test"

target_binary_dir = "./ida_preprocessing/target_binary"

logfile = "./log.txt"

for root, subfolder, files in os.walk(target_dir):
    for file in tqdm(files):
        # print(file)
        target = os.path.join(target_binary_dir, file)
        
        # copy file to target
        os.system("cp {} {}".format(os.path.join(root, file), target))
        
        # run ida
        os.system(f'idat64 -B -S"./ida_preprocessing/ida.py" {target}')
        
        # run bpe
        os.system(f"python ./ida_preprocessing/simple_bpe_normalization.py")
        
        # convert txt to json
        os.system("python txtToJson.py")
        
        # inference
        os.system("python inference.py")
        
        # run evaluation.py copy the results
        res = subprocess.check_output("python evaluation.py", shell=True)
        
        with open(logfile, "a") as f:
            f.write(target + "\n")
            f.write(res.decode("utf-8"))
            f.write("\n")
        
        # remove once done
        os.system("rm {}".format(target))
        os.system("rm {}".format(target+".dmp.gz"))
        os.system("rm {}".format(target+".i64"))
        
        
