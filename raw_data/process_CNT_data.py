import numpy as np
import re
#import pandas as pd

filename = "./DoS_CNT_11_0.dat"

lines = []
with open(filename, "r") as file:
    i = 0
    for line in file:
        cleanline = line.strip()
        if not cleanline :
            continue
        cleanline = re.sub("\s+", ",", cleanline)
        lines.append(cleanline)
           
outfilename = "./DoS_CNT_11_0_clean.dat"
with open(outfilename, "w") as file:
    for line in lines:
        file.write(line + "\n")

