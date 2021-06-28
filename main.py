
from ocr import OCR
import os
import pandas as pd
import hashlib
from tqdm import tqdm
import pathlib
import cv2
import warnings

def main():
    warnings.filterwarnings("ignore")
    images = []
    root = os.path.join(pathlib.Path().resolve(),'src')
    for root, dirs, files in os.walk(root):
        if len(files) > 0:
            for i in files:
                images.append(os.path.join(root, i))

    df = pd.DataFrame(columns=['fio', 'teacher', 'level', 'sum', 'err', 'filename'])
    pbar = tqdm(images)
    for i, file in enumerate(pbar):
        pbar.set_description("Processing %s" % file)
        teacher = file[4:6]
        l = OCR(file)
        try:
            l = OCR(file)
        except:
            df = df.append(
                pd.DataFrame([['-', '-', 0, 0, '', file]], columns=['fio', 'teacher', 'level', 'sum', 'err', 'filename']))
        else:
            print(l.getTeacher())    
            r = l.doCheck()
            newf = hashlib.md5(file.encode('utf-8')).hexdigest() + '.jpeg'
            cv2.imwrite('dst/' + newf, l.colorImg)
            df = df.append(pd.DataFrame([[r[0], teacher, r[1], r[2], r[3], newf]],
                                        columns=['fio', 'teacher', 'level', 'sum', 'err', 'filename']))
        if i % 5 == 0:
            df.to_csv('result.csv', sep='\t', encoding='utf-8')
    df.to_csv('result.csv', sep='\t', encoding='utf-8')

if __name__ == '__main__':
    main()
