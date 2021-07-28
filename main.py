from argparse import ArgumentParser, SUPPRESS
import sys
from ocr import OCR
import os
import pandas as pd
import hashlib
from tqdm import tqdm
import pathlib
import cv2
import warnings

def do_process(src, dst, data):
    warnings.filterwarnings("ignore")
    images = []
    #root = os.path.join(pathlib.Path().resolve(),'src')
    root= src
    for root, dirs, files in os.walk(root):
        if len(files) > 0:
            for i in files:
                images.append(os.path.join(root, i))

    df = pd.DataFrame(columns=['fio', 'teacher', 'level', 'sum', 'err', 'filename'])
    pbar = tqdm(images)
    for i, file in enumerate(pbar):
        pbar.set_description("Processing %s" % file)
        try:
            l = OCR(file, data)
        except:
            df = df.append(
                pd.DataFrame([['-', '-', 0, 0, '', file]], columns=['fio', 'teacher', 'level', 'sum', 'err', 'filename']))
        else:
            teacher = l.getTeacher()
            r = l.doCheck()
            newf = hashlib.md5(file.encode('utf-8')).hexdigest() + '.jpeg'
            #cv2.imwrite('dst/' + newf, l.colorImg)
            cv2.imwrite(dst + newf, l.colorImg)
            df = df.append(pd.DataFrame([[r[0], teacher, r[1], r[2], r[3], newf]],
                                        columns=['fio', 'teacher', 'level', 'sum', 'err', 'filename']))
        if i % 5 == 0:
            df.to_csv('result.csv', sep='\t', encoding='utf-8')
    df.to_csv('result.csv', sep='\t', encoding='utf-8')

def main():
    parser = ArgumentParser(description='OCR for school ',
                            add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-s', '--source',
                      help='Path to source images folder.',
                      type=str, default='./src/')
    args.add_argument('-d', '--destination',
                      help='Path to destination folder.',
                      type=str, default='./dst/')
    args.add_argument('-b', '--binary',
                      help='Path to models data folder.',
                      type=str, default='./data/')
    args = parser.parse_args()

    source_path= args.source
    destination_path=  args.destination
    data_path= args.binary
    
    if not os.path.exists(source_path):
        print('Error: Source folder not exists.') 
        sys.exit(0)
    if not os.path.exists(destination_path):
        os.mkdir(destination_path)
    data_files = ['answ.dat','comnist.dat','digits.dat']
    if not os.path.exists(data_path):
        for x in data_files:  
            if not os.path.exists(data_path+x):
                print(f'Error: Binary files not exists. {x}') 
                sys.exit(0)
    
    do_process(source_path, destination_path, data_path)
if __name__ == '__main__':
    main()

