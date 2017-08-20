from darkflow.net.build import TFNet
import cv2
import os
import numpy as np
import pandas as pd
from skimage import io
import multiprocessing as mproc
import sys
import argparse
import traceback
import time
import datetime

options = {"model": "./cfg/yolo.cfg", "load": "./bin/yolo.weights", "threshold": 0.0, "gpu": 0.8}

tfnet = TFNet(options)

DATA_FOLDER = '../data'




parser = argparse.ArgumentParser(description='Count images')

# Required positional argument
parser.add_argument('input_file', type=str,
                    help='Input file')

parser.add_argument('-p', type=int,
                    help='Num cores')

def detect_and_count_objects(data_source, outfile, alloutfile):

    res = dict()

    for idx, row in data_source.iterrows():
        attempt = 1
        while True:
            try:
                img = io.imread(row['x'])
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                pred = tfnet.return_predict(img)
                if len(pred) > 0:
                    res[row['id']] = pred
                print("Processed image: " + str(row['id']))
                sys.stdout.flush()
            except Exception:
                print ("Error processing image: " + row['x'])
                print (traceback.format_exc())
                attempt += 1
                if attempt < 6:
                    continue
            break
    predictions = pd.DataFrame(columns=['Image', 'Object'])
    predictions_all = pd.DataFrame(columns=['Image', 'Object', 'Confidence'])

    for img in res.keys():
        preds = res[img]
        for obj in preds:
            nrow = [{'Image': img, 'Object': obj['label'], 'Confidence': obj['confidence']}]
            predictions_all = predictions_all.append(nrow, ignore_index=True)

    predictions_all.to_csv(alloutfile, header=True, index=False)


    for img in res.keys():
        preds = res[img]
        for obj in preds:
            if (obj['label'] in ['person', 'bicycle', 'motorbike', 'truck', 'car', 'bus', 'train']) and \
                ((obj['label'] == 'person' and obj['confidence'] > 0.35) or
                 (obj['label'] == 'bicycle' and obj['confidence'] > 0.18) or
                 (obj['label'] in ['car', 'truck'] and obj['confidence'] > 0.23) or
                 (obj['label'] not in ['bicycle', 'person', 'car', 'truck'] and obj['confidence'] > 0.1)
                ):
                nrow = [{'Image': img, 'Object': obj['label']}]
                predictions = predictions.append(nrow, ignore_index=True)

    predictions = predictions.groupby(['Image', 'Object']).size().reset_index(name='count')

    predictions.to_csv(outfile, header=True, index=False)

def combile_csvs(lst_filename, finalname):
    lst_result = []
    for file in lst_filename:
        if os.path.isfile(file):
            try:
                result = pd.read_csv(file, index_col=None, header=0)
                lst_result.append(result)
            except pd.errors.EmptyDataError:
                continue
    result = pd.concat(lst_result)
    result.to_csv(finalname, header=True, index=False)
    for file in lst_filename:
        os.remove(file)
    return result

def main(argv):
    input_file = ''
    ncores = 28

    args = parser.parse_args()
    inputfile = os.path.join(DATA_FOLDER, 'in', args.input_file)
    ncores = args.p

    data_source = pd.read_csv(inputfile, index_col=None, header=0)
    tic = time.time()
    print ("Launched at: " + str(datetime.now()))

    if ncores > 1:
        i = 0
        lst_subfile = []
        lst_subfile_all = []
        procs = []
        for res in np.array_split(data_source, ncores):
            lst_subfile.append(os.path.join(DATA_FOLDER, 'out', 'part_' + str(i) + '_' + os.path.basename(inputfile)))
            lst_subfile.append(os.path.join(DATA_FOLDER, 'out', 'part_all' + str(i) + '_' + os.path.basename(inputfile)))

            proc = mproc.Process(target=detect_and_count_objects,
                                 args=(res,
                                       os.path.join(DATA_FOLDER, 'out', 'part_' + str(i) + '_' + os.path.basename(inputfile)),
                                       os.path.join(DATA_FOLDER, 'out', 'part_all' + str(i) + '_' + os.path.basename(inputfile)),
                                      )
                                 )
            procs.append(proc)
            proc.start()
            i += 1

        for proc in procs:
            proc.join()

        combile_csvs(lst_subfile, os.path.join(DATA_FOLDER, 'out', 'predictions_' + os.path.basename(inputfile)))
        combile_csvs(lst_subfile_all, os.path.join(DATA_FOLDER, 'out', 'predictions_all' + os.path.basename(inputfile)))


    else:
        detect_and_count_objects(data_source,
                                 os.path.join(DATA_FOLDER, 'predictions_' + os.path.basename(inputfile)),
                                 os.path.join(DATA_FOLDER, 'predictions_all' + os.path.basename(inputfile))
                                )

    toc = time.time()
    print ("Finished at: " + str(datetime.now()))

    tac = round(toc - tic)
    (t_min, t_sec) = divmod(tac, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Time elapsed: {0} hours {1} minutes and {2} seconds'.format(t_hour, t_min, t_sec))
    
if __name__ == "__main__":
    main(sys.argv[1:])


