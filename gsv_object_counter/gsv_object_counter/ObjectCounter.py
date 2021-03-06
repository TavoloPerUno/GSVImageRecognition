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
from datetime import datetime
import logging

logger = logging.getLogger('ObjectCounter')
options = {"model": "./cfg/yolo.cfg", "load": "./bin/yolo.weights", "threshold": 0.0, "gpu": 0.8}

tfnet = TFNet(options)

DATA_FOLDER = '../data'


parser = argparse.ArgumentParser(description='Count images')

# Required positional argument
parser.add_argument('input_file', type=str,
                    help='Input file')

parser.add_argument('-opt', type=str,
                    help='Option')

parser.add_argument('-p', type=int,
                    help='Num cores')

def detect_and_count_objects(data_source, outfile, alloutfile):

    res = dict()
    predictions = pd.DataFrame(columns=['Image', 'Object', 'count', 'Threshold'])
    predictions_all = pd.DataFrame(columns=['Image', 'Object', 'Confidence'])

    lst_threshold = np.arange(0.0, 1.05, 0.025).tolist()

    for idx, row in data_source.iterrows():
        attempt = 1
        while True:
            try:
                img = io.imread(row['URL'])
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                pred = tfnet.return_predict(img)
                if len(pred) > 0:
                    predictions_all = predictions_all.append([{'Image': str(row['ID']), 'Object': obj['label'], 'Confidence': obj['confidence']} for obj in pred], ignore_index=True)
                logger.info("Processed image: " + str(row['ID']))
            except Exception:
                logger.error ("Error processing image: " + str(row['URL']))
                logger.error (traceback.format_exc())
                attempt += 1
                if attempt < 6:
                    continue
            break

    predictions_all.to_csv(alloutfile, header=True, index=False)


    for thres in lst_threshold:
        new_predictions = predictions_all.loc[predictions_all['Confidence'] >= thres].groupby(['Image', 'Object']).size().reset_index(name='count')
        predictions = predictions.append(pd.concat([new_predictions, pd.DataFrame({'Threshold': thres}, index=new_predictions.index)], axis=1), ignore_index=True)

    predictions.to_csv(outfile, header=True, index=False)

def detect_and_count_objects_from_downloaded(sourcefolder, outfile, alloutfile):

    res = dict()
    predictions = pd.DataFrame(columns=['Image', 'Object', 'count', 'Threshold'])    
    predictions_all = pd.DataFrame(columns=['Image', 'Object', 'Confidence'])

    lst_threshold = np.arange(0.0, 1.05, 0.025).tolist()

    id = 0
    for filename in os.path.listdir(sourcefolder):
        attempt = 1
        while True:
            try:
                id = os.path.splitext(filename)[0]
                imgcv = cv2.imread(os.path.join(sourcefolder, filename))
                pred = tfnet.return_predict(imgcv)
                if len(pred) > 0:
                    predictions_all = predictions_all.append([{'Image': str(id), 'Object': obj['label'], 'Confidence': obj['confidence']} for obj in pred], ignore_index=True)
                logger.info("Processed image: " + str(id))
            except Exception:
                logger.error ("Error processing image: " + str(id))
                logger.error (traceback.format_exc())
                attempt += 1
                if attempt < 6:
                    continue
            break

    predictions_all.to_csv(alloutfile, header=True, index=False)


    for thres in lst_threshold:
        new_predictions = predictions_all.loc[predictions_all['Confidence'] >= thres].groupby(['Image', 'Object']).size().reset_index(name='count')
        predictions = predictions.append(pd.concat([new_predictions, pd.DataFrame({'Threshold': thres}, index=new_predictions.index)], axis=1), ignore_index=True)

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
    ncores = 28

    args = parser.parse_args()
    inputfile = os.path.join(DATA_FOLDER, 'in', args.input_file)
    ncores = args.p
    handler = logging.FileHandler(os.path.join('..', 'log', os.path.splitext(args.input_file)[0] + '.log'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    data_source = ''
    if args.opt == 'URL':
        data_source = pd.read_csv(inputfile, index_col=None, header=0)
    else:
        data_source = os.path.join(DATA_FOLDER, inputfile)
    tic = time.time()
    logger.info ("Launched at: " + str(datetime.now()))

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
        if args.opt == 'URL':
            detect_and_count_objects(data_source,
                                         os.path.join(DATA_FOLDER, 'out', 'predictions_' + os.path.basename(inputfile)),
                                         os.path.join(DATA_FOLDER, 'out', 'confidence_predictions' + os.path.basename(inputfile))
                                        )
        else:
            detect_and_count_objects_from_downloaded(data_source,
                                         os.path.join(DATA_FOLDER, 'out', 'predictions_' + args.input_file),
                                         os.path.join(DATA_FOLDER, 'out', 'confidence_predictions' + args.input_file)
                                        )

    toc = time.time()
    logger.info ("Finished at: " + str(datetime.now()))

    tac = round(toc - tic)
    (t_min, t_sec) = divmod(tac, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    logger.info('Time elapsed: {0} hours {1} minutes and {2} seconds'.format(t_hour, t_min, t_sec))

if __name__ == "__main__":
    main(sys.argv[1:])

