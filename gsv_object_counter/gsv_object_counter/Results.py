import pandas as pd
import numpy as np
import argparse
import os
import sys
import json
import multiprocessing as mproc


parser = argparse.ArgumentParser(description='Summarise findings')
DATA_FOLDER = '../data'

OBJECTS = ['car', 'person', 'bicycle', 'bus', 'truck', 'motorbike']

# Required positional argument
parser.add_argument('input_file', type=str,
                    help='Input file')

parser.add_argument('-o',
                    help='output file')

def fill_true_values(calc_dict, df_ground_truth, df_pred, thres):
    dct_val = dict()
    for idx, row in df_ground_truth.iterrows():
        for obj in OBJECTS:
            nobj = row[obj]
            npred = df_pred.loc[(df_pred.Image == row['id']) & (df_pred.Object == obj) , 'count'].values[0] if (df_pred[(df_pred['Image'] == row['id']) & (df_pred['Object'] == obj) & (df_pred.Threshold == thres)].shape[
                           0] > 0) else 0

            if obj not in dct_val:
                dct_val[obj] = dict()            
            dct_val[obj]['TP'] = dct_val[obj].get('TP', 0) + min(npred, nobj)

            dct_val[obj]['FP'] = dct_val[obj].get('FP', 0) + (npred - min(nobj, npred))

            dct_val[obj]['BTP'] = dct_val[obj].get('BTP', 0) + (1 if npred > 0 and nobj > 0 else 0)

            dct_val[obj]['BFP'] = dct_val[obj].get('BFP', 0) + (1 if npred > 0 and nobj == 0 else 0)

            dct_val[obj]['TN'] = dct_val[obj].get('TN', 0) + (1 if npred == nobj == 0 else 0)

            dct_val[obj]['FN'] = dct_val[obj].get('FN', 0) + (nobj - min(npred, nobj))

            dct_val[obj]['BFN'] = dct_val[obj].get('BFN', 0) + (1 if npred == 0 and nobj > 0 else 0)

    calc_dict[str(thres)] = dct_val
    print("Finished filling truth rate for threshold " + str(thres))
    return



def confusion_matrix(infile, outfile):

    mgr = mproc.Manager()

    df_ground_truth_source = pd.read_csv(infile, header=0)
    df_pred = pd.read_csv(outfile)
    df_pred = df_pred.loc[(df_pred.Object.isin(OBJECTS))]

    # failids = np.genfromtxt(os.path.join(DATA_FOLDER, 'in', 'fail.csv'), dtype=int).tolist()
    # df_ground_truth_source = df_ground_truth_source[~df_ground_truth_source.id.isin(failids)]
    df_ground_truth_source.replace(-1, 0, inplace=True)

    df_ground_truth = pd.DataFrame(columns = ['id', 'car', 'person', 'bicycle', 'bus', 'truck', 'motorbike'])


    df_ground_truth = df_ground_truth.append([{'id': row['id'],
                             'car': row['Cars'],
                             'person': row['Ped'] + row['Cyclists'],
                             'bicycle': row['Cyclists'] + row['Parked_cycles'],
                             'bus': row['Buses'],
                             'truck': row['Vans'],
                             'motorbike': row['Motorbikes']
                             } for idx, row in df_ground_truth_source.iterrows()], ignore_index=True)



    result = dict()

    result_sample = dict()

    dct_calc = mgr.dict()

    dct_neg_pos = dict()

    lst_threshold = np.arange(0.0, 1.05, 0.025).tolist()

    lst_proc = []

    for thres in lst_threshold:
        proc = mproc.Process(target=fill_true_values,
                             args=(dct_calc, 
                                   df_ground_truth,
                                   df_pred.loc[df_pred.Threshold == thres],
                                   thres,
                                   )
                             )
        proc.start()

        lst_proc.append(proc)

    for obj in OBJECTS:
        dct_neg_pos[obj] = dict()
        dct_neg_pos[obj]['N'] = df_ground_truth[(df_ground_truth[obj] == 0)].shape[0]
        dct_neg_pos[obj]['P'] = df_ground_truth[obj].sum()
        dct_neg_pos[obj]['BP'] = df_ground_truth[(df_ground_truth[obj] > 0)].shape[0]

    for proc in lst_proc:
        proc.join()

    df_result_sample = pd.DataFrame(columns=['Class', 'Threshold', 'Specificity', 'Sensitivity', 'Sensitivity_b'])
    df_result = pd.DataFrame(columns=['Class', 'Threshold', 'Specificity', 'Specificity_b', 'Sensitivity', 'Sensitivity_b'])

    for thres in lst_threshold:
        df_result_sample = df_result_sample.append([{'Class': obj,
                                  'Threshold': str(thres),
                                  'Sensitivity': (dct_calc[str(thres)][obj]['TP']/ dct_neg_pos[obj]['P'] if dct_neg_pos[obj]['P'] > 0 else 0),
                                  'Sensitivity_b': (dct_calc[str(thres)][obj]['BTP'] / dct_neg_pos[obj]['BP'] if dct_neg_pos[obj]['BP'] > 0 else 0),
                                 'Specificity': (dct_calc[str(thres)][obj]['TN'] / dct_neg_pos[obj]['N'] if dct_neg_pos[obj]['N'] > 0 else 0)}
                                 for obj in OBJECTS], ignore_index=True)

        df_result = df_result.append([{'Class': obj,
                           'Threshold': str(thres),
                           'Sensitivity': (dct_calc[str(thres)][obj]['TP'] / (dct_calc[str(thres)][obj]['TP'] + dct_calc[str(thres)][obj]['FN']) if (dct_calc[str(thres)][obj]['TP'] + dct_calc[str(thres)][obj]['FN']) > 0 else 0),
                           'Sensitivity_b': ((dct_calc[str(thres)][obj]['BTP'] / (dct_calc[str(thres)][obj]['BTP'] + dct_calc[str(thres)][obj]['FN'])) if (dct_calc[str(thres)][obj]['BTP'] + dct_calc[str(thres)][obj]['FN']) > 0 else 0),
                           'Specificity': ((dct_calc[str(thres)][obj]['TN'] / (dct_calc[str(thres)][obj]['TN'] + dct_calc[str(thres)][obj]['FP'])) if (dct_calc[str(thres)][obj]['TN'] + dct_calc[str(thres)][obj]['FP']) > 0 else 0),
                           'Specificity_b': ((dct_calc[str(thres)][obj]['TN'] / (dct_calc[str(thres)][obj]['TN'] + dct_calc[str(thres)][obj]['BFP'])) if (dct_calc[str(thres)][obj]['TN'] + dct_calc[str(thres)][obj]['FP']) > 0 else 0)}
                        for obj in OBJECTS], ignore_index=True)


    # result = pd.DataFrame([(obj, stat, val) for obj, value in result.items() for stat, val in value.items()], columns=['Class', 'Statistic', 'Val'])
    # result_sample = pd.DataFrame([(obj, stat, val) for obj, value in result_sample.items() for stat, val in value.items()],
    #                       columns=['Class', 'Statistic', 'Val'])

    df_result_sample.to_csv(os.path.join(DATA_FOLDER, 'out', 'summary_sample.csv'), index=False, header=True)
    df_result.to_csv(os.path.join(DATA_FOLDER, 'out', 'summary.csv'), index=False, header=True)

    return



def main(argv):
    args = parser.parse_args()
    inputfile = os.path.join(DATA_FOLDER, 'in', args.input_file)
    outputfile = os.path.join(DATA_FOLDER, 'out', args.o)

    confusion_matrix(inputfile, outputfile)

if __name__ == "__main__":
    main(sys.argv[1:])
