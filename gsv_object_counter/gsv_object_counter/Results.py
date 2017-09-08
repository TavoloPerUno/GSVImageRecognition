import pandas as pd
import numpy as np
import argparse
import os
import sys
import json


parser = argparse.ArgumentParser(description='Summarise findings')
DATA_FOLDER = '../data'

# Required positional argument
parser.add_argument('input_file', type=str,
                    help='Input file')

parser.add_argument('-o',
                    help='output file')

def confusion_matrix(infile, outfile):

    ground_truth = pd.read_csv(infile, header=0)
    pred = pd.read_csv(outfile)
    failids = np.genfromtxt(os.path.join(DATA_FOLDER, 'in', 'fail.csv'), dtype=int).tolist()
    ground_truth = ground_truth[~ground_truth.id.isin(failids)]
    ground_truth.replace(-1, 0, inplace=True)

    result = dict()

    result_sample = dict()

    calc_dict = dict()

    objects = ['car', 'person', 'bicycle', 'bus', 'truck', 'motorbike']

    for idx, row in ground_truth.iterrows():

        for obj in objects:
            nobj = 0
            if obj == 'person':
                nobj = row['Ped'] + row['Cyclists']
            if obj == 'bicycle':
                nobj = row['Cyclists'] + row['Parked_cycles']
            if obj == 'bus':
                nobj = row['Buses']
            if obj == 'truck':
                nobj = row['Vans']
            if obj == 'motorbike':
                nobj = row['Motorbikes']
            if obj == 'car':
                nobj = row['Cars']

            npred = pred.loc[(pred.Image == row['id']) & (pred.Object == obj), 'count'].values[0] if (pred[(pred['Image'] == row['id']) & (pred['Object'] == obj)].shape[0] > 0) else 0

            if obj not in calc_dict:
                calc_dict[obj] = dict()

            calc_dict[obj]['TP'] = calc_dict[obj].get('TP', 0) + min(npred, nobj)

            calc_dict[obj]['FP'] = calc_dict[obj].get('FP', 0) + (npred - min(nobj, npred))

            calc_dict[obj]['BTP'] = calc_dict[obj].get('BTP', 0) + (1 if npred > 0 and nobj > 0 else 0)

            calc_dict[obj]['BFP'] = calc_dict[obj].get('BFP', 0) + (1 if npred > 0 and nobj == 0 else 0)

            calc_dict[obj]['TN'] = calc_dict[obj].get('TN', 0) + (1 if npred == nobj == 0 else 0)

            calc_dict[obj]['FN'] = calc_dict[obj].get('FN', 0) + (nobj - min(npred, nobj))

            calc_dict[obj]['BFN'] = calc_dict[obj].get('BFN', 0) + (1 if npred == 0 and nobj > 0 else 0)


    for obj in calc_dict.keys():

        if obj == 'person':
            calc_dict[obj]['N'] = ground_truth[(ground_truth['Ped'] == 0) | (ground_truth['Cyclists'] == 0)].shape[0]
            calc_dict[obj]['P'] = ground_truth['Ped'].sum() + ground_truth['Cyclists'].sum()
            calc_dict[obj]['PB'] = ground_truth[(ground_truth['Ped'] > 0) | (ground_truth['Cyclists'] > 0)].shape[0]
        if obj == 'bicycle':
            calc_dict[obj]['N'] = ground_truth[(ground_truth['Cyclists'] == 0) | (ground_truth['Parked_cycles'] == 0)].shape[0]
            calc_dict[obj]['P'] = ground_truth['Cyclists'].sum() + ground_truth['Parked_cycles'].sum()
            calc_dict[obj]['PB'] = ground_truth[(ground_truth['Cyclists'] > 0) | (ground_truth['Parked_cycles'] > 0)].shape[0]
        if obj == 'bus':
            calc_dict[obj]['N'] = ground_truth[ground_truth['Buses'] == 0].shape[0]
            calc_dict[obj]['P'] = ground_truth['Buses'].sum()
            calc_dict[obj]['PB'] = ground_truth[ground_truth['Buses'] > 0].shape[0]
        if obj == 'truck':
            calc_dict[obj]['N'] = ground_truth[ground_truth['Vans'] == 0].shape[0]
            calc_dict[obj]['P'] = ground_truth['Vans'].sum()
            calc_dict[obj]['PB'] = ground_truth[ground_truth['Vans'] > 0].shape[0]
        if obj == 'motorbike':
            calc_dict[obj]['N'] = ground_truth[ground_truth['Motorbikes'] == 0].shape[0]
            calc_dict[obj]['P'] = ground_truth['Motorbikes'].sum()
            calc_dict[obj]['PB'] = ground_truth[ground_truth['Motorbikes'] > 0].shape[0]
        if obj == 'car':
            calc_dict[obj]['N'] = ground_truth[ground_truth['Cars'] == 0].shape[0]
            calc_dict[obj]['P'] = ground_truth['Cars'].sum()
            calc_dict[obj]['PB'] = ground_truth[ground_truth['Cars'] > 0].shape[0]

        result[obj] = dict()

        result_sample[obj] = dict()

        result_sample[obj]['sensitivity'] = calc_dict[obj]['TP'] / calc_dict[obj]['P'] if (calc_dict[obj]['P'] > 0) else 0

        result_sample[obj]['sensitivity_b'] = calc_dict[obj]['BTP'] / calc_dict[obj]['PB']  if (calc_dict[obj]['PB'] > 0) else 0

        result_sample[obj]['specificity'] = calc_dict[obj]['TN'] / calc_dict[obj]['N'] if (calc_dict[obj]['N'] > 0) else 0

        print(obj + 'TP:' + str(calc_dict[obj]['TP']))
        print(obj + 'TP + FN:' + str(calc_dict[obj]['BTP'] + calc_dict[obj]['FN']))
        print(obj + 'P:' + str(calc_dict[obj]['P']) + 'PB' + str(calc_dict[obj]['PB']))
        print(obj + 'N:' + str(calc_dict[obj]['N']))

        result[obj]['sensitivity'] = (calc_dict[obj]['TP'] / (calc_dict[obj]['TP'] + calc_dict[obj]['FN'])) if (
                                                                                                               calc_dict[
                                                                                                                   obj][
                                                                                                                   'TP'] +
                                                                                                               calc_dict[
                                                                                                                   obj][
                                                                                                                   'FN']) > 0 else 0

        result[obj]['sensitivity_b'] = (calc_dict[obj]['BTP'] / (calc_dict[obj]['BTP'] + calc_dict[obj]['FN'])) if (
                                                                                                                   calc_dict[
                                                                                                                       obj][
                                                                                                                       'BTP'] +
                                                                                                                   calc_dict[
                                                                                                                       obj][
                                                                                                                       'BFN']) > 0 else 0

        result[obj]['specificity'] = (calc_dict[obj]['TN'] / (calc_dict[obj]['TN'] + calc_dict[obj]['FP'])) if (
                                                                                                               calc_dict[
                                                                                                                   obj][
                                                                                                                   'TN'] +
                                                                                                               calc_dict[
                                                                                                                   obj][
                                                                                                                   'FP']) > 0 else 0

        result[obj]['specificity_b'] = (calc_dict[obj]['TN'] / (calc_dict[obj]['TN'] + calc_dict[obj]['BFP'])) if (
                                                                                                                  calc_dict[
                                                                                                                      obj][
                                                                                                                      'TN'] +
                                                                                                                  calc_dict[
                                                                                                                      obj][
                                                                                                                      'BFP']) > 0 else 0

    result = pd.DataFrame([(obj, stat, val) for obj, value in result.items() for stat, val in value.items()], columns=['Class', 'Statistic', 'Val'])
    result_sample = pd.DataFrame([(obj, stat, val) for obj, value in result_sample.items() for stat, val in value.items()],
                          columns=['Class', 'Statistic', 'Val'])

    result.to_csv(os.path.join(DATA_FOLDER, 'out', 'summary.csv'), index=False, header=True)
    result_sample.to_csv(os.path.join(DATA_FOLDER, 'out', 'summary_sample.csv'), index=False, header=True)

    return



def main(argv):
    args = parser.parse_args()
    inputfile = os.path.join(DATA_FOLDER, 'in', args.input_file)
    outputfile = os.path.join(DATA_FOLDER, 'out', args.o)

    confusion_matrix(inputfile, outputfile)

if __name__ == "__main__":
    main(sys.argv[1:])