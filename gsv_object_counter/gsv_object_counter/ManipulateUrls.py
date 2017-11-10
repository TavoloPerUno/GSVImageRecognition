from urllib.parse import urlparse, parse_qs, parse_qsl, urlunparse, urlencode
import pandas as pd
import argparse
import os
import logging
import traceback
import sys
from urllib.request import urlopen
import json
import numpy as np
import multiprocessing as mproc
import shutil

logger = logging.getLogger('ManipulateUrls')

parser = argparse.ArgumentParser(description='Create GSV URLs')
APICALLBASE = 'https://maps.googleapis.com/maps/api/streetview/metadata?&pano='
DATA_FOLDER = '../data'

# Required positional argument
parser.add_argument('input_file', type=str,
                    help='Input file')

parser.add_argument('--api_key', type=str,
                    help='API key')

parser.add_argument('--fov', type=int,
                    help='fov')

parser.add_argument('--heading', type=int,
                    help='heading')

parser.add_argument('--pitch', type=int,
                    help='pitch')

parser.add_argument('--p', type=int,
                    help='nCores')

parser.add_argument('op', type=str,
                    help='Operation')

def change_api_key(input_file, output_file, apikey, pitch, heading, fov):

    param_dict = {'pitch' : str(pitch), 'heading' : str(heading), 'pov': str(fov), 'key' : apikey}
    input = pd.read_csv(input_file, header=0)
    output = pd.DataFrame(columns=['id',
                                   'x',
                                   'city',
                                   'task',
                                   'seq',
                                   'Ped',
                                   'Cyclists',
                                   'Parked_cycles',
                                   'Cars',
                                   'Buses',
                                   'Motorbikes',
                                   'Vans',
                                   'Flag'
                                   ]
                          )


    output = output.append([{'id' : row['id'],
                             'x' : urlunparse((
                                                urlparse(row['x']).scheme,
                                                urlparse(row['x']).netloc,
                                                urlparse(row['x']).path,
                                                urlparse(row['x']).params,
                                                urlencode([(f, (v if f not in param_dict else param_dict[f])) for (f, v) in parse_qsl(urlparse(row['x']).query)]),
                                                urlparse(row['x']).fragment
                                                )
                                             ),
                             'city' : row['city'],
                             'task' : row['task'],
                             'seq' : row['seq'],
                             'Ped' : row['Ped'],
                             'Cyclists' : row['Cyclists'],
                             'Parked_cycles' : row['Parked_cycles'],
                             'Cars' : row['Cars'],
                             'Buses' : row['Buses'],
                             'Motorbikes' : row['Motorbikes'],
                             'Vans' : row['Vans'],
                             'Flag' : row['Flag']
                             } for idx, row in input.iterrows()], ignore_index=True)



    output.to_csv(output_file, header=True, index=False)

def get_lat_long(inputdf, output_file, apikey):
    output = pd.DataFrame(columns=['id',
                                   'panoid',
                                   'latlong']
                          )

    for idx, row in inputdf.iterrows():
        i = 0
        while True:
           
            pano = parse_qs(urlparse(row['x']).query)
            if 'pano' in pano:
                pano = pano['pano']

                requesturl = APICALLBASE + pano[0] + '&key=' + apikey
                if i > 0:
                    print("Timed out : " + APICALLBASE + row['pano_id'] + '&key=' + apikey)
                    if i > 6:
                        break
                try:
                    i += 1
                    metadata = json.load(urlopen(requesturl))
                    if 'location' in metadata:
                        nrow = [{'id' : row['id'],
                                'panoid' : pano[0],
                                'latlong': str(metadata['location']['lat']) + ',' + str(metadata['location']['lng'])}]
                        output = output.append(nrow, ignore_index= True)
                    else:
                        print("Skipping " + str(row['id']))
                    break
                except Exception:
                    logger.info(traceback.print_exc())
                    continue
            else:
                logger.info("Id: " + str(row['id']) + "Invalid URL:" + row['x'])
                break 

    output.to_csv(output_file, header=True, index=False)


def pano_to_lat_long(input_file, output_file, apikey, cores):
    input = pd.read_csv(input_file, header=0)
    output = pd.DataFrame(columns=['id',
                                   'panoid',
                                   'latlong']
                          )

    lst_subfile = []
    procs = []
    i = 0

    if not os.path.exists(os.path.join(DATA_FOLDER, 'temp')):
        os.makedirs(os.path.join(DATA_FOLDER, 'temp'))

    for res in np.array_split(input, cores):
        subfile = os.path.join(DATA_FOLDER, 'temp', str(i) + '_'+ os.path.basename(input_file))
        lst_subfile.append(subfile)
        proc = mproc.Process(target=get_lat_long,
                             args=(res,
                                   subfile,
                                   apikey,
                                   )
                             )
        procs.append(proc)
        proc.start()

        i += 1

    for proc in procs:
        proc.join()

    lst_result = []
    for file in lst_subfile:
        if os.path.isfile(file):
            try:
                result = pd.read_csv(file, index_col=None, header=0)
                lst_result.append(result)
            except pd.errors.EmptyDataError:
                continue

    result = pd.concat(lst_result)

    result.to_csv(output_file, header=True, index=False)

    shutil.rmtree(os.path.join(DATA_FOLDER, 'temp'))

def main(argv):
    input_file = ''
    pitch = 0
    heading = 0
    fov = 90
    p = 28


    args = parser.parse_args()
    input_file = os.path.join(DATA_FOLDER, 'in', args.input_file)

    if args.p is not None and args.p > 0:
        p = args.p

    if args.fov is not None and args.fov > 0:
        fov = args.fov
    if args.pitch is not None and args.pitch > 0:
        pitch = args.pitch
    if args.heading is not None and args.heading > 0:
        heading = args.heading

    handler = logging.FileHandler(os.path.join('..', 'log', os.path.splitext(args.input_file)[0] + '_manipulation' + '.log'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    if args.op == 'ChangeKey':
        if args.api_key is None:
            print("No key input")
        else:
            output_file = os.path.join(DATA_FOLDER, 'out', 'key_' + args.input_file)
            change_api_key(input_file, output_file, args.api_key, pitch, heading, fov)

    if args.op == 'PanoToLatLong':
        output_file = os.path.join(DATA_FOLDER, 'out', 'latlong_' + args.input_file)
        pano_to_lat_long(input_file, output_file, args.api_key, p)

if __name__ == "__main__":
    main(sys.argv[1:])
