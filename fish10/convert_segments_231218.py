import os

import json
import pandas as pd
import numpy as np
import sys
from google.cloud import bigquery
import db_dtypes  # type: ignore
import util as ut 

# setting up credentials to access google database
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\\Users\\bzfvohra\\Documents\\sumit\\laptop_data\\EM\\Em_bigdata_query\\mariela_fish_credentials.json'


def segment_query(base_seg_str):
    """
    fetching new segment ids from google cloud using old segment ids

    arguments:
        base_seg_str: str
            comma separated segment ids
    returns:
        segments_list: list
            neuron new segment ids
    """
    bigquery_client = bigquery.Client()
    QUERY = """
    SELECT
        objects.id_a  as seg_id_new,
        objects.id_b  as seg_id_old,
    FROM
        `engert-goog-connectomics.fish1_231218.curr_to_phase1` as objects
    WHERE objects.id_b in {}
    """.format('('+base_seg_str+')')

    df = bigquery_client.query(QUERY).to_dataframe()
    return df['seg_id_new'].tolist()

def agglomerated_segments_query(base_seg_str):
    """
    fetching new segment ids from google cloud using old segment ids

    arguments:
        base_seg_str: str
            comma separated segment ids
    returns:
        segments_list: list
            neuron new segment ids
    """
    bigquery_client = bigquery.Client()
    QUERY = """
    SELECT
        objects.id_a  as seg_id_base,
        objects.id_b  as seg_id_agglomerated,
    FROM
        `engert-goog-connectomics.fish1_231218.240107b_ph12_spl_itr` as objects
    WHERE objects.id_a in {}
    """.format('('+base_seg_str+')')

    df = bigquery_client.query(QUERY).to_dataframe()

    return df

def convert_mutiple_neurons(input_folder,skeleton_file, output_folder):
    """
    fetching new segment ids from google cloud using old segment ids

    arguments:
        input_folder: str
            folder path that contains old jsons 
        skeleton_file: str
            skeleton of the new neuroglancer instance
        output_folder: str
            folder path where new jsons will be stored
    returns:
        segments_list: list
            neuron new segment ids
    """
    neuron_files=ut.get_all_files(input_folder)
    for neuron_file in neuron_files:
        convert_single_neuron(neuron_file, skeleton_file, output_folder)
        

def convert_segments(neuron_file):
    f=open(neuron_file, 'r')
    neuron_json = json.load(f)
    layers=neuron_json['layers']
    segmentation_layer=layers[5]
    base_segments=segmentation_layer['segments']  
    base_seg_str = ','.join(str(x) for x in base_segments)
    neuron_segments_list=segment_query(str(base_seg_str))
    return neuron_segments_list

def convert_single_neuron(neuron_file, layerId, skeleton_file, output_folder):
    sf=open(skeleton_file, 'r')
    skeleton_json = json.load(sf)
    base_name=os.path.basename(neuron_file)
    f=open(neuron_file, 'r')
    neuron_json = json.load(f)
    layers=neuron_json['layers']
    segmentation_layer=layers[layerId]
    base_segments=segmentation_layer['segments']  
    base_seg_str = ','.join(str(x) for x in base_segments)
    neuron_segments_list=segment_query(str(base_seg_str))
    new_layers=skeleton_json['layers']
    new_segmentation_layer=new_layers[7]
    final_segments = [str(x) for x in neuron_segments_list]
    new_segmentation_layer['segments']=final_segments
    out_file=os.path.join(output_folder, base_name)
    ut.write_json(skeleton_json,out_file)
    return neuron_segments_list

"""
if __name__ == "__main__":
    print(len(sys.argv))
    input_folder = sys.argv[1]
    skeleton_file= sys.argv[2]
    output_folder=sys.argv[3]
    convert_mutiple_neurons(input_folder, skeleton_file, output_folder)
"""