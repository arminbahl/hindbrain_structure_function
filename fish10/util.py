

import os
import json
import numpy as np
from bson import ObjectId

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)  

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def get_all_files(directory): 
  
    # initializing empty file paths list 
    file_paths = [] 
  
    # crawling through directory and subdirectories 
    for root, directories, files in os.walk(directory): 
        for filename in files: 
            # join the two strings in order to form the full filepath. 
            filepath = os.path.join(root, filename) 
            file_paths.append(filepath) 
  
    # returning all file paths 
    return file_paths


def get_all_directories(directory): 
  
    # initializing empty file paths list 
    file_paths = [] 
  
    # crawling through directory and subdirectories 
    for root, directories, files in os.walk(directory): 
        for filename in directories: 
            # join the two strings in order to form the full filepath. 
            filepath = os.path.join(root, filename) 
            file_paths.append(filepath) 
  
    # returning all file paths 
    return file_paths


def get_all_file_paths(directory): 
  
    # initializing empty file paths list 
    file_paths = [] 
  
    # crawling through directory and subdirectories 
    for root, directories, files in os.walk(directory): 
        for filename in files: 
            # join the two strings in order to form the full filepath. 
            #filepath = os.path.join(root, filename) 
            file_paths.append(filename) 
  
    # returning all file paths 
    return file_paths 

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def write_dir_zip(input_dir, output_dir):
    all_files=get_all_file_paths()
    for file_name in all_files:
        outfile=os.path.join(output_dir, file_name+".zip")
        input_file=os.path.join(input_dir, file_name)
        print(input_file)

def loadJson(filename):
    with open(filename) as f:
        return json.load(f)
    
def write_json(skeleton_data, out_json):
    json_file= open(out_json, 'w')
    json.dump(skeleton_data,json_file) 

def write_json_encoded(out_file, data):
    json_file=open(out_file, 'w')
    json.dump(data,json_file, cls=JSONEncoder) 

