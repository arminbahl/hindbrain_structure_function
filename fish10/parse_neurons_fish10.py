import os

import json
import pandas as pd
import numpy as np
import sys
import toml
import util as ut
from urllib.parse import unquote
import navis
import cloudvolume as cv
import convert_segments_231218 as convert_segments


class EMSegmentsHelpers():
    def __init__(self, manual_opts_dict=dict()):
        self.set_opts_dict(manual_opts_dict)
   
    def set_opts_dict(self, manual_opts_dict):
         #'cloudvolume_url':'precomputed://gs://fish1-public/seg_231218',
        default_opts_dict = {          
            'cloudvolume_url':'precomputed://gs://fish1-public/seg_231218_agg240107b_ph12_spl',
            'old_segments': 'old_segments',
            'new_segments': 'new_segments',
            'output_data': 'output_data',
            'prefix_string':'https://neuroglancer-demo.appspot.com/#!',
            'skeleton_file': 'skeleton_seg_231218_1.json',
        }

        for key in manual_opts_dict.keys():
            default_opts_dict[key] = manual_opts_dict[key]

        self.opts_dict = default_opts_dict
        navis.patch_cloudvolume()
        self.vol = cv.CloudVolume(self.opts_dict["cloudvolume_url"],use_https=True, progress=False)

    def download_neuron_segments(self, seg_ids, out_file):
        """
        download segments as a mesh using clould volume

        arguments:
            seg_ids: list
                comma separated segment ids
            out_file: str
                path where mesh will be stored
        """    
        m = self.vol.mesh.get(seg_ids, as_navis=True)
        comb = navis.combine_neurons(m)
        navis.write_mesh(comb, out_file)

    def process_csv_file(self, neurons_file, base_folder):
        """
        process neurons and their meta information in the csv file

        arguments:
            neurons_file: str
                csv file containing neuron information
            base_folder: str
                path where neurons data will be stored, 
                make sure new skeleton file should be stored in same folder
        """    
        old_segments_folder=os.path.join(base_folder, self.opts_dict["old_segments"])
        new_segments_folder=os.path.join(base_folder, self.opts_dict["new_segments"])
        cells_data_folder=os.path.join(base_folder, self.opts_dict["output_data"])
        skeleton_file=os.path.join(base_folder, self.opts_dict["skeleton_file"])
        ut.create_directory(old_segments_folder)
        ut.create_directory(new_segments_folder)
        ut.create_directory(cells_data_folder)
        df = self.read_excel_file(neurons_file)
        for index, row in df.iterrows():
            cell_id=row['id']
            cell_name=row['name']
            cell_folder=os.path.join(cells_data_folder,cell_name)
            ut.create_directory(cell_folder)
            cell_units=row['units']
            layerId=row['layer']
            cell_tracing_date=row['date_of_tracing']
            cell_tracers=row['tracer_name']
            cell_classifier=row['classifier']
            cell_modality=row['imaging_modality']
            cell_neuroglancer_link=row['neuroglancer_link']
            url = unquote(cell_neuroglancer_link)
            url=url.removeprefix(self.opts_dict["prefix_string"]) 
            old_segments_data=json.loads(url)    
            old_json_file=os.path.join(old_segments_folder, cell_name+".json")
            ut.write_json(old_segments_data, old_json_file)
            seg_ids=convert_segments.convert_single_neuron(old_json_file, layerId, skeleton_file, new_segments_folder)
            base_seg_str = ','.join(str(x) for x in seg_ids)
            segment_data_frame=convert_segments.agglomerated_segments_query(base_seg_str)
            old_segments=segment_data_frame['seg_id_base'].tolist()
            agglomerated_segments=segment_data_frame['seg_id_agglomerated'].tolist()
            final_list = list(set(seg_ids) - set(old_segments))
            final_list.extend(agglomerated_segments)
            self.download_neuron_segments(final_list,os.path.join(cell_folder, (cell_name+'.obj')) )
            cell_soma=row['soma_position']
            cell_mece_regions=row['mece_regions']
            #cell_synapses=row['synapses']
            #print(cell_synapses)
            pre_synaptic=[]
            post_synaptic=[]
            cell_synapses=None
            if not pd.isna(cell_synapses):
                cell_synapses_dict=json.loads(cell_synapses)
                pre_synaptic=cell_synapses_dict['presynaptic']
                post_synaptic=cell_synapses_dict['postsynaptic']

            cell_others = row['others'] if not pd.isna(row['others']) else {}
            output_data={
                'id':cell_id,
                'name': cell_name,
                'units':cell_units,
                'tracer_name':cell_tracers,
                'imaging_modality':cell_modality,
                'soma_position':cell_soma,
                'classifier': cell_classifier,
                'mece_regions':cell_mece_regions,
                'others': cell_others,
                'date_of_tracing':cell_tracing_date,
                'presynaptic':pre_synaptic,
                'postsynaptic':post_synaptic           
            }

            output_file_name = os.path.join(cell_folder, cell_name+".txt")
            toml_file=open(output_file_name, "w")
            toml.dump(output_data, toml_file)
            print(cell_name)

    def test_segment(self):
        all_segments=[217030381, 200752462, 175738782, 122967887, 315052108, 208966913, 60212122, 138647536, 299111671, 239043596, 149756350, 53798951, 81605282, 392121992, 384255747, 192577630, 158346092, 160622366, 153347638, 300614500, 392871904, 234151035, 351914309, 249750438, 264200116, 42462198]
        manifest_not_avail=[300614500, 299111671, 53798951, 192577630, 158346092, 264200116]
        manifest_avail_segments=[217030381, 200752462, 175738782, 122967887, 315052108, 208966913, 60212122, 138647536, 239043596, 149756350, 81605282, 392121992, 384255747, 160622366, 153347638, 392871904, 234151035, 351914309, 249750438, 42462198]
        #m = self.vol.mesh.get(all_segments, as_navis=True)
        base_seg_str = ','.join(str(x) for x in all_segments)
        segment_data_frame=convert_segments.agglomerated_segments_query(base_seg_str)
        old_segments=segment_data_frame['seg_id_base'].tolist()
        agglomerated_segments=segment_data_frame['seg_id_agglomerated'].tolist()
        final_list = list(set(all_segments) - set(old_segments))
        final_list.extend(agglomerated_segments)
        m = self.vol.mesh.get(final_list, as_navis=True)
        print(m)
        print(agglomerated_segments)

    def read_excel_file(self, file_name):
        if(file_name.endswith('.csv')):
            return pd.read_csv(neurons_file)
        else:
            return pd.read_excel(file_name)

if __name__ == "__main__":
    print(len(sys.argv))
    neurons_file = sys.argv[1]
    output_folder=sys.argv[2]
    helper=EMSegmentsHelpers()
    helper.process_csv_file(neurons_file, output_folder)
    #helper.test_segment()
    #df=helper.read_excel_file(neurons_file)
    #print(df)