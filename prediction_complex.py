import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import re
import sys
import subprocess
from os.path import join, isdir, isfile
from timeit import default_timer as timer

import numpy as np
import warnings
# from sklearn.svm import SVR
# -*-coding:utf-8-*-
from script.paths import PATHS
from script.add_GDT import get_gdt
from data import ProteinComplexQAData_Generator
import tensorflow as tf
import pandas as pd
import time
import csv


warnings.filterwarnings("ignore")

warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

PYTHON_INSTALL = 'python3'



def preprocess_input(pathToInput, pathToSave):
    """
    This method is responsible for taking the pdb input files and extract
    all of the necesary features into the pickle files that can be easily
    transformed into the correct format for the model input, saves it to a temp
    directory in the output folder

    Parameters:
    ----------------
    pathToInput: string
        This is a string representation to the path to the input data

    pathToSave: string
        This is a string representation to the path to the save folder, so we can
        make a temp folder to store the intermediary steps

    Return:
    ---------------
    type: string
        The return is the path to the final step (step2_generate_casp_fragment_structures)
        output so that the next step can use it as the input

    """
    pattern = re.compile(r"T\d{4}[a-zA-Z]*[0-9]*")
    target_name = re.search(pattern, pathToInput)

    if target_name is not None:
        target_name = str(target_name[0])
    else:
        target_name = 'Target'

    pathToTempDirectory = join(pathToSave, 'tmp')
    create_folder(pathToTempDirectory)
    # os.system("chmod +777 " + pathToTempDirectory)

    clean_data_path = join(pathToTempDirectory, 'cleaned_pdbs')
    pathToStep0 = join(pathToTempDirectory, 'step_0')
    pathToJSON = join(pathToTempDirectory, 'JSON_Data')
    pathToComplexQAInputData = join(pathToTempDirectory, 'ComplexQA_feature')

    print("Processing input data...")
    create_folder(clean_data_path)
    clean_data_program = join(PATHS.sw_install, 'script/re_number_residue_index.pl')
    clean_data_command = "perl {} {} {}"
    for pdb in os.listdir(pathToInput):

        command = clean_data_command.format(clean_data_program, join(pathToInput, pdb), join(clean_data_path, pdb))
        subprocess.run(command.split(" "))
    print("Cleaned PDB's")

    # chain_add_command = f'python ./script/step0_prepare_add_chain_to_folder.py ./script/assist_add_chainID_to_one_pdb.pl {pathToInput} {pathToStep0} > {join(pathToTempDirectory, "step0_log.txt")} 2>&1'
    create_folder(pathToStep0)
    pathToStep0OUT = join(pathToStep0, target_name)
    # this was originally step0_prepare_add_chain_to_folder.py
    step0_location = join(PATHS.sw_install, 'script/step0_prepare_add_chain_to_folder_v2.py')
    chain_add_location = join(PATHS.sw_install, 'script/assist_add_chainID_to_one_pdb.pl')
    chain_add_command = f'{PYTHON_INSTALL} {step0_location} {chain_add_location} {clean_data_path} {pathToStep0OUT}'
    subprocess.run(chain_add_command.split(" "), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print('1/3 done...')

    # change it to _linux for linux run, mac for mac run
    # json_command = f'python ./script/step1_create_json_from_PDB.py ./script/stride_mac {pathToStep0} {pathToJSON} > {join(pathToTempDirectory, "step1_log.txt")} 2>&1'
    step1_location = join(PATHS.sw_install, 'script/step1_create_json_from_PDB.py')
    stride_location = join(PATHS.sw_install, 'script/stride_linux')
    json_command = f'{PYTHON_INSTALL} {step1_location} {stride_location} {pathToStep0} {pathToJSON}'
    subprocess.run(json_command.split(" "), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print('2/3 done...')

    step2_location = join(PATHS.sw_install, 'script/step2_generate_casp_fragment_structures.py')
    rfpredictions_locations = join(PATHS.sw_install, 'script/assist_generation_scripts/RF_Predictions/')
    frag_structure_command = f'{PYTHON_INSTALL} {step2_location} {pathToJSON} {rfpredictions_locations} {pathToComplexQAInputData}'
    subprocess.run(frag_structure_command.split(" "), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print('3/3 done...')

    return f'{pathToComplexQAInputData}'


def evaluate_QA_results(model_translator, eva_generator, info, pathToSave, name='Predict'):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    from tensorflow.python.ops.numpy_ops import np_config

    np_config.enable_numpy_behavior()

    eval_num = 0


    # Grouping dictionary keys by value
    from collections import defaultdict
    length2keys = defaultdict(list)
    for key, val in sorted(eva_generator.id2seq.items()):
        length2keys[val].append(key)

    per_target_models_stat = pd.DataFrame(
        columns=['dataset', 'target', 'model', 'interface_local_score', 'interface_score', 'interface_residue'])

    start = time.time()
    for id_length in length2keys.keys():
        protein_ids = length2keys[id_length]

        # if id_length < 100:
        #     target_batch_size = 200
        # elif id_length < 500:
        #     target_batch_size = 100
        # elif id_length < 800:
        #     target_batch_size = 50
        # else:
        #     target_batch_size = 20

        target_batch_size = 1

        batch_len = int(len(protein_ids) / target_batch_size) + 1
        print(batch_len)
        for indx in range(0, batch_len):
            #print(str(eval_num) + ": done", end=" ", flush=True)
            batch_start = indx * target_batch_size
            batch_end = (indx + 1) * target_batch_size
            if batch_end <= len(protein_ids):
                batch_list = protein_ids[batch_start:batch_end]
            else:
                batch_list = protein_ids[batch_start:]

            if len(batch_list) == 0:
                continue
            # print(batch_list[0])
            pdb_node_batch, model_contact_batch, pdb_distance_pair_batch = eva_generator.collect_data(
                batch_list)

         
            pred_logits = model_translator.tf_translate(pdb_node_batch, model_contact_batch, pdb_distance_pair_batch)[
                'outputs']


            pred_logits = tf.clip_by_value(pred_logits, clip_value_min=0, clip_value_max=1)
            # dock_logits = tf.clip_by_value(dock_logits, clip_value_min=0, clip_value_max=1)

            for model_idx in range(0, len(batch_list)):
                eval_num += 1
                # id_model = 'T0860|FLOUDAS_SERVER_TS1|136'
                model_name = batch_list[model_idx]
                interface_residue = read_interface_csv(info, model_name)
                node_prob = tf.reshape(pred_logits[model_idx], [-1])  # (1, 343, 1) -> (343, )


                summary_pred = pd.DataFrame({'Pred': node_prob},
                                            columns=['Pred'])
                model_pred_lddt_global = np.mean(summary_pred['Pred'])
                ##filter those missing residues with lddt = -1

                model_pred_lddt_local = summary_pred['Pred'].to_list()
                model_pred_lddt_local = [round(x, 5) for x in model_pred_lddt_local]

                per_target_models_stat = per_target_models_stat.append(
                    {'dataset': name, 'target': model_name.split('_')[0], 'model': model_name,
                     'interface_local_score': model_pred_lddt_local, 'interface_score': model_pred_lddt_global,
                     'interface_residue': interface_residue}, ignore_index=True)

    # per_target_models_stat['interface_local_score'] = per_target_models_stat['interface_local_score'].astype(float)
    per_target_models_stat['interface_score'] = per_target_models_stat['interface_score'].astype(float)

    for targetid in per_target_models_stat['target'].unique():
        data_subset = per_target_models_stat[per_target_models_stat['target'] == targetid].reset_index()

        data_subset.sort_values(by='interface_score', ascending=False, inplace=True)

        data_subset.loc[:, 'interface_score'] = data_subset.loc[:, 'interface_score'].round(5)


        data_subset.to_csv(join(pathToSave, f'Rank_{targetid}_complexqa.csv'), index=False)




def load_model(pathToModel):

    predictor = tf.saved_model.load(pathToModel)
    return predictor


def read_interface_csv(csv_file, model_name):
    interface_residue = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['model'] == model_name:
                interface_residue = row['interface_residue']
    return interface_residue



def main(pathToInput, pathToInterface_info, pathToSave):
    start = timer()

    pathToModel = PATHS.model_path

    pattern = re.compile(r"T\d{4}[a-zA-Z]*[0-9]*")
    target_name = re.search(pattern, pathToInput)

    if target_name is not None:
        target_name = str(target_name[0])
    else:
        target_name = 'Target'

    # create save folder
    create_folder(pathToSave)

    # make the data the proper input format for the model
    model_input_path = preprocess_input(pathToInput, pathToSave)
    print("Input data created...")


    # load models
    model = load_model(pathToModel)


    val_generator = ProteinComplexQAData_Generator(data_path=model_input_path, protein_info_pickle=pathToInterface_info, min_seq_size=0,
                                                   max_seq_size=10000,
                                                   batch_size='single', max_msa_seq=1000, max_id_nums=1000000)

    evaluate_QA_results(model, val_generator, pathToInterface_info, pathToSave, name='Predict')

    print(f"Prediction saved to {pathToSave}")
    # remove tmp folder
    print("Cleaning up...")

    folder_to_remove = join(pathToSave, 'tmp')
    os.system(f'rm -rf {folder_to_remove}')
    end = timer()
    total_t = end - start
    print(f"Prediction complete, elapsed time: {total_t}")


def create_folder(pathToFolder):
    '''
    Method to create folder if does not exist, pass if it does exist,
    cancel the program if there is another error (e.g writing to protected directory)
    '''
    try:
        os.mkdir(pathToFolder)
    except FileExistsError:
        print(f"{pathToFolder} already exists...")
    except:
        print(f"Fatal error making {pathToFolder}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('Not enough arguments... example command: ')
        print(f'python {sys.argv[0]} /path/To/Input/folder/ /path/To/interface.csv  /path/to/output/save')
        sys.exit()


    pathToInput = sys.argv[1]
    pathToInterface_info = sys.argv[2]
    pathToSave = sys.argv[3]

    main(pathToInput, pathToInterface_info, pathToSave)
