import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import re
import sys
import math
import pickle
import subprocess
from os.path import join, isdir, isfile
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
# from sklearn.svm import SVR
# -*-coding:utf-8-*-
from script.paths import PATHS
from script.add_GDT import get_gdt
from script.generate_formatted_input import parse_server_data

from interface_calculate import calculate_interface, generate_interface_mask
from Bio import PDB
import warnings
from Bio.PDB import Selection

warnings.filterwarnings("ignore")

warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

matchPattern = re.compile(r'TER')
PYTHON_INSTALL = 'python3'

parser = PDB.PDBParser()

TOP_N = 100



def get_num_chains(structure):
    '''
    Returns a list of chain objects in a structure
    '''
    ch_list = Selection.unfold_entities(structure, 'C')
    return ch_list




def calculate_num_chains(data_path,
                         pdbfile,
                         out=None):
    pdbfile = join(data_path, pdbfile)
    structure = parser.get_structure("id", pdbfile)
    ch_list = get_num_chains(structure)

    return ch_list


def calculate_score(prediction):
    # score = (25 - float(prediction)) / 25
    newQA = np.array(prediction)

    return (1 / (1 + newQA * newQA / 12))

    # return score


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
    pathToZoomQAInputData = join(pathToTempDirectory, 'ZoomQA_Input')

    print("Processing input data...")
    create_folder(clean_data_path)
    clean_data_program = join(PATHS.sw_install, 'script/re_number_residue_index.pl')
    clean_data_command = "perl {} {} {}"

    # length = []
    dictionary = dict()

    for pdb in os.listdir(pathToInput):
        dictionary[pdb] = dict()
        chains_name_list = calculate_num_chains(pathToInput, pdb)
        index_list, coordinate, count_list, chain_list = calculate_interface(pathToInput, pdb)
        index_mask = generate_interface_mask(index_list, coordinate, count_list, chain_list)
        for i in range(len(index_mask)):
            count_index = index_mask[i] - 1
            dictionary[pdb][count_index] = dict()
            chain_index = chain_list[count_index] - 1
            dictionary[pdb][count_index]['chain_name'] = chains_name_list[chain_index].get_id()
            dictionary[pdb][count_index]['index'] = index_list[count_index]

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

    step2_location = join(PATHS.sw_install, 'script/step2_generate_casp_fragment_structures_v2.py')
    rfpredictions_locations = join(PATHS.sw_install, 'script/assist_generation_scripts/RF_Predictions/')
    frag_structure_command = f'{PYTHON_INSTALL} {step2_location} {pathToJSON} {rfpredictions_locations} {pathToZoomQAInputData}'
    subprocess.run(frag_structure_command.split(" "), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print('3/3 done...')

    return f'{pathToZoomQAInputData}', dictionary


def load_input_data(pathToData):
    """
    This method is repsonsible for loading the output from preprocess_input and getting
    it staged for going into the model

    Parameters:
    ---------------
    pathToData: string
        This is a path to the tmp folder returned from preprocess_input(), it holds
        all of the input data

    Returns:
    -----------
    dictionary {string, dictionary{feature_name: list}}
        The return is a dictionary where the keys are the server names, and the values
        are dictionaries of input features, will get formatted just before predictions are made

    """
    # check if this path is to the data, or just to the folder containing it
    '''I need a better way to do this, this is messy and unreliable, but works for now'''
    pathToServers = pathToData
    folder_contents = os.listdir(pathToServers)
    while isdir(join(pathToServers, folder_contents[0])):
        pathToServers = join(pathToServers, folder_contents[0])
        folder_contents = os.listdir(pathToServers)

    server_names = os.listdir(pathToServers)
    # print(pathToServers)

    input_data = []
    for server_name in server_names:
        pathToServer = join(pathToServers, server_name)
        data = pickle.load(open(pathToServer, 'rb'))
        input_data.append((server_name, data))

    return input_data


def load_model(pathToModel):


    # with open(pathToModel, 'rb') as f:
    #     model = pickle.load(f)

    # print("Model parameters: ")
    # print(model)

    # dir_out = './test_out_le-4_d2e-4_feature_select300/'
    predictor = tf.saved_model.load(pathToModel)
    return predictor


def make_predictions(model, input_data):
    """
    This method is responsible for making predictions on the input data

    Parameters:
    --------------
    model: SVR model
        The pretrained model for QA prediction

    input_data: dictionary {string:dictionary:{feature_name:list}}
        this is a dictionary with the server name as the key, and the value is
        the input feature dictionary, with keys of feature names and values are lists with feature Values

    Return:
    dictionary: {string: list[float]}
        The return is a dictionary with keys being the name of the input server, and the
        values are a list that correspond to the distance in angstroms of the predicted
        amino acid compared to an unknown ground truth

    """


    predictions = {}
    for (server_name, whole_target_data) in input_data:
        server_prediction = []
        # turn data into correct input form

        pdb_node_batch, model_contact_batch, pdb_distance_pair_batch = parse_server_data(whole_target_data)
        # get predictions
        # server_prediction_normalized = model.predict(server_X)
        pred_logits = model.tf_translate(pdb_node_batch, model_contact_batch, pdb_distance_pair_batch)['outputs']
        pred_logits = tf.clip_by_value(pred_logits, clip_value_min=0, clip_value_max=1)
        local_qa = tf.squeeze(pred_logits)
        qa_score = np.array(local_qa).tolist()

        predictions[server_name] = qa_score

    return predictions


def write_predictions(dictionary, prediction_data, pathToSave, target_name):
    """
    This method writes out the predictions in CASP format

    Parameters:
    -------------
    prediction_data: dictionary {string: list[float]}
        The prediction data a dictionary with keys being the name of the input server, and the
        values are a list that correspond to the distance in angstroms of the predicted
        amino acid compared to an unknown ground truth

    pathToSave: string
        This is a string representation to the path to the save folder, this holds the output.txt file

    target_name: string
        The name of the input target, taken from the input data folder names


    Return:
    ------------
    None:
        This method just writes to an output text file


    """

    '''
    Header format:
    PFRMAT QA
    TARGET T0999
    AUTHOR 1234-5678-9000
    REMARK Error estimate is CA-CA distance in Angstroms
    METHOD Description of methods used
    MODEL 1
    '''

    with open(join(pathToSave, f'{target_name}.txt'), 'w+') as f:
        # set up the header
        f.write('PFRMAT\tQA\n')
        f.write(f'TARGET\t{target_name}\n')
        f.write(f"AUTHOR\t0658-1947-3419\n")
        f.write("REMARK\tReliability of residues being in Interfaces\n")
        f.write(f"METHOD\tSVR\n")
        f.write("MODEL\t1\n")
        f.write("QMODE\t2\n")

        # write results
        for server_name_unformatted, server_predictions in prediction_data.items():
            server_name_formatted = server_name_unformatted.replace('.pkl', '')
            # prediction_string = ' '.join([str(round(pred, 3)) for pred in server_predictions])
            prediction_string = ''
            sum = 0

            gdt = get_gdt(server_predictions)
            server_predictions.insert(0, gdt)

            prediction_string += str(round(server_predictions[0], 2)) + " "

            key_list = list(dictionary[server_name_formatted].keys())

            lddt_score = []
            for i in range(len(key_list)):
                lddt_score.append(server_predictions[key_list[i]])

            interface_score = get_gdt(lddt_score)

            prediction_string += str(round(interface_score, 2)) + " "

            for i in range(len(key_list)):
                chain_name = dictionary[server_name_formatted][key_list[i]]['chain_name']
                index = dictionary[server_name_formatted][key_list[i]]['index']
                score = server_predictions[key_list[i]+1]
                prediction_string += str(chain_name) + str(index) + ":" + str(round(score, 2)) + " "
                if sum % 25 == 0 and sum != 0:
                    prediction_string += "\n"
                sum += 1

            # for prediction in server_predictions:
            #     prediction_string += str(round(prediction, 3)) + " "
            #     if i % 25 == 0 and i != 0:
            #         prediction_string += "\n"
            #     i += 1
            f.write(f"{server_name_formatted} {prediction_string}")
            f.write("\n")

        # write ending
        f.write("END\n")


def main(pathToInput, pathToSave):
    start = timer()

    pathToModel = PATHS.model_path

    pattern = re.compile(r"H\d{4}[a-zA-Z]*[0-9]*")
    target_name = re.search(pattern, pathToInput)

    if target_name is not None:
        target_name = str(target_name[0])
    else:
        target_name = 'Target'

    # create save folder
    create_folder(pathToSave)

    # make the data the proper input format for the model
    model_input_path, index_dictionary = preprocess_input(pathToInput, pathToSave)
    print("Input data created...")

    # load input data
    # remove when done testing
    # model_input_path = './TEST_OUT/tmp/ZoomQA_Input/step/T1096/'
    input_data = load_input_data(model_input_path)
    print("Input data loaded...")

    # load models
    model = load_model(pathToModel)

    # make predictions
    target_predictions = make_predictions(model, input_data)

    # write predictions
    write_predictions(index_dictionary, target_predictions, pathToSave, target_name)
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
    if len(sys.argv) < 3:
        print('Not enough arguments... example command: ')
        print(f'python {sys.argv[0]} /path/To/Input/folder/ /path/to/output/save')
        sys.exit()

    
    # sys.exit()
    pathToInput = sys.argv[1]
    pathToSave = sys.argv[2]

    main(pathToInput, pathToSave)
