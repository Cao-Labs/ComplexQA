import math
import os

import numpy as np

import pandas as pd
import argparse
from Bio.PDB import Selection
from Bio import PDB
import warnings

warnings.filterwarnings("ignore")

pdb_parser = PDB.PDBParser()

#Calculate the Euclidean distance of two amino acids
def calculate_distacne(residue_one, residue_two):
    sum = 0
    distance_list = [residue_one[i]-residue_two[i] for i in range(len(residue_one))]
    for i in range(len(distance_list)):
        sum += math.pow(distance_list[i], 2)
    distance = np.sqrt(sum)
    return  distance


def get_num_chains(structure):
    '''
    Returns a list of chain objects in a structure
    '''
    ch_list = Selection.unfold_entities(structure, 'C')
    return ch_list



def calculate_num_chains(pdbfile,
                         out=None):

    structure = pdb_parser.get_structure("id", pdbfile)
    ch_list = get_num_chains(structure)

    return ch_list

def calculate_interface(pdbfile):
    count = 0
    chain_number = 0
    index = 0
    atom_index = 0
    chain_id = '#'
    count_list = []
    index_list =[]
    coordinate_list = []
    chain_number_list = []

    with open(pdbfile) as pdb:
        for lne in pdb:
            # print((lne[13:16]))
            if(lne[13:16]=='CA '):
                if(int(lne[22:26]) > index and lne[21]==chain_id):
                    atom_index = int(lne[6:11])
                    count = count+1
                    count_list.append(count)
                    index = int(lne[22:26])   #pdb氨基酸的序号
                    index_list.append(index)
                    coordinate_list.append([float(lne[30:38]), float(lne[38:46]), float(lne[46:54])])
                    chain_number_list.append(chain_number)
                elif(int(lne[22:26])<index or lne[21]!=chain_id):
                    atom_index = int(lne[6:11])
                    chain_id = lne[21]
                    chain_number = chain_number+1
                    count = count+1
                    count_list.append(count)
                    index = int(lne[22:26])
                    index_list.append(index)
                    coordinate_list.append([float(lne[30:38]), float(lne[38:46]), float(lne[46:54])])
                    chain_number_list.append(chain_number)
            else:
                continue

        return index_list, coordinate_list, count_list, chain_number_list



def calculate_interface_index(index_list, coordinate, count_list, chain_list):
    DB = dict()
    for i in range(chain_list[-1]):
        chain = 'chain' + str(i + 1)
        DB[chain] = dict()
    # print(DB)
    for i in range(len(chain_list)):
        chain = 'chain' + str(chain_list[i])
        DB[chain][count_list[i]] = dict()
        DB[chain][count_list[i]]['index'] = index_list[i]
        DB[chain][count_list[i]]['coordinate'] = coordinate[i]

    key_list = list(DB.keys())
    # pprint(DB)
    chain_list_all = [[] for i in range(len(key_list))]
    for i in range(len(key_list)):
        key_list2 = list(DB[key_list[i]])
        for j in range(len(key_list2)):
            count = key_list2[j]
            # print(count)
            x = DB[key_list[i]][key_list2[j]]['coordinate'][0]
            y = DB[key_list[i]][key_list2[j]]['coordinate'][1]
            z = DB[key_list[i]][key_list2[j]]['coordinate'][2]
            chain_list_all[i].append([count, x, y, z])

    interface_index = set()
    interface_mask = [0 for i in range(len(count_list))]

    for i in range(len(chain_list_all)):
        for j in range(i + 1, len(chain_list_all)):
            for k in range(len(chain_list_all[i])):
                for f in range(len(chain_list_all[j])):
                    residue_one = chain_list_all[i][k][1:]
                    residue_two = chain_list_all[j][f][1:]
                    one_index = chain_list_all[i][k][0]
                    two_index = chain_list_all[j][f][0]
                    distance = calculate_distacne(residue_one, residue_two)
                    if (distance < 8):
                        interface_index.add(one_index)
                        interface_index.add(two_index)

    index_sort = sorted(list(interface_index))
    for i in range(len(index_sort)):
        index = index_sort[i]
        interface_mask[index - 1] = 1

    return index_sort, interface_mask



def complex2map(pdbfile):
    index_list, coordinate, count_list, chain_list = calculate_interface(pdbfile)

    DB = dict()  # Store protein chains, location and index information
    for i in range(chain_list[-1]):
        chain = 'chain' + str(i + 1)
        DB[chain] = dict()
    # print(DB)
    for i in range(len(chain_list)):
        chain = 'chain' + str(chain_list[i])
        DB[chain][count_list[i]] = dict()
        DB[chain][count_list[i]]['index'] = index_list[i]
        DB[chain][count_list[i]]['coordinate'] = coordinate[i]

    key_list = list(DB.keys())
    # pprint(DB)
    chain_list_all = [[] for i in range(len(key_list))]
    for i in range(len(key_list)):
        key_list2 = list(DB[key_list[i]])
        for j in range(len(key_list2)):
            count = key_list2[j]
            # print(count)
            x = DB[key_list[i]][key_list2[j]]['coordinate'][0]
            y = DB[key_list[i]][key_list2[j]]['coordinate'][1]
            z = DB[key_list[i]][key_list2[j]]['coordinate'][2]
            chain_list_all[i].append([count, x, y, z])

    chain2map = {}
    for i in range(len(chain_list_all)):
        chain2map[i] = []
        for j in range(0, len(chain_list_all)):
            if i == j:
                continue
            compelx_map = np.full((len(chain_list_all[i]), len(chain_list_all[j])), np.nan)
            for k in range(len(chain_list_all[i])):
                for f in range(len(chain_list_all[j])):
                    residue_one = chain_list_all[i][k][1:]
                    residue_two = chain_list_all[j][f][1:]
                    one_index = chain_list_all[i][k][0]
                    two_index = chain_list_all[j][f][0]
                    distance = calculate_distacne(residue_one, residue_two)
                    compelx_map[k][f] = distance
            chain2map[i].append(compelx_map)

    return chain2map



def generate_interface_info(pdbfile):
    index_list, coordinate, count_list, chain_list = calculate_interface(pdbfile)
    interface_index, interface_mask = calculate_interface_index(index_list, coordinate, count_list, chain_list)
    interface_res_info = []
    if len(interface_index)==0:
        return interface_res_info, interface_mask
    else:
        for i in range(len(interface_index)):
            chains_name_list = calculate_num_chains(pdbfile)
            count_index = interface_index[i] - 1
            chain_index = chain_list[count_index] - 1
            chain_name = chains_name_list[chain_index].get_id()
            index = index_list[count_index]
            interface_res_info.append(f'{chain_name}:{index}')
    return interface_res_info, interface_mask



## We have improved the lddt metric for calculating protein complexes

def get_complex_LDDT(ref_chain2map, pred_chain2map, R=15, T=8, sep_thresh=-1, T_set=[0.5, 1, 2, 4], precision=4):
    '''
    Mariani V, Biasini M, Barbato A, Schwede T.
    lDDT: a local superposition-free score for comparing protein structures and models using distance difference tests.
    Bioinformatics. 2013 Nov 1;29(21):2722-8.
    doi: 10.1093/bioinformatics/btt473.
    Epub 2013 Aug 27.
    PMID: 23986568; PMCID: PMC3799472.
    '''
    import pandas as pd

    # return a 1D boolean array indicating where the distance in the
    # upper triangle meets the threshold comparison
    def get_dist_thresh_b_indices(dmap_flat, thresh, comparator):
        assert comparator in {'gt', 'lt', 'ge', 'le'}, "ERROR: Unknown comparator for thresholding!"
        if comparator == 'gt':
            threshed = dmap_flat > thresh
        elif comparator == 'lt':
            threshed = dmap_flat < thresh
        elif comparator == 'ge':
            threshed = dmap_flat >= thresh
        elif comparator == 'le':
            threshed = dmap_flat <= thresh
        return threshed

    # Helper for number preserved in a threshold
    def get_n_preserved(ref_flat, mod_flat, thresh):
        err = np.abs(ref_flat - mod_flat)
        n_preserved = (err < thresh).sum()
        return n_preserved

    lddt_scores = []
    interface_mask = []
    chain_list = []
    for chainid in pred_chain2map.keys():
        pred_map_list = pred_chain2map[chainid]
        ref_map_list = ref_chain2map[chainid]

        true_map = np.concatenate(ref_map_list, axis=1)

        pred_map = np.concatenate(pred_map_list, axis=1)

        for i in range(len(true_map)):
            chain_list.append(chainid)
            true_flat_map = true_map[i]
            pred_flat_map = pred_map[i]

            # Find set L
            R_thresh_indices = get_dist_thresh_b_indices(true_flat_map, R, 'lt')
            # print(R_thresh_indices)
            interface_thresh_indices = get_dist_thresh_b_indices(pred_flat_map, T, 'lt')
            # print(interface_thresh_indices)
            L_indices = R_thresh_indices

            true_flat_in_L = true_flat_map[L_indices]
            # print(true_flat_in_L)
            pred_flat_in_L = pred_flat_map[L_indices]
            # print(pred_flat_in_L)

            # Number of pairs in L
            L_n = L_indices.sum()
            # print(L_n)
            interface_n = interface_thresh_indices.sum()


            if interface_n > 0:
                interface_mask.append(1)
            else:
                interface_mask.append(0)

            # Calculated lDDT
            preserved_fractions = []
            for _thresh in T_set:
                _n_preserved = get_n_preserved(true_flat_in_L, pred_flat_in_L, _thresh)

                if L_n == 0:
                    _f_preserved = 0
                else:
                    _f_preserved = _n_preserved / L_n
                preserved_fractions.append(_f_preserved)

            # preserved_fractions

            lDDT = np.mean(preserved_fractions)
            if precision > 0:
                lDDT = round(lDDT, precision)

            # print(i,": ",preserved_fractions, lDDT)
            lddt_scores.append(lDDT)

    return lddt_scores



def get_complex_interface(pred_chain2map, T=8):

    import pandas as pd

    # return a 1D boolean array indicating where the distance in the
    # upper triangle meets the threshold comparison
    def get_dist_thresh_b_indices(dmap_flat, thresh, comparator):
        assert comparator in {'gt', 'lt', 'ge', 'le'}, "ERROR: Unknown comparator for thresholding!"
        if comparator == 'gt':
            threshed = dmap_flat > thresh
        elif comparator == 'lt':
            threshed = dmap_flat < thresh
        elif comparator == 'ge':
            threshed = dmap_flat >= thresh
        elif comparator == 'le':
            threshed = dmap_flat <= thresh
        return threshed


    interface_mask = []
    chain_list = []
    for chainid in pred_chain2map.keys():
        pred_map_list = pred_chain2map[chainid]

        pred_map = np.concatenate(pred_map_list, axis=1)

        for i in range(len(pred_map)):
            chain_list.append(chainid)

            pred_flat_map = pred_map[i]

            interface_thresh_indices = get_dist_thresh_b_indices(pred_flat_map, T, 'lt')

            interface_n = interface_thresh_indices.sum()
            # print(interface_n)

            if interface_n > 0:
                interface_mask.append(1)
            else:
                interface_mask.append(0)

    return interface_mask



def prepare_data_information(inputfile, outfile):


    target_list = []
    complex_all = []
    mask = []
    interface_residue = []
    for target in os.listdir(inputfile):
        target_path = os.path.join(inputfile, target)
        for pdb in os.listdir(target_path):
            complex_name = pdb
            complex_all.append(complex_name)
            file_path = os.path.join(inputfile, target, pdb)
            target_list.append(target)
            # pred_chain2map = complex2map(file_path)
            interface_res_info, interface_mask = generate_interface_info(file_path)

            # interface_mask = get_complex_interface(pred_chain2map)

            mask.append(interface_mask)
            interface_residue.append(interface_res_info)

    dataframe = pd.DataFrame({'target': target_list, 'model': complex_all, 'interface_residue':interface_residue, 'interface_mask': mask})
    dataframe.to_csv(outfile, index=False, sep=',')




def prepare_information_with_native(inputfile, native_file, outfile):


    target_list = []
    complex_all = []
    scores = []
    mask = []
    interface_residue = []
    ref_chain2map = complex2map(native_file)
    for target in os.listdir(inputfile):
        target_path = os.path.join(inputfile, target)
        for pdb in os.listdir(target_path):
            complex_name = pdb
            complex_all.append(complex_name)
            file_path = os.path.join(inputfile, target, pdb)
            target_list.append(target)
            pred_chain2map = complex2map(file_path)
            lddt_complex = get_complex_LDDT(ref_chain2map, pred_chain2map, R=30, T=8, sep_thresh=-1,
                                            T_set=[0.5, 1, 2, 4], precision=4)
            interface_res_info, interface_mask = generate_interface_info(file_path)
            scores.append(lddt_complex)
            interface_residue.append(interface_res_info)
            mask.append(interface_mask)


    dataframe = pd.DataFrame({'target': target_list, 'model': complex_all, 'lddt_complex':scores , 'interface_residue':interface_residue, 'interface_mask': mask})
    dataframe.to_csv(outfile, index=False, sep=',')




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='prepare data information')

    parser.add_argument('-i', '--input_dir', type=str, required=True, help="number of pdbs to use for each batch")
    parser.add_argument('-n', '--native', help='native PDB file ', default=None)
    parser.add_argument('-o', '--outfile', type=str, required=True, help="saving complex information")

    args = parser.parse_args()

    inputfile = args.input_dir
    native_pdb = args.native
    outfile = args.outfile

    if args.native != None:
        prepare_information_with_native(inputfile, native_pdb, outfile)

    else:
        prepare_data_information(inputfile, outfile)


