from os.path import join
import math
import numpy as np
from pprint import pprint
import os
import json

# Calculate the Euclidean distance between amino acids
def calculate_distacne(residue_one, residue_two):
    sum = 0
    distance_list = [residue_one[i]-residue_two[i] for i in range(len(residue_one))]
    for i in range(len(distance_list)):
        sum += math.pow(distance_list[i], 2)
    distance = np.sqrt(sum)
    return  distance


def calculate_interface(data_path, pdbfile):
    count = 0
    chain_number = 0
    index = 0
    atom_index = 0
    chain_id = '#'
    count_list = []
    index_list =[]
    coordinate_list = []
    chain_number_list = []
    pdb = join(data_path, pdbfile)
    with open(pdb) as pdb:
        for lne in pdb:
            # print((lne[13:16]))
            if(lne[13:16]=='CB '):
                if(int(lne[22:26]) > index and lne[21]==chain_id):
                    # atom_index = int(lne[6:11])
                    count = count+1
                    count_list.append(count)
                    index = int(lne[22:26])
                    index_list.append(index)
                    coordinate_list.append([float(lne[30:38]), float(lne[38:46]), float(lne[46:54])])
                    chain_number_list.append(chain_number)
                elif(int(lne[22:26])<index  or  lne[21]!=chain_id):
                    # atom_index = int(lne[6:11])
                    chain_id = lne[21]
                    chain_number = chain_number+1
                    count = count+1
                    count_list.append(count)
                    index = int(lne[22:26])
                    index_list.append(index)
                    coordinate_list.append([float(lne[30:38]), float(lne[38:46]), float(lne[46:54])])
                    chain_number_list.append(chain_number)

            elif(lne[13:16]=='CA ' and lne[17:20]=='GLY'):
                if (int(lne[22:26]) > index and lne[21]==chain_id):
                    # atom_index = int(lne[6:11])
                    count = count + 1
                    count_list.append(count)
                    index = int(lne[22:26])
                    index_list.append(index)
                    coordinate_list.append([float(lne[30:38]), float(lne[38:46]), float(lne[46:54])])
                    chain_number_list.append(chain_number)
                elif (int(lne[22:26]) < index or lne[21]!=chain_id):
                    # atom_index = int(lne[6:11])
                    chain_id = lne[21]
                    chain_number = chain_number + 1
                    count = count + 1
                    count_list.append(count)
                    index = int(lne[22:26])
                    index_list.append(index)
                    coordinate_list.append([float(lne[30:38]), float(lne[38:46]), float(lne[46:54])])
                    chain_number_list.append(chain_number)
            else:
                continue
        return index_list, coordinate_list, count_list, chain_number_list



def generate_interface_mask(index_list, coordinate, count_list, chain_list):
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

    key_list = list(DB.keys())  # Return to the list of protein chains
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

    mask = set()  # Create a collection for saving the location of the protein interface
    index_mask = [0 for i in range(len(count_list))]
    # print(mask)
    # pprint(chain_list_all[1])
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
                        mask.add(one_index)
                        mask.add(two_index)

    mask_sort = sorted(list(mask))
    # for i in range(len(mask_sort)):
    #     index = mask_sort[i]
    #     index_mask[index - 1] = 1

    return mask_sort





