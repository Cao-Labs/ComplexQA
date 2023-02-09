'''
This file is responsible for generating the contact distance statistics

'''

import os
import sys
import json
import numpy as np

from os.path import join

RADII = RADII = list(range(5, 56, 1))

def get_contact_stats(casp_input, index):
    '''
    This method gets all of the contact statistics, the average distance vector, std_dev vector, and the percent contact vector.

    - 'average_distance' a length 21 vector representing the average distance between the center and all acids in contact with the center.
       This is normalized by the radius of gyration (the longest distance still considered by the contact)
    - 'std_dev_distance' a length 21 vector representing the std_deviation of the distance between the center and all acids in contact
       with the center. The distances used for std dev calculation are normalized by the radius of gyration (the longest distance still considered by the contact),
       so the std dev will be between 0 and 1
    - 'percent_contact' a length 21 vector representing the total percentage of the protein that is in contact with the center as the
       radius increases (e.g. index 0 is the percentage of the protein in contact with the center amino acid with radius 5)

    Parameters:
    ----------
    casp_input: dictionary
        This dictionary comes from one server prediction for one target (one JSON file) from Dr. Cao's CASP JSON database

    index: int
        This is the integer representing the index in the sequence that we are considering as the center

    Returns:
    -----------
    dictionary: 'average_distance': [length 21 vector], 'std_dev_distance':[length 21 vector], 'percent_contact':[length 21 vector]
        This dictionary returns the vectors that correlate to the data described above. The average is the average distance (normalized by radius of gyration) of residues
        in contact with the center
        The std_deviation vector is a length 21 vector representing the std_deviation of the distance between the center and all acids in contact
        with the center. The distances used for std dev calculation are normalized by the radius of gyration (the longest distance still considered by the contact),
        so the std dev will be between 0 and 1
        the percent_contact vector is a length 21 vector representing the total percentage of the protein that is in contact with the center as the
        radius increases (e.g. index 0 is the percentage of the protein in contact with the center amino acid with radius 5)


    '''
    aa_adjacency_list = casp_input['ContactMap'][index]

    local_contact_info = _get_local_contacts(aa_adjacency_list)

    ave_vector = _get_ave(local_contact_info)
    std_vector = _get_std_ave(local_contact_info)
    percent_contact = _get_percent_contact(local_contact_info, casp_input['aa'])

    return {'average_distance': ave_vector, 'std_dev_distance':std_vector, 'percent_contact':percent_contact}

def _get_local_contacts(aa_adjacency_list):
    '''
    This method compiles the dictionary with every contact for every radius for a target residue in the sequence

    Parameters:
    ------------
    aa_adjacency_list: list[length of sequence]
        This is a row of the contact list, index 0 represents the contact distance for the acid we are considering to the residue at index 0. Index 1 is the contact distance between
        the residue we are considering and residue at index 2

    Returns:
    ---------
    dictionary:
        Returns a dictionary with the keys being the radius and the values being a list of the contact distances of the residues that are in contact


    '''
    local_contact_info = {}
    for radius in RADII:
        local_contact_info[radius] = []

    for contact_distance in aa_adjacency_list:
        threshhold_radius = _get_category(contact_distance)
        if threshhold_radius is not None:
            #TODO: the original was 26
            for radius_category in range(threshhold_radius, 56, 1):
                local_contact_info[radius_category].append(contact_distance)
    return local_contact_info

def _get_ave(local_contact_info):
    """
    This method gets the ave distance vector for the target residue. Distances are normalized by the radius of gyration

    Parameters:
    ----------
    local_contact_info: dictionary
        This is a dictionary generated by the _get_local_contacts method. For each key (the radius) we are given a list of distances representing every distance for every acid within
        a radius (the key) of the target residue

    Returns:
    ---------
    np.ndarray(21, ):
        This vector represents the ave distance for all the residues in contact for a specific radius. index 0 is radius 5 and index 1 is radius 6 and so on.

    """
    ave_vector = []

    for radius, distance_list in sorted(local_contact_info.items()):
        if len(distance_list) < 1:
            ave_vector.append(0.0)
            continue
        radius_ave = np.mean(np.asarray(distance_list))
        radius_of_gyration = np.max(np.asarray(distance_list))
        if radius_of_gyration > 0: 
            ave_vector.append(radius_ave/radius_of_gyration)
        else: 
            ave_vector.append(0)
    return np.asarray(ave_vector)

def _get_std_ave(local_contact_info):
    """
    This method gets the std deviation of the  distance vector for the target residue. Distances are normalized by radius of gyration before being considered for std deviation.

    Parameters:
    ----------
    local_contact_info: dictionary
        This is a dictionary generated by the _get_local_contacts method. For each key (the radius) we are given a list of distances representing every distance for every acid within
        a radius (the key) of the target residue

    Returns:
    ---------
    np.ndarray(21, ):
        This vector represents the std deviation of the distance for all the residues in contact for a specific radius. index 0 is radius 5 and index 1 is radius 6 and so on.

    """
    std_vector = []
    for radius, distance_list in sorted(local_contact_info.items()):
        norm_distance_list = []
        norm_max = np.max(distance_list)
        norm_min = np.min(distance_list)
        for dist in distance_list:
            if norm_max == norm_min:
                norm_distance_list.append(0.0)
                continue
            norm_distance_list.append((dist - norm_min)/(norm_max- norm_min))


        radius_std_dev = np.std(np.asarray(norm_distance_list))
        std_vector.append(radius_std_dev)
    return np.asarray(std_vector)

def _get_percent_contact(local_contact_info, sequence):
    """
    This method gets the precentage of the total protein in contact with the target amino acid as the radius increases.

    Parameters:
    ----------
    local_contact_info: dictionary
        This is a dictionary generated by the _get_local_contacts method. For each key (the radius) we are given a list of distances representing every distance for every acid within
        a radius (the key) of the target residue

    Returns:
    ---------
    np.ndarray(21, ):
        This vector represents the percentage of the protein in contact with the center amino acid as the radius increases.

    """

    percent_contact_vector = []
    for radius, dist_list in sorted(local_contact_info.items()):
        total_contacts = len(dist_list)
        percent_contact_vector.append(total_contacts/len(sequence))

    return np.asarray(percent_contact_vector)


def _get_category(distance):
    '''
    This method is responsible for finding the distance category a distance falls into

    Parameters:
    ----------
    distance: float
        This represents the distance in angstroms a target amino acid is away from the current center

    Returns:
    --------
    int/None
        This integer represents the radius category that the distance falls into (e.g. a radius of 6.7 would return 7 meaning
        that the target amino acid is within a radius of 7 angstroms). It will return None if the distance is greater
        than the threshhold set by RADII

    '''
    for radius in RADII:
        if distance <= radius:
            return radius
    return None

if __name__ == "__main__":
    pathToCASP = '/media/kyle/IronWolf/CASP_ALL/'

    target_paths = []

    for potential_path in os.listdir(pathToCASP):
        if 'tmp' in potential_path:
            continue
        target_paths.append(join(pathToCASP, potential_path))


    data = json.load(open(sorted(target_paths)[0]))


    for server_name, server_data in data.items():
        #test feature generation here
        ret = get_contact_stats(server_data, 0)
        print(server_name)
        for k,v in ret.items():
            print(k, v)
