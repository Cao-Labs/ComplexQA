import tensorflow as tf
from scipy.spatial.distance import pdist, squareform
import random
import time
import numpy as np
import pickle
from collections import defaultdict
import pandas as pd
import json



def get_feature_ranks(pathToFeatureScores):
    '''
    This method returns the ordered feature ranks, precomputed and saved

    Parameters:
    -------------
    pathToFeatureScores: string
        This is a hard-coded path, points to the Pearson_Correlation_Features.txt

    Returns:
    -----------
    list: [string]
        A list of strings of the feature indexes ranked from best to worst by pearson correlation
    '''
    # load and parse the feature ranks

    raw_data = open(pathToFeatureScores).read()
    feature_ranks = []
    for line in raw_data.split('\n'):
        line_data = line.split('\t')
        feature_ranks.append(line_data[0])

    return feature_ranks


class ProteinComplexQAData_Generator(tf.keras.utils.Sequence):
    def __init__(self, data_path='./', protein_info_pickle='./model_summary_mask_2022June.csv', min_seq_size=0,
                 max_seq_size=10000, batch_size=1, max_msa_seq=100000, max_id_nums=1000000):

        self.data_path = data_path
        self.protein_info = protein_info_pickle
        self.max_id_nums = max_id_nums
        self.min_seq_size = min_seq_size
        self.max_seq_size = max_seq_size
        self.batch_size = batch_size
        self.max_msa_seq = max_msa_seq

        # self.zoomqa_node_pickle_path = os.path.join(self.data_path, "Features/") #zdock/1zwh/complext_1_features/Input/1zwh.pkl
        # self.contact_json_path = os.path.join(self.data_path, "Json_data/")

        self.zoomqa_node_pickle_path = self.data_path
        self.contact_json_path = self.data_path

        self.id2seq,  self.id2interfacemask = self.get_filenames(head=self.max_id_nums)
        self.id_list = list(self.id2seq.keys())  # 1Z5Y.complex.1.pdb

        ### speed up running using batch list by length
        batchsize = 10
        self.seq2id = defaultdict(list)
        for seq_id, seq_len in sorted(self.id2seq.items()):
            self.seq2id[seq_len].append(seq_id)

        protein_model_id_list = []
        for seq_len in self.seq2id.keys():
            if seq_len < 400:
                batchsize = 50
            elif seq_len < 600:
                batchsize = 30
            elif seq_len < 800:
                batchsize = 10
            else:
                batchsize = 5

            if self.batch_size == 'single':
                batchsize = 1
            comb = self.seq2id[seq_len]
            batch_len = int(len(comb) / batchsize) + 1
            for index in range(batch_len):
                batch_list = comb[index * batchsize: (index + 1) * batchsize]
                protein_model_id_list.append(batch_list)

        self.protein_model_id_list = protein_model_id_list

    def on_epoch_begin(self):
        self.indexes = np.arange(len(self.protein_model_id_list))

    def __len__(self):
        # return int(len(self.id_list) / self.batch_size)
        return int(len(self.protein_model_id_list) / self.batch_size)

    def __getitem__(self, index):
        batch_id_list = self.protein_model_id_list[index]
        # print(batch_id_list)
        if len(batch_id_list) == 0:
            batch_id_list = random.sample(self.protein_model_id_list, 1)[0]

        # batch_id_list = self.id_list[index * self.batch_size: (index + 1) * self.batch_size]
        node_feat_batch, model_contact_batch, pdb_distance_pair_batch = self.collect_data(
            batch_id_list)

        return node_feat_batch, model_contact_batch, pdb_distance_pair_batch

    def __call__(self):
        self.i = 0
        return self

    def get_filenames(self, head=10000000):
        """
        - returns all file names without extension
        - output: lists of file names
        """
        # files_no_ext = [".".join(f.split(".")[:-1]) for f in os.listdir(self.path_msa) \
        #                if isfile(join(self.path_msa, f))]
        model2seq = {}
        model2interfacemask = {}


        start = time.time()
        train_dataframe = pd.read_csv(self.protein_info, sep=',')

        protein_list = set()
        for i in train_dataframe.index:
            pdb_id = train_dataframe['target'][i]
            model_id = train_dataframe['model'][i]
            # model_lddt = eval(train_dataframe['lddt_complex'][i])
            # model_rmsd = train_dataframe['IRMSD'][i]
            model_interfacemask = eval(train_dataframe['interface_mask'][i])

            # print(len(model_lddt),type(model_lddt),'->',model_lddt)
            seq_len = len(model_interfacemask)

            protein_list.add(pdb_id)

            if seq_len <= self.max_seq_size and seq_len >= self.min_seq_size:
                model2seq[model_id] = seq_len
                # model2lddt[model_id] = model_lddt
                # model2rmsd[model_id] = model_rmsd

                model2interfacemask[model_id] = model_interfacemask

            if len(model2seq) >= head:
                break

        end = time.time()
        print("Finish loading ", self.protein_info, " takes ", end - start)

        return model2seq, model2interfacemask

    def get_node_edges_features(self, pdb_name):
        # print("pdb_name: ",pdb_name)
        # pdb = pdb_name.split(':')[0] + '_' + pdb_name.split(':')[1]

        # 1Z5Y.complex.1.pdb
        target_id = pdb_name.split('_')[0]
        pdb_index = pdb_name.split('.')[0]
        node_features = []

        feature_path = self.data_path + '/' + pdb_index + '.pdb.pkl'
        feature_ranks = get_feature_ranks('Pearson_Correlation_Features.txt')

        with open(feature_path, 'rb') as f:
            data = pickle.load(f)
            distance_map = []
            local_scores = []
            for index in data:
                aa_density_change = data[index]['aa_density_change']
                aa_density = aa_density_change.flatten('F')
                avg_distance = data[index]['average_distance']
                hydro_change = data[index]['hydro_change']
                iso_change = data[index]['iso_change']
                mass_change = data[index]['mass_change']
                per_contact = data[index]['percent_contact']
                sol_change = data[index]['sol_change']
                std_dev_distance = data[index]['std_dev_distance']
                structure_contact_matrix = data[index]['structure_contact_matrix']
                structure_contact = structure_contact_matrix.flatten('F')
                X = np.concatenate(
                    [aa_density, avg_distance, hydro_change, iso_change, mass_change, per_contact, sol_change,
                     std_dev_distance, structure_contact]).astype(None)
                filter_feature = []
                for feature_number in feature_ranks[:300]:
                    filter_feature.append(X[int(feature_number)])
                node_features.append(filter_feature)
                distance_map.append(data[index]['contact_map'])


            # print("pdb_name: ",pdb_name)
            # local_scores = self.id2lddt[pdb_name]
            local_interfacemask = self.id2interfacemask[pdb_name]
            # dockq_score = self.id2dockq[pdb_name]
            return np.array(node_features), np.array(distance_map), np.array(
                local_interfacemask)

    def collect_data(self, batch_id_list):
        # ['1Z5Y.complex.1.pdb']

        # target_feat_batch:  (None, L_max, 21)

        # (Part I): load sequence data
        # get maximum size from this batch
        max_seq_len = 0
        for i in range(len(batch_id_list)):
            try:
                if self.id2seq[batch_id_list[i]] > max_seq_len:
                    max_seq_len = self.id2seq[batch_id_list[i]]
                # interface_size = (np.array(self.id2interfacemask[batch_id_list[i]])==1).sum()
                # if interface_size > max_seq_len:
                #    max_seq_len = interface_size
            except:
                print("error batch_id_list[i]: ", batch_id_list[i])
                exit(-1)

        self.pad_size = max_seq_len

        ## qa features
        pdb_node_batch = np.full((len(batch_id_list), self.pad_size, 300), 0.0)
        model_contact_batch = np.full((len(batch_id_list), self.pad_size, self.pad_size, 1), 0.0)
        pdb_distance_pair_batch = np.full((len(batch_id_list), self.pad_size, self.pad_size, 1), 0.0)
        model_aa_lddt_batch = np.full((len(batch_id_list), self.pad_size, 1), 0.0)
        # model_aa_dockq_batch = np.full((len(batch_id_list), 1), 0.0)
        model_aa_interfacemask_batch = np.full((len(batch_id_list), self.pad_size, 1), 0.0)

        updated_max_seq_len = 0
        ### here we can optimize file loading for the models from same target
        target2models = defaultdict(list)
        target2models_seq = defaultdict(list)
        for i in range(len(batch_id_list)):
            try:
                info = batch_id_list[i].split('.')  # 1Z5Y.complex.1.pdb
            except:
                print("batch_id_list[i]: ", batch_id_list[i])
                exit(-1)
            seq_len = self.id2seq[batch_id_list[i]]
            # target_id = info[0]  # '1Z5Y'
            target_id = info[0].split('_')[0]  # '1Z5Y'

            # target_model = info[1]  # 1Z5Y.complex.1.pdb
            target2models[target_id].append(batch_id_list[i])
            target2models_seq[target_id].append(seq_len)

        ### load models per target
        loaded_count = 0
        for target_id in target2models.keys():

            models = target2models[target_id]  # ['1Z5Y.complex.1.pdb','1Z5Y.complex.1.pdb']

            seq_len_unique = np.unique(target2models_seq[target_id])
            # print(target_id,'->',seq_len_unique)

            if len(seq_len_unique) > 1:
                print('Warning: pdb structures from same target have different length, check it')
                exit(-1)
            seq_len = seq_len_unique[0]

            ##### (part 2) start loading each model of target the protein quality assessment data
            for model in models:  # 1Z5Y.complex.1.pdb
                node_features, pdb_distance_feature, node_interface = self.get_node_edges_features(
                    model)  ## L * 233

                # print("pdb_distance_feature: ",pdb_distance_feature.shape)
                l = len(pdb_distance_feature)

                if l != seq_len:
                    print("warning: pdb length is not equal to sequence length", l, "!=", seq_len)

                if l != len(node_interface):
                    print("warning: pdb length is not equal to sequence interface length", l, "!=", node_interface)

                # if l != len(node_lddt):
                #     print("warning: pdb length is not equal to node_lddt length", l, "!=", node_lddt)

                ### select interface neighbors' features
                contact_feature = pdb_distance_feature.copy()
                contact_feature[contact_feature < 10] = 1
                contact_feature[contact_feature >= 10] = 0
                contact_feature = contact_feature.astype(np.uint8)

                # set diagnol to 1?
                np.fill_diagonal(contact_feature, 1)

                model_distance = np.multiply(contact_feature, pdb_distance_feature)

                ## get interface neighbors
                interface = set(np.where(node_interface == 1)[0].flatten())
                for row in range(len(interface)):
                    expand_set = set(np.where(pdb_distance_feature[node_interface != 0, :][row] < 0)[0].flatten())
                    interface = interface.union(expand_set)

                node_interface[list(interface)] = 1
                ## filter by interface mask
                interface_mask = np.array(node_interface).astype(float)

                # filter_lddt_score = interface_mask * self.id2lddt[model]
                # filter_lddt_score = filter_lddt_score[interface_mask != 0]

                interface_size = np.sum(interface_mask == 1)
                # interface_size = len(filter_lddt_score)

                filter_node_features = node_features[interface_mask != 0, :]

                filter_contact_feature = contact_feature[interface_mask != 0, interface_mask != 0]
                filter_model_distance = model_distance[interface_mask != 0, interface_mask != 0]

                if updated_max_seq_len < interface_size:
                    updated_max_seq_len = interface_size

                pdb_node_batch[loaded_count, 0:interface_size, :] = filter_node_features
                model_contact_batch[loaded_count, 0:interface_size, 0:interface_size, 0] = filter_contact_feature
                pdb_distance_pair_batch[loaded_count, 0:interface_size, 0:interface_size, 0] = filter_model_distance
                # model_aa_lddt_batch[loaded_count, 0:interface_size, 0] = filter_lddt_score
                # model_aa_dockq_batch[loaded_count, 0] = dockq
                loaded_count += 1
                # label.append(temp_list)

        # (None, L_max, 300)
        pdb_node_batch = pdb_node_batch[0:loaded_count, 0:updated_max_seq_len, :]

        model_contact_batch = model_contact_batch[0:loaded_count, 0:updated_max_seq_len, 0:updated_max_seq_len, :]
        pdb_distance_pair_batch = pdb_distance_pair_batch[0:loaded_count, 0:updated_max_seq_len, 0:updated_max_seq_len,
                                  :]

        return pdb_node_batch.astype(np.float32), model_contact_batch.astype(
            np.float32), pdb_distance_pair_batch.astype(np.float32)
