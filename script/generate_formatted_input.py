import numpy as np


def get_node_edges_features(data):

    node_features = []
    distance_map = []
    for index in data:
        aa_encoded = data[index]['aa_encoded']
        ss_encoded = data[index]['ss_encoded']
        iso_change = data[index]['iso_change']
        hydro_change = data[index]['hydro_change']
        # mass_change = data[index]['mass_change']
        sol_change = data[index]['sol_change']
        psi_phi = np.array(data[index]['psiphi'])
        aa_iso = np.array(data[index]['aa_iso'])
        std_dev_distance = data[index]['std_dev_distance']
        X = np.concatenate(
            [aa_encoded, ss_encoded, hydro_change, iso_change, std_dev_distance, sol_change, psi_phi,
             aa_iso]).astype(None)
        node_features.append(X)
        distance_map.append(data[index]['contact_map'])


    return np.array(node_features), np.array(distance_map)


def parse_server_data(server_data):
    node_features, distance_map = get_node_edges_features(server_data)
    pdb_len = len(distance_map[0])
    pdb_node_batch = np.full((1, pdb_len, 233), 0.0)
    model_contact_batch = np.full((1, pdb_len, pdb_len, 1), 0.0)
    pdb_distance_pair_batch = np.full((1, pdb_len, pdb_len, 1), 0.0)

    contact_feature = distance_map.copy()
    contact_feature[contact_feature < 10] = 1
    contact_feature[contact_feature >= 10] = 0
    contact_feature = contact_feature.astype(np.uint8)

    # set diagnol to 1?
    np.fill_diagonal(contact_feature, 1)

    model_distance = np.multiply(contact_feature, distance_map)

    pdb_node_batch[:, :, :] = node_features
    model_contact_batch[:, :, :, 0] = contact_feature
    pdb_distance_pair_batch[:, :, :, 0] = model_distance

    return pdb_node_batch.astype(np.float32), model_contact_batch.astype(np.float32), \
           pdb_distance_pair_batch.astype(np.float32)