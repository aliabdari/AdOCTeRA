import os, time
import pickle

import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


def compute_similarity(list_sentences, slm_model):
    if slm_model == 'distilroberta':
        model = SentenceTransformer('paraphrase-distilroberta-base-v2')
    else:
        print('Model cannot be found')
        exit(0)

    model.eval()
    relations = dict()
    counter = 0
    test_sent = 'This is a test sent'
    test_emb = model.encode(test_sent)
    embeddings = torch.empty(len(list_sentences), test_emb.shape[0])
    print('Obtaining embeddings of the sentences ...')
    with torch.no_grad():
        for i, sent in tqdm(enumerate(list_sentences), total=len(list_sentences)):
            embeddings[i, :] = model.encode(sent, convert_to_tensor=True)
    print('Calculating Cosine Similarity ...')
    list_sims = []
    for i, h1 in tqdm(enumerate(list_sentences), total=len(list_sentences)):
        for j, h2 in enumerate(list_sentences[i + 1:]):
            cosine_sim = util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
            relations[(i, i + j + 1)] = cosine_sim
            list_sims.append(cosine_sim)
            counter += 1

    print('Statistics of the obtained similarities')
    min_num = min(list_sims)
    max_num = max(list_sims)
    print('min similarity = ', min_num)
    print('max similarity = ', max_num)
    print('avg similarity = ', sum(list_sims) / len(list_sims))

    relations_norm = dict()
    list_sims_norm = []
    for i in range(6081):
        for j in range(i + 1, 6081):
            relations_norm[(i, j)] = (relations[(i, j)] - min_num) / (max_num - min_num)
            list_sims_norm.append(relations_norm[(i, j)])

    print('Statistics of the obtained similarities after normalization')
    print('min similarity = ', min(list_sims_norm))
    print('max similarity = ', max(list_sims_norm))
    print('avg similarity = ', sum(list_sims_norm) / len(list_sims_norm))

    print('counter of processed file', counter)

    return relations_norm


def process_sentences(root_descs_path, list_apartments, slm_model):
    files = [x['json_file'] for x in list_apartments]
    list_second_sents = []
    for f in files:
        with open(root_descs_path + os.sep + f + '.txt', 'r') as file:
            list_second_sents.append(file.readlines()[0].split('.')[1])
    return compute_similarity(list_second_sents, slm_model)


if __name__ == '__main__':
    slm_model = 'distilroberta'
    selected_model_ind = 0
    file = open('../apartments_data/apartments_data.pkl', 'rb')
    houses = pickle.load(file)
    root_descs_path = '../../FArMARe/descriptions'
    start_time = time.time()
    relations_norm = process_sentences(root_descs_path, houses, slm_model)
    with open(f'../scenes_relevances/relevance_slm_normalized.pkl', 'wb') as pickle_file:
        pickle.dump(relations_norm, pickle_file)
    print('Elapsed time: ', time.time() - start_time)
