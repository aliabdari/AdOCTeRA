import pickle
from tqdm import tqdm
from collections import Counter
import time


def find_objects(house):
    furniture_list = []
    for room in house['rooms']:
        furniture_list.extend(obj['category'] for obj in room['objects'])
    return furniture_list


def calculate_iou(list1, list2):
    if len(list1) == 0 and len(list2) == 0:
        return 1
    counter_rooms_1 = Counter(list1)
    counter_rooms_2 = Counter(list2)
    counter = counter_rooms_1 & counter_rooms_2
    return sum(list(counter.values())) / (len(list1) + len(list2) - sum(list(counter.values())))


def check_mutuality(house1, house2):
    dict_data = dict()

    # Rooms Overlapping Cal
    dict_data['rooms'] = calculate_iou(house1['entire_rooms'], house2['entire_rooms'])

    # Furniture Overlapping Cal
    furniture1 = find_objects(house1)
    furniture2 = find_objects(house2)
    dict_data['furniture'] = calculate_iou(furniture1, furniture2)

    return dict_data


def process_file(list_houses):
    counter = 0
    relations = dict()
    for i, h1 in tqdm(enumerate(list_houses), total=len(list_houses)):
        for j, h2 in enumerate(list_houses[i + 1:]):
            relations[(i, i + j + 1)] = check_mutuality(h1, h2)
            counter += 1

    print('counter of processed file', counter)
    return relations


if __name__ == '__main__':
    houses_file = '../apartments_data/apartments_data.pkl'
    file = open(houses_file, 'rb')
    houses = pickle.load(file)
    start_time = time.time()
    relations = process_file(houses)
    with open('../scenes_relevances/relevance_structural.pkl', 'wb') as pickle_file:
        pickle.dump(relations, pickle_file)
    print('Elapsed time: ', time.time() - start_time)
