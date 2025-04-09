import pickle
from DNNs import GRUNet, OneDimensionalCNN
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Subset, DataLoader
from torch.optim.lr_scheduler import StepLR
import train_utility
from Data_utils import DescriptionScene
import argparse
import time
import numpy as np
import random


def collate_fn(data):
    # desc
    tmp_description_povs = [x[0] for x in data]
    tmp = pad_sequence(tmp_description_povs, batch_first=True)
    descs_pov = pack_padded_sequence(tmp,
                                     torch.tensor([len(x) for x in tmp_description_povs]),
                                     batch_first=True,
                                     enforce_sorted=False)

    tmp_pov = [x[1] for x in data]
    padded_pov = pad_sequence(tmp_pov, batch_first=True)
    padded_pov = torch.transpose(padded_pov, 1, 2)

    indexes = [x[2] for x in data]
    return descs_pov, padded_pov, indexes


def get_similarity_function(path):
    return pickle.load(open(path, 'rb'))


def set_seed(seed_num):
    np.random.seed(chosen_seed % 2**32)
    random.seed(chosen_seed % 2**32)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def start_train(args):

    set_seed(chosen_seed % 2**32)
    
    approach_name = str(time.time())
    output_feature_size = args.output_feature_size
    similarity_metric = get_similarity_function(
        f'../scenes_relevances/relevance_slm_normalized.pkl')
    is_txt_similarity = True if args.txt_sim else False
    relevance_info = similarity_metric if is_txt_similarity else pickle.load(open(
        '../scenes_relevances/relevance_structural.pkl', 'rb'))

    margins = dict()
    margins['margin_low'] = args.margin_l
    margins['margin_mid'] = args.margin_m
    margins['margin_high'] = args.margin_h
    thresh = (args.thresh_l, args.thresh_u)

    is_bidirectional = args.is_bidirectional
    model_desc_pov = GRUNet(hidden_size=output_feature_size, num_features=512, is_bidirectional=is_bidirectional)
    model_pov = OneDimensionalCNN(in_channels=512, out_channels=512, kernel_size=5,
                                  feature_size=output_feature_size)
    cont_loss = train_utility.LossContrastive(name=approach_name, patience=25, delta=0.0001, verbose=False)
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print('device = ', device)
    model_desc_pov.to(device=device)
    model_pov.to(device=device)

    #     data section
    train_indices, val_indices, test_indices = train_utility.retrieve_indices()
    descriptions_path, pov_path = train_utility.get_entire_data()
    dataset = DescriptionScene(data_description_path=descriptions_path, mem=True, data_scene_path=pov_path,
                               customized_margin=True, verbose=False)
    train_subset = Subset(dataset, train_indices.tolist())
    val_subset = Subset(dataset, val_indices.tolist())
    test_subset = Subset(dataset, test_indices.tolist())

    train_loader = DataLoader(train_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4,worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4,worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4,worker_init_fn=seed_worker, generator=g)

    '''Train Procedure'''
    params = list(model_desc_pov.parameters()) + list(model_pov.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    step_size = args.step_size
    gamma = args.gamma
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    best_r10 = 0
    for _ in range(num_epochs):

        if not cont_loss.is_val_improving():
            break

        total_loss_train = 0
        total_loss_val = 0
        num_batches_train = 0
        num_batches_val = 0

        output_description_val = torch.empty(len(val_indices), output_feature_size)
        output_pov_val = torch.empty(len(val_indices), output_feature_size)

        for i, (data_desc_pov, data_pov, indexes) in enumerate(train_loader):
            data_desc_pov = data_desc_pov.to(device)
            data_pov = data_pov.to(device)

            optimizer.zero_grad()

            output_desc_pov = model_desc_pov(data_desc_pov)
            output_pov = model_pov(data_pov)

            multiplication_dp = train_utility.cosine_sim(output_desc_pov, output_pov)

            margin_tensor = train_utility.get_margin_tensor(indexes, relevance_info, margins, thresholds=thresh,
                                                            sent_similarity=is_txt_similarity)
            loss_contrastive = cont_loss.calculate_loss(multiplication_dp, margin_tensor=margin_tensor)

            loss_contrastive.backward()

            optimizer.step()

            total_loss_train += loss_contrastive.item()
            num_batches_train += 1

        scheduler.step()
        # print(scheduler.get_last_lr())
        epoch_loss_train = total_loss_train / num_batches_train

        model_desc_pov.eval()
        model_pov.eval()
        # Validation Procedure
        with torch.no_grad():
            for j, (data_desc_pov, data_pov, indexes) in enumerate(val_loader):

                data_desc_pov = data_desc_pov.to(device)
                data_pov = data_pov.to(device)

                output_desc_pov = model_desc_pov(data_desc_pov)
                output_pov = model_pov(data_pov)

                initial_index = j * batch_size
                final_index = (j + 1) * batch_size
                if final_index > len(val_indices):
                    final_index = len(val_indices)

                output_description_val[initial_index:final_index, :] = output_desc_pov
                output_pov_val[initial_index:final_index, :] = output_pov

                multiplication_dp = train_utility.cosine_sim(output_desc_pov, output_pov)

                margin_tensor = train_utility.get_margin_tensor(indexes=indexes, relevance_info=relevance_info,
                                                                margins=margins, thresholds=thresh,
                                                                sent_similarity=is_txt_similarity)
                loss_contrastive = cont_loss.calculate_loss(multiplication_dp, margin_tensor=margin_tensor)

                total_loss_val += loss_contrastive.item()
                num_batches_val += 1

            epoch_loss_val = total_loss_val / num_batches_val

            cont_loss.on_epoch_end(epoch_loss_train, train=True)
            cont_loss.on_epoch_end(epoch_loss_val, train=False)

        r1, r5, r10, _, _, _, _, _, _, _ = train_utility.evaluate(output_description=output_description_val,
                                                                  output_scene=output_pov_val, section='val',
                                                                  is_print=False)

        model_desc_pov.train()
        model_pov.train()

        if r10 > best_r10:
            best_r10 = r10
            train_utility.save_best_model(approach_name, model_pov.state_dict(), model_desc_pov.state_dict())
    bm_pov, bm_desc_pov = train_utility.load_best_model(approach_name)
    model_pov.load_state_dict(bm_pov)
    model_desc_pov.load_state_dict(bm_desc_pov)
    model_pov.eval()
    model_desc_pov.eval()
    output_description_test = torch.empty(len(test_indices), output_feature_size)
    output_pov_test = torch.empty(len(test_indices), output_feature_size)
    # Evaluate test set
    with torch.no_grad():
        for j, (data_desc_pov, data_pov, indexes) in enumerate(test_loader):

            data_desc_pov = data_desc_pov.to(device)
            data_pov = data_pov.to(device)

            output_desc_pov = model_desc_pov(data_desc_pov)
            output_pov = model_pov(data_pov)

            initial_index = j * batch_size
            final_index = (j + 1) * batch_size
            if final_index > len(test_indices):
                final_index = len(test_indices)
            output_description_test[initial_index:final_index, :] = output_desc_pov
            output_pov_test[initial_index:final_index, :] = output_pov
    return train_utility.evaluate(output_description=output_description_test,
                                  output_scene=output_pov_test,
                                  section="test", is_print=False)


parser = argparse.ArgumentParser(description='Train Specs')
parser.add_argument('-txt_sim', action='store_true', help='set if using the sentence similarity or not set if using rooms structure similarity')
parser.add_argument("--output_feature_size", type=int, default=256, required=False,
                    help='The size of the output feature')
parser.add_argument("--is_bidirectional", type=bool, default=True, required=False,
                    help='Use the Bidirectional GRU or not')
parser.add_argument("--num_epochs", type=int, default=50, required=False, help='number of epochs')
parser.add_argument("--batch_size", type=int, default=64, required=False, help='batch size')
parser.add_argument("--lr", type=float, default=.008, required=False, help='learning rate')
parser.add_argument("--step_size", type=int, default=27, required=False,
                    help='Step size for the decay of the learning rate')
parser.add_argument("--gamma", type=float, default=0.75, required=False,
                    help='learning rate decay factor with which the learning rate will be reduced')
parser.add_argument("--margin_l", type=float, default=0.25, required=False,
                    help='lower margin')
parser.add_argument("--margin_m", type=float, default=0.30, required=False,
                    help='lower middle')
parser.add_argument("--margin_h", type=float, default=0.35, required=False,
                    help='lower high')
parser.add_argument("--thresh_l", type=float, default=0.25, required=False,
                    help='lower threshold')
parser.add_argument("--thresh_u", type=float, default=0.75, required=False,
                    help='upper threshold')
parser.add_argument("--seed", type=int, default=5965528221795689748, required=False,
                    help='upper threshold')
args = parser.parse_args()


chosen_seed = args.seed
g = torch.Generator()
g.manual_seed(chosen_seed)

l_ds_r1, l_ds_r5, l_ds_r10, l_sd_r1, l_sd_r5, l_sd_r10, l_ds_medr, l_sd_medr = ([] for _ in range(8))
for i in range(3):
    ds_r1, ds_r5, ds_r10, sd_r1, sd_r5, sd_r10, _, _, ds_medr, sd_medr = start_train(args)
    for lst, value in zip(
            (l_ds_r1, l_ds_r5, l_ds_r10, l_sd_r1, l_sd_r5, l_sd_r10, l_ds_medr, l_sd_medr),
            (ds_r1, ds_r5, ds_r10, sd_r1, sd_r5, sd_r10, ds_medr, sd_medr),
    ):
        lst.append(value)

print(f"{sum(l_ds_r1) / len(l_ds_r1)},{sum(l_ds_r5) / len(l_ds_r5)},{sum(l_ds_r10) / len(l_ds_r10)},"
      f"{sum(l_sd_r1) / len(l_sd_r1)},{sum(l_sd_r5) / len(l_sd_r5)},{sum(l_sd_r10) / len(l_sd_r10)},"
      f"{sum(l_ds_medr) / len(l_ds_medr)},{sum(l_sd_medr) / len(l_sd_medr)}")
