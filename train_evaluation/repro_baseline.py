from DNNs import GRUNet, OneDimensionalCNN
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from Data_utils import DescriptionScene
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import train_utility
import time
import numpy as np
import random


def collate_fn(data):
    # desc
    tmp_description = [x[0] for x in data]
    tmp = pad_sequence(tmp_description, batch_first=True)
    descs_ = pack_padded_sequence(tmp,
                                  torch.tensor([len(x) for x in tmp_description]),
                                  batch_first=True,
                                  enforce_sorted=False)
    tmp_scenes = [x[1] for x in data]
    list_length = [len(x[1]) for x in data]
    padded_scenes = pad_sequence(tmp_scenes, batch_first=True)
    padded_scenes = torch.transpose(padded_scenes, 1, 2)
    return descs_, padded_scenes, list_length


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


def start_train():

    set_seed(chosen_seed % 2**32)

    approach_name = str(time.time())
    output_feature_size = 256

    is_bidirectional = True
    model_desc_pov = GRUNet(hidden_size=output_feature_size, num_features=512, is_bidirectional=is_bidirectional)
    model_pov = OneDimensionalCNN(in_channels=512, out_channels=512, kernel_size=5,
                                  feature_size=output_feature_size)

    num_epochs = 50
    batch_size = 64

    # Loading Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_desc_pov.to(device)
    model_pov.to(device)

    train_indices, val_indices, test_indices = train_utility.retrieve_indices()
    descriptions_path, pov_path = train_utility.get_entire_data()
    dataset = DescriptionScene(data_description_path=descriptions_path, mem=True, data_scene_path=pov_path,
                               customized_margin=False, verbose=False)
    train_subset = Subset(dataset, train_indices.tolist())
    val_subset = Subset(dataset, val_indices.tolist())
    test_subset = Subset(dataset, test_indices.tolist())

    train_loader = DataLoader(train_subset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4, shuffle=True, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_subset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4, shuffle=False, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_subset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4, shuffle=False, worker_init_fn=seed_worker, generator=g)

    params = list(model_desc_pov.parameters()) + list(model_pov.parameters())
    optimizer = torch.optim.Adam(params, lr=0.008)

    train_losses = []
    val_losses = []

    scheduler = StepLR(optimizer, step_size=27, gamma=0.75)

    r10_hist = []
    best_r10 = 0

    for _ in range(num_epochs):
        total_loss_train = 0
        total_loss_val = 0
        num_batches_train = 0
        num_batches_val = 0

        output_description_val = torch.empty(len(val_indices), output_feature_size)
        output_scene_val = torch.empty(len(val_indices), output_feature_size)

        for i, (data_description, data_scene, length) in enumerate(train_loader):
            data_scene = data_scene.to(device)
            data_description = data_description.to(device)

            optimizer.zero_grad()

            output_descriptor = model_desc_pov(data_description)
            output_scene = model_pov(data_scene, length)

            multiplication = train_utility.cosine_sim(output_descriptor, output_scene)

            loss = train_utility.contrastive_loss(multiplication)

            loss.backward()

            optimizer.step()

            total_loss_train += loss.item()
            num_batches_train += 1

        scheduler.step()
        epoch_loss_train = total_loss_train / num_batches_train

        model_desc_pov.eval()
        model_pov.eval()

        with torch.no_grad():
            for j, (data_description, data_scene, length) in enumerate(val_loader):
                data_description = data_description.to(device)
                data_scene = data_scene.to(device)
                output_descriptor = model_desc_pov(data_description)
                output_scene = model_pov(data_scene, length)

                initial_index = j * batch_size
                final_index = (j + 1) * batch_size
                if final_index > len(val_indices):
                    final_index = len(val_indices)
                output_description_val[initial_index:final_index, :] = output_descriptor
                output_scene_val[initial_index:final_index, :] = output_scene

                multiplication = train_utility.cosine_sim(output_descriptor, output_scene)

                loss = train_utility.contrastive_loss(multiplication)

                total_loss_val += loss.item()

                num_batches_val += 1

            epoch_loss_val = total_loss_val / num_batches_val

        r1, r5, r10, _, _, _, _, _, _, _ = train_utility.evaluate(output_description=output_description_val,
                                                                  output_scene=output_scene_val,
                                                                  section="val",
                                                                  is_print=False)

        model_desc_pov.train()
        model_pov.train()

        r10_hist.append(r10)
        if r10 > best_r10:
            best_r10 = r10
            train_utility.save_best_model(approach_name, model_pov.state_dict(), model_desc_pov.state_dict())

        train_losses.append(epoch_loss_train)
        val_losses.append(epoch_loss_val)

    bm_pov, bm_desc_pov = train_utility.load_best_model(approach_name)
    model_pov.load_state_dict(bm_pov)
    model_desc_pov.load_state_dict(bm_desc_pov)
    model_pov.eval()
    model_desc_pov.eval()
    output_description_test = torch.empty(len(test_indices), output_feature_size)
    output_scene_test = torch.empty(len(test_indices), output_feature_size)
    # Evaluate test set
    with torch.no_grad():
        for j, (data_description, data_scene, length) in enumerate(test_loader):
            data_description = data_description.to(device)
            data_scene = data_scene.to(device)
            output_descriptor = model_desc_pov(data_description)
            output_scene = model_pov(data_scene, length)

            initial_index = j * batch_size
            final_index = (j + 1) * batch_size
            if final_index > len(test_indices):
                final_index = len(test_indices)
            output_description_test[initial_index:final_index, :] = output_descriptor
            output_scene_test[initial_index:final_index, :] = output_scene
    return train_utility.evaluate(
        output_description=output_description_test,
        output_scene=output_scene_test,
        section="test",
        is_print=False)


chosen_seed = 5965528221795689748
g = torch.Generator()
g.manual_seed(chosen_seed)
l_ds_r1, l_ds_r5, l_ds_r10, l_sd_r1, l_sd_r5, l_sd_r10, l_ds_medr, l_sd_medr = ([] for _ in range(8))
for i in range(3):
    ds_r1, ds_r5, ds_r10, sd_r1, sd_r5, sd_r10, _, _, ds_medr, sd_medr = start_train()
    for lst, value in zip(
            (l_ds_r1, l_ds_r5, l_ds_r10, l_sd_r1, l_sd_r5, l_sd_r10, l_ds_medr, l_sd_medr),
            (ds_r1, ds_r5, ds_r10, sd_r1, sd_r5, sd_r10, ds_medr, sd_medr),
    ):
        lst.append(value)

print(f"{sum(l_ds_r1) / len(l_ds_r1)},{sum(l_ds_r5) / len(l_ds_r5)},{sum(l_ds_r10) / len(l_ds_r10)},"
      f"{sum(l_sd_r1) / len(l_sd_r1)},{sum(l_sd_r5) / len(l_sd_r5)},{sum(l_sd_r10) / len(l_sd_r10)},"
      f"{sum(l_ds_medr) / len(l_ds_medr)},{sum(l_sd_medr) / len(l_sd_medr)}")
