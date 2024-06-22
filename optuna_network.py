import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from utils.loss_function import SaliencyLoss
from utils.data_process_uni import TrainDataset, ValDataset

from net.models.SUM import salu_mamba
from net.configs.config_setting import setting_config

import optuna


train_datasets_info = [
    {"id_train": 'datasets/salicon_256/train_ids.csv', "stimuli_dir": 'datasets/salicon_256/stimuli/train/', "saliency_dir": 'datasets/salicon_256/saliency/train/', "fixation_dir": 'datasets/salicon_256/fixations/train_edit/', "label": 0},
    {"id_train": 'datasets/OSIE_256/train_id.csv', "stimuli_dir": 'datasets/OSIE_256/train/train_stimuli/', "saliency_dir": 'datasets/OSIE_256/train/train_saliency/', "fixation_dir": 'datasets/OSIE_256/train/train_fixation/', "label": 1},
    {"id_train": 'datasets/CAT2000_256/train_id.csv', "stimuli_dir": 'datasets/CAT2000_256/train/train_stimuli/', "saliency_dir": 'datasets/CAT2000_256/train/train_saliency/', "fixation_dir": 'datasets/CAT2000_256/train/train_fixation/', "label": 1},
    {"id_train": 'datasets/MIT1003_256/train_id.csv', "stimuli_dir": 'datasets/MIT1003_256/train/train_stimuli/', "saliency_dir": 'datasets/MIT1003_256/train/train_saliency/', "fixation_dir": 'datasets/MIT1003_256/train/train_fixation/', "label": 1},
    {"id_train": 'datasets/SalEC/train_ids.csv', "stimuli_dir": 'datasets/SalEC/train/train_stimuli/', "saliency_dir": 'datasets/SalEC/train/train_saliency/', "fixation_dir": 'datasets/SalEC/train/train_fixation/', "label": 2},
    {"id_train": 'datasets/fiwi_256/train_id.csv', "stimuli_dir": 'datasets/fiwi_256/fiwi_train/stimuli/', "saliency_dir": 'datasets/fiwi_256/fiwi_train/saliency/', "fixation_dir": 'datasets/fiwi_256/fiwi_train/fixations/', "label": 3},
    {"id_train": 'datasets/datasets_UI_256/train_id.csv', "stimuli_dir": 'datasets/datasets_UI_256/train/train_images/', "saliency_dir": 'datasets/datasets_UI_256/train/train_saliency/', "fixation_dir": 'datasets/datasets_UI_256/train/train_fixation/', "label": 4}
]

val_datasets_info = [
    {"id_val": 'datasets/salicon_256/val_ids.csv', "stimuli_dir": 'datasets/salicon_256/stimuli/val/', "saliency_dir": 'datasets/salicon_256/saliency/val/', "fixation_dir": 'datasets/salicon_256/fixations/val_edit/', "label": 0},
    {"id_val": 'datasets/OSIE_256/val_id.csv', "stimuli_dir": 'datasets/OSIE_256/val/val_stimuli/', "saliency_dir": 'datasets/OSIE_256/val/val_saliency/', "fixation_dir": 'datasets/OSIE_256/val/val_fixation/', "label": 1},
    {"id_val": 'datasets/CAT2000_256/val_id.csv', "stimuli_dir": 'datasets/CAT2000_256/val/val_stimuli/', "saliency_dir": 'datasets/CAT2000_256/val/val_saliency/', "fixation_dir": 'datasets/CAT2000_256/val/val_fixation/', "label": 1},
    {"id_val": 'datasets/MIT1003_256/val_id.csv', "stimuli_dir": 'datasets/MIT1003_256/val/val_stimuli/', "saliency_dir": 'datasets/MIT1003_256/val/val_saliency/', "fixation_dir": 'datasets/MIT1003_256/val/val_fixation/', "label": 1},
    {"id_val": 'datasets/SalEC/val_ids.csv', "stimuli_dir": 'datasets/SalEC/val/val_stimuli/', "saliency_dir": 'datasets/SalEC/val/val_saliency/', "fixation_dir": 'datasets/SalEC/val/val_fixation/', "label": 2},
    {"id_val": 'datasets/fiwi_256/val_id.csv', "stimuli_dir": 'datasets/fiwi_256/fiwi_val/stimuli/', "saliency_dir": 'datasets/fiwi_256/fiwi_val/saliency/', "fixation_dir": 'datasets/fiwi_256/fiwi_val/fixations/', "label": 3},
    {"id_val": 'datasets/datasets_UI_256/val_id.csv', "stimuli_dir": 'datasets/datasets_UI_256/val/val_images/', "saliency_dir": 'datasets/datasets_UI_256/val/val_saliency/', "fixation_dir": 'datasets/datasets_UI_256/val/val_fixation/', "label": 4}
]


train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class SubsetDataset(Dataset):
    def __init__(self, base_dataset, subset_ratio=0.20):
        self.base_dataset = base_dataset
        total_count = len(self.base_dataset)
        subset_count = int(total_count * subset_ratio)
        self.indices = torch.randperm(total_count)[:subset_count].tolist()

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


# Assuming TrainDataset and ValDataset classes are defined as before

# Load training datasets with subset
train_datasets = [
    SubsetDataset(
        TrainDataset(datasets_info=[info], transform=train_transform),
        subset_ratio=0.10
    ) for info in train_datasets_info
]

train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=16, shuffle=True, num_workers=0)

# Load validation datasets with subset
val_datasets = [
    SubsetDataset(
        ValDataset(
            ids_path=info["id_val"],
            stimuli_dir=info["stimuli_dir"],
            saliency_dir=info["saliency_dir"],
            fixation_dir=info["fixation_dir"],
            label=info["label"],
            transform=val_transform
        ),
        subset_ratio=0.10
    ) for info in val_datasets_info
]

val_loaders = {
    f"val_loader_{idx}": DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    for idx, dataset in enumerate(val_datasets)
}


def mean_std(test_list):
    mean = sum(test_list) / len(test_list)
    variance = sum([((x - mean) ** 2) for x in test_list]) / len(test_list)
    res = variance ** 0.5
    return mean, res


def objective(trial):
    log_file_path = "optuna_logs.txt"
    # Suggest values for the hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 9e-3)
    step_size = trial.suggest_int('step_size', 1, 6)
    gamma = trial.suggest_uniform('gamma', 0.05, 0.3)
    coef_kl = trial.suggest_float('coef_kl', 1, 20)
    coef_cc = trial.suggest_float('coef_cc', -5, 0)
    coef_sim = trial.suggest_float('coef_sim', -5, 0)
    coef_nss = trial.suggest_float('coef_nss', -5, 0)
    coef_mse = trial.suggest_float('coef_mse', 0, 10)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = setting_config

    model_cfg = config.model_config

    if config.network == 'sum':
        model = salu_mamba(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
        )
        model.load_from()
        model.cuda(0)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    loss_fn = SaliencyLoss()
    mse_loss = nn.MSELoss()

    # Initialize best loss for this trial
    best_loss = float('inf')

    for epoch in range(10):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc="Training"):
            stimuli, smap, fmap, condition = batch['image'].to(device), batch['saliency'].to(device), batch[
                'fixation'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(stimuli, condition)

            kl = loss_fn(outputs, smap, loss_type='kldiv')
            cc = loss_fn(outputs, smap, loss_type='cc')
            sim = loss_fn(outputs, smap, loss_type='sim')
            nss = loss_fn(outputs, fmap, loss_type='nss')

            loss1 = coef_cc * cc + coef_kl * kl + coef_sim * sim + coef_nss * nss
            loss2 = mse_loss(outputs, smap)
            loss = loss1 + coef_mse * loss2

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * stimuli.size(0)

        scheduler.step()

        # Validation phase
        model.eval()
        val_kl = 0.0
        val_cc = 0.0
        val_sim = 0.0
        val_nss = 0.0

        with torch.no_grad():
            for name, loader in val_loaders.items():
                for batch in tqdm(loader, desc=f"Validating {name}"):
                    stimuli, smap, fmap, condition = batch['image'].to(device), batch['saliency'].to(device), batch[
                        'fixation'].to(device), batch['label'].to(device)
                    outputs = model(stimuli, condition)

                    kl = loss_fn(outputs, smap, loss_type='kldiv')
                    cc = loss_fn(outputs, smap, loss_type='cc')
                    sim = loss_fn(outputs, smap, loss_type='sim')
                    nss = loss_fn(outputs, fmap, loss_type='nss')

                    val_cc += cc.item() * stimuli.size(0)
                    val_sim += sim.item() * stimuli.size(0)
                    val_nss += nss.item() * stimuli.size(0)
                    val_kl += kl.item() * stimuli.size(0)

            # Compute the average validation loss
            val_kl /= len(loader.dataset)
            val_cc /= len(loader.dataset)
            val_sim /= len(loader.dataset)
            val_nss /= len(loader.dataset)

            combined_loss = val_kl - (val_cc + val_sim + val_nss)

            # Update best loss if the current validation KL is lower
            if combined_loss < best_loss:
                best_loss = combined_loss
    # After each trial, log the results
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Trial {trial.number}, Loss: {best_loss}\n")
        log_file.write(
            f"  Params: lr: {lr}, step_size: {step_size}, gamma: {gamma}, coef_kl: {coef_kl}, coef_cc: {coef_cc}, coef_sim: {coef_sim}, coef_nss: {coef_nss}, coef_mse: {coef_mse}\n")

    return best_loss


# Create a study with specified storage, direction, and name
study = optuna.create_study(
    storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
    study_name="quadratic-simple",
    direction='minimize'  # Specify the optimization direction here.
)

# Optimize the study
study.optimize(objective, n_trials=40)

# Logging the best trial information
with open("optimization_log_final.txt", 'a') as log_file:
    log_file.write("Best trial:\n")
    log_file.write(f"  Value: {study.best_trial.value}\n")
    log_file.write("  Params: \n")
    for key, value in study.best_trial.params.items():
        log_file.write(f"    {key}: {value}\n")

print("Best trial:")
print(f"  Value: {study.best_trial.value}")
print("  Params: ")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")
