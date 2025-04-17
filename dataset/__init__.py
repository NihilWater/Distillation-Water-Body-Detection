import torch

from dataset.floods11_s1 import load_flood_train_data_s1, FloodsS1Dataset, processAndAugmentS1, \
    load_flood_valid_data_s1, processTestImS1
from dataset.floods11_s1_jrc import load_flood11_s1_jrc_data, FloodsS1JRCDataset, processAndAugmentS1JRC, \
    processTestImS1JRC
from dataset.floods11_s1_with_rate import load_flood_data_s1_with_rate, FloodsS1WithRateDataset, \
    processAndAugmentS1_with_rate, processTestImS1_with_rate
from dataset.floods11_s1s2 import load_flood11_s1s2_data, FloodsS1S2Dataset, processAndAugmentS1S2, processTestImS1S2
from dataset.floods11_s1s2_rate import load_flood11_s1s2_with_rate_data, FloodsS1S2WithRateDataset, \
    processAndAugmentS1S2_with_rate, processTestImS1S2_with_rate
from dataset.floods11_s2 import load_flood_train_data_s2, FloodsS2Dataset, processAndAugmentS2, \
    load_flood_valid_data_s2, processTestImS2
from dataset.test_dataset import TestDataSet


def get_data_load(dateset_name, mode="train"):
    if mode == "train":
        shuffle = True
        valid_bs = 4
    else:
        shuffle = False
        valid_bs = 1

    if dateset_name == 'floods_s1':
        train_data = load_flood_train_data_s1('/home/amax/SSD1/zjzRoot/project/data/sen1floods11')
        train_dataset = FloodsS1Dataset(train_data, processAndAugmentS1)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=shuffle, sampler=None,
                                                   batch_sampler=None, num_workers=0, collate_fn=None,
                                                   pin_memory=True, drop_last=False, timeout=0,
                                                   worker_init_fn=None)

        valid_data = load_flood_valid_data_s1('/home/amax/SSD1/zjzRoot/project/data/sen1floods11')
        valid_dataset = FloodsS1Dataset(valid_data, processTestImS1)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=valid_bs, shuffle=shuffle, sampler=None,
                                                   batch_sampler=None, num_workers=0, collate_fn=lambda x: (
                torch.cat([a[0] for a in x], 0), torch.cat([a[1] for a in x], 0)),
                                                   pin_memory=True, drop_last=False, timeout=0,
                                                   worker_init_fn=None)
        return train_loader, valid_loader

    elif dateset_name == 'floods_s2':
        train_data = load_flood_train_data_s2('/home/amax/SSD1/zjzRoot/project/data/sen1floods11')
        train_dataset = FloodsS2Dataset(train_data, processAndAugmentS2)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=shuffle, sampler=None,
                                                   batch_sampler=None, num_workers=0, collate_fn=None,
                                                   pin_memory=True, drop_last=False, timeout=0,
                                                   worker_init_fn=None)

        valid_data = load_flood_valid_data_s2('/home/amax/SSD1/zjzRoot/project/data/sen1floods11')
        valid_dataset = FloodsS2Dataset(valid_data, processTestImS2)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=valid_bs, shuffle=shuffle, sampler=None,
                                                   batch_sampler=None, num_workers=0, collate_fn=lambda x: (
                torch.cat([a[0] for a in x], 0), torch.cat([a[1] for a in x], 0)),
                                                   pin_memory=True, drop_last=False, timeout=0,
                                                   worker_init_fn=None)
        return train_loader, valid_loader

    elif dateset_name == 'floods_s1_with_rate':
        train_data = load_flood_data_s1_with_rate('/home/amax/SSD1/zjzRoot/project/data/sen1floods11', 'train')
        train_dataset = FloodsS1WithRateDataset(train_data, processAndAugmentS1_with_rate)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=shuffle, sampler=None,
                                                   batch_sampler=None, num_workers=0, collate_fn=None,
                                                   pin_memory=True, drop_last=False, timeout=0,
                                                   worker_init_fn=None)
        valid_data = load_flood_data_s1_with_rate('/home/amax/SSD1/zjzRoot/project/data/sen1floods11', 'val')
        valid_dataset = FloodsS1WithRateDataset(valid_data, processTestImS1_with_rate)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=valid_bs, shuffle=shuffle, sampler=None,
                                                   batch_sampler=None, num_workers=0, collate_fn=lambda x: (
                torch.cat([a[0] for a in x], 0), torch.cat([a[1] for a in x], 0)),
                                                   pin_memory=True, drop_last=False, timeout=0,
                                                   worker_init_fn=None)
        return train_loader, valid_loader

    elif dateset_name == 'floods_s1s2':
        train_data = load_flood11_s1s2_data('/home/amax/SSD1/zjzRoot/project/data/sen1floods11', 'train')
        train_dataset = FloodsS1S2Dataset(train_data, processAndAugmentS1S2)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=shuffle, sampler=None,
                                                   batch_sampler=None, num_workers=0, collate_fn=None,
                                                   pin_memory=True, drop_last=False, timeout=0,
                                                   worker_init_fn=None)
        valid_data = load_flood11_s1s2_data('/home/amax/SSD1/zjzRoot/project/data/sen1floods11', 'val')
        valid_dataset = FloodsS1S2Dataset(valid_data, processTestImS1S2)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=valid_bs, shuffle=shuffle, sampler=None,
                                                   batch_sampler=None, num_workers=0, collate_fn=lambda x: (
                torch.cat([a[0] for a in x], 0), torch.cat([a[1] for a in x], 0), torch.cat([a[2] for a in x], 0)),
                                                   pin_memory=True, drop_last=False, timeout=0,
                                                   worker_init_fn=None)
        return train_loader, valid_loader

    elif dateset_name == 'floods_s1s2_with_rate':
        train_data = load_flood11_s1s2_with_rate_data('/home/amax/SSD1/zjzRoot/project/data/sen1floods11', 'train')
        train_dataset = FloodsS1S2WithRateDataset(train_data, processAndAugmentS1S2_with_rate)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=shuffle, sampler=None,
                                                   batch_sampler=None, num_workers=0, collate_fn=None,
                                                   pin_memory=True, drop_last=False, timeout=0,
                                                   worker_init_fn=None)

        valid_data = load_flood11_s1s2_with_rate_data('/home/amax/SSD1/zjzRoot/project/data/sen1floods11', 'val')
        valid_dataset = FloodsS1S2WithRateDataset(valid_data, processTestImS1S2_with_rate)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=valid_bs, shuffle=shuffle, sampler=None,
                                                   batch_sampler=None, num_workers=0, collate_fn=lambda x: (
                torch.cat([a[0] for a in x], 0), torch.cat([a[1] for a in x], 0), torch.cat([a[2] for a in x], 0)),
                                                   pin_memory=True, drop_last=False, timeout=0,
                                                   worker_init_fn=None)
        return train_loader, valid_loader

    elif dateset_name == 'floods11_s1_jrc':

        train_data = load_flood11_s1_jrc_data('/home/amax/SSD1/zjzRoot/project/data/sen1floods11')
        train_dataset = FloodsS1JRCDataset(train_data, processAndAugmentS1JRC)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=shuffle, sampler=None,
                                                   batch_sampler=None, num_workers=0, collate_fn=None,
                                                   pin_memory=True, drop_last=False, timeout=0,
                                                   worker_init_fn=None)

        valid_data = load_flood11_s1_jrc_data('/home/amax/SSD1/zjzRoot/project/data/sen1floods11')
        valid_dataset = FloodsS1JRCDataset(valid_data, processTestImS1JRC)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=valid_bs, shuffle=shuffle, sampler=None,
                                                   batch_sampler=None, num_workers=0, collate_fn=lambda x: (
                torch.cat([a[0] for a in x], 0), torch.cat([a[1] for a in x], 0), torch.cat([a[2] for a in x], 0)),
                                                   pin_memory=True, drop_last=False, timeout=0,
                                                   worker_init_fn=None)
        return train_loader, valid_loader

    elif dateset_name == 'test':
        train_dataset = TestDataSet()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=shuffle, sampler=None,
                                                   batch_sampler=None, num_workers=0, collate_fn=None,
                                                   pin_memory=True, drop_last=False, timeout=0,
                                                   worker_init_fn=None)

        valid_dataset = TestDataSet()
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=valid_bs, shuffle=shuffle, sampler=None,
                                                   batch_sampler=None, num_workers=0, collate_fn=None,
                                                   pin_memory=True, drop_last=False, timeout=0,
                                                   worker_init_fn=None)
        return train_loader, valid_loader
