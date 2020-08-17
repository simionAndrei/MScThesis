import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import os
import random
import pickle
import numpy as np
from tqdm import tqdm

from temporal_transform import TemporalRandomCrop, TemporalCenterCrop
from resnet3d import generate_model
from ucf101 import get_ucf_dataset
from hmdb51 import get_hmdb_dataset
from logger import Logger


train_acc = []
valid_acc = []


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(optimizer, epoch, decay_epochs, lr_decay_rate):
    
    prev_lr = optimizer.param_groups[0]['lr']
    if epoch in decay_epochs:
        lr = prev_lr * lr_decay_rate
        print("Adjusting learning rate to {}".format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    return optimizer


def test(model, test_loader, device):

    model = model.eval()
    eval_acc = []
    running_loss = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            acc = accuracy(outputs, labels)
            eval_acc.append(acc[-1].item())

    eval_acc = np.mean(eval_acc)
    logger.log("Test accuracy {:.2f}".format(eval_acc), show_time = True)
    return eval_acc



def run(logger):

    torch.manual_seed(1)
    random.seed(1)

    frame_size = logger.config_dict['frame_resize']
    spatial_transform = transforms.Compose([transforms.Resize((frame_size, frame_size)), transforms.ToTensor()])
    logger.log("Frames resized to {}x{}".format(frame_size, frame_size))

    sampling_method_str = logger.config_dict['sampling_method']
    if "rand" in sampling_method_str:
        crop_size = int(sampling_method_str.replace("rand", ""))
        sampling_method = TemporalRandomCrop(size = crop_size)
        logger.log("Sampling strategy selecting {} consecutives frames from a random index".format(
            crop_size))

    video_path = os.path.join(logger.config_dict['data_folder'], 
                          logger.config_dict['video_folder'], logger.config_dict['frame_folder'])
    annotation_path = logger.get_data_file(logger.config_dict['annotation_file'])
    batch_size  = logger.config_dict['batch_size']
    num_workers = logger.config_dict['num_workers']
    dataset_type = logger.config_dict['dataset_type']

    if dataset_type == "ucf101":
        train_dataset = get_ucf_dataset(video_path, annotation_path, "training", sampling_method, 
                                        spatial_transform, temporal_transform = "None",
                                        stack_clip = True, is_simclr_transform = False, 
                                        apply_same_per_clip = True)
        test_dataset = get_ucf_dataset(video_path, annotation_path, "validation", sampling_method, 
                                       spatial_transform, temporal_transform = "None",
                                       stack_clip = True, is_simclr_transform = False, 
                                       apply_same_per_clip = True)
    elif dataset_type == "hmdb51":
        train_dataset = get_hmdb_dataset(video_path, annotation_path, "training", sampling_method, 
                                         spatial_transform, temporal_transform = "None",
                                         stack_clip = True, is_simclr_transform = False, 
                                         apply_same_per_clip = True)
        test_dataset = get_hmdb_dataset(video_path, annotation_path, "validation", sampling_method, 
                                        spatial_transform, temporal_transform = "None",
                                        stack_clip = True, is_simclr_transform = False, 
                                        apply_same_per_clip = True)

    train_loader  = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True, 
                               num_workers = num_workers)
    logger.log("{} Train data loaded with {} clips".format(dataset_type.upper(), len(train_dataset)))

    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, drop_last = False,
                             num_workers = num_workers)
    logger.log("{} Test data loaded with {} clips".format(dataset_type.upper(), len(test_dataset)))

    base_model  = logger.config_dict['base_convnet']
    num_classes = logger.config_dict['num_classes']
    model = generate_model(model_depth = 18, n_classes = num_classes)
    logger.log("Model {} 3D simCLR loaded {}".format(base_model, model))

    checkpoint_epoch = logger.config_dict['model_checkpoint_epoch']
    checkpoint_file  = logger.config_dict['model_checkpoint_file']
    training_batch_size = int(checkpoint_file.split("_")[1].replace("b", ""))
    logger.log("Loading simCLR model weights at epoch {} with {} training batchsize from {}".format(
        checkpoint_epoch, training_batch_size, checkpoint_file))
    checkpoint = torch.load(logger.get_model_file(checkpoint_file))

    state_dict = checkpoint['model']

    for k in list(state_dict.keys()):
        if k.startswith('module') and not k.startswith('module.fc'):
            state_dict[k[len("module."):]] = state_dict[k]

        del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", 
                                     "fc3.weight", "fc3.bias"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log("Training on: {}".format("GPU" if device == "cuda" else CPU))

    if torch.cuda.device_count() > 1:
        logger.log("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    params = None
    print('=> finetune backbone with smaller lr')
    params = []
    for name, param in model.module.named_parameters():
        if 'fc' not in name:
            print("{} with small lr".format(name))
            params.append({'params': param, 'lr': 1e-3/10})
        else:
            print("{} with normal lr".format(name))
            params.append({'params': param})

    criterion = torch.nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(params, lr = 1e-3, weight_decay = 1e-3)


    num_epochs = logger.config_dict['num_epochs']
    for epoch in  tqdm(range(num_epochs)):
        
        model = model.train()
        logger.log("Starting epoch#{}...".format(epoch))
        running_acc  = []
        running_loss = 0.0 
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
             
            
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            acc = accuracy(outputs, labels)
            running_acc.append(acc[-1].item())
            running_loss += loss.item()
        
        optimizer = adjust_learning_rate(optimizer, epoch,
                                         decay_epochs = [50, 100, 130], 
                                         lr_decay_rate = 0.5)
        
        if epoch % 5 == 0:
            test_acc = test(model, test_loader, device)
            logger.log("Saving epoch {} with accuracy {:.2f}".format(epoch, test_acc))
            checkpoint = {'model': model.state_dict()}
            model_filename = "smallr_3fc_finet_{}_{}_{}f_{}_3d".format(base_model, dataset_type, frame_size, 
                sampling_method_str)
            logger.save_model(checkpoint, model_filename, epoch, test_acc)
            train_acc.append(np.mean(running_acc))
            valid_acc.append(test_acc)


        logger.log("Epoch#{} with training accuracy {:.2f} and loss {:.3f}".format(
            epoch, np.mean(running_acc), running_loss), show_time = True)

    logger.log("Training acc {}".format(train_acc))
    logger.log("Validation acc {}".format(valid_acc))

    acc_filename = "smallr_fc_finet_" + dataset_type + "_trainacc.pkl"
    with open(acc_filename, "wb") as fp:
        pickle.dump(train_acc, fp)

    acc_filename = "smallr_fc_finet_" + dataset_type + "_validacc.pkl"
    with open(acc_filename, "wb") as fp:
        pickle.dump(valid_acc, fp)



if __name__=='__main__':
    logger = Logger(show = True, html_output = True, config_file = "config.txt")
    run(logger) 