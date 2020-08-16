import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from data_transform import get_spatial_transform, get_temporal_transform, get_improved_spatial_transform

from temporal_transform import TemporalRandomCrop
from resnet3d import generate_model
from nt_xent import NT_Xent

from ucf101 import get_ucf_dataset
from hmdb51 import get_hmdb_dataset

from logger import Logger

from distutils import util
from tqdm import tqdm
import pickle
import time
import os

def adjust_learning_rate(optimizer, epoch, decay_epochs, lr_decay_rate):
    
    prev_lr = optimizer.param_groups[0]['lr']
    if epoch in decay_epochs:
        lr = prev_lr * lr_decay_rate
        print("Adjusting learning rate to {}".format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    return optimizer



train_dataset = None
train_loader = None

def run(logger):

  torch.manual_seed(0)

  frame_size = logger.config_dict['frame_resize']
  logger.log("Frames resized to {}x{}".format(frame_size, frame_size))

  strength = logger.config_dict['strength']
  logger.log("Color jitter of strength {} and with Gaussian blur".format(strength))

  apply_same_per_clip = bool(util.strtobool(logger.config_dict['same_per_clip']))
  logger.log("Random spatial data augmentation per {}".format("clip" if apply_same_per_clip else "frame"))

  temporal_transform_str = logger.config_dict['temporal_transform_type']
  step = logger.config_dict['temporal_transform_step']

  spatial_transform  = get_spatial_transform(strength = strength, crop_size = frame_size, 
    with_gauss_blur = True)

  temporal_transform = get_temporal_transform(temporal_transform_str)

  sampling_method_str = logger.config_dict['sampling_method']
  if "rand" in sampling_method_str:
    crop_size = int(sampling_method_str.replace("rand", ""))
    sampling_method = TemporalRandomCrop(size = crop_size)
    logger.log("Sampling strategy selecting {} consecutives frames from a random index".format(
      crop_size))

  # /data/VisionLab or data/
  video_path = os.path.join(logger.config_dict['data_folder'], 
    logger.config_dict['video_folder'], logger.config_dict['frame_folder'])
  annotation_path = logger.get_data_file(logger.config_dict['annotation_file'])
  batch_size  = logger.config_dict['batch_size']
  num_workers = logger.config_dict['num_workers']
  dataset_type = logger.config_dict['dataset_type']

  if temporal_transform is not None:
    logger.log("Using {} as temporal transform with step {}".format(temporal_transform_str, step))

  if dataset_type == "ucf101":
    train_dataset = get_ucf_dataset(video_path, annotation_path, "training", sampling_method, 
      spatial_transform, temporal_transform, temporal_step = step,
      stack_clip = True, is_simclr_transform = True, 
      apply_same_per_clip = apply_same_per_clip, add_random_per_clip = False) 
  elif dataset_type == "hmdb51":
    train_dataset = get_hmdb_dataset(video_path, annotation_path, "training", sampling_method, 
      spatial_transform, temporal_transform, temporal_step = step,
      stack_clip = True, is_simclr_transform = True, apply_same_per_clip = apply_same_per_clip)

  train_loader  = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True, 
    num_workers = num_workers)
  logger.log("{} Train data loaded with {} clips".format(dataset_type.upper(), len(train_dataset)))

  base_model = logger.config_dict['base_convnet']
  output_dim = logger.config_dict['simclr_out_dim']
  model = generate_model(model_depth = 18, add_projection_layers = True, projection_dim = output_dim)
  logger.log("Model {} 3D simCLR loaded {}".format(base_model, model))

  device = "cuda" if torch.cuda.is_available() else "cpu"
  logger.log("Training on: {}".format("GPU" if device == "cuda" else CPU))

  optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-6)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), 
    eta_min=0, last_epoch=-1)
  temp = logger.config_dict['temp']
  nt_xent   = NT_Xent(device, batch_size, temp, True)

  checkpoint_epoch = 0
  use_kinet = bool(util.strtobool(logger.config_dict['use_kinet']))
  if logger.config_dict['model_checkpoint_epoch'] != 0 and not use_kinet:
    checkpoint_epoch = logger.config_dict['model_checkpoint_epoch']
    checkpoint_file  = logger.config_dict['model_checkpoint_file']
    logger.log("Loading model from checkpoint at epoch {} from {}".format(checkpoint_epoch, checkpoint_file))

    checkpoint = torch.load(logger.get_model_file(checkpoint_file), map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model'])

    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-6)
    optimizer.load_state_dict(checkpoint['optimizer'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), 
      eta_min=0, last_epoch=-1)
    scheduler.load_state_dict(checkpoint['scheduler'])

    if device == "cuda":
      for state in optimizer.state.values():
        for k, v in state.items():
          if torch.is_tensor(v):
            state[k] = v.cuda()

  if use_kinet:
    checkpoint_epoch = logger.config_dict['model_checkpoint_epoch']
    checkpoint_file  = logger.config_dict['model_checkpoint_file']
    logger.log("Loading kinetics pretrained model weights at epoch {} from {}".format(
      checkpoint_epoch, checkpoint_file))
    checkpoint = torch.load(logger.get_model_file(checkpoint_file))
    msg = model.load_state_dict(checkpoint['state_dict'], strict = False)
    assert set(msg.missing_keys) == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
    checkpoint_epoch = 0

  if torch.cuda.device_count() > 1:
    logger.log("Using {} GPUs".format(torch.cuda.device_count()))
    model = nn.DataParallel(model)

  # dataset[i][0] clip frame list
  # dataset[i][1] clip target
  # dataset[i][0][j] j-frame of the clip
  # dataset[i][0][j][0] first  changed frame
  # dataset[i][0][j][1] second changed frame

  model = model.to(device)
  model = model.train()
  losses = []
  num_epochs = logger.config_dict['num_epochs']
  logger.log("Starting training for {} epochs with batch size {} ...".format(num_epochs, batch_size))
  start_time = time.time()
  for epoch in tqdm(range(num_epochs)):

    crt_epoch = epoch + checkpoint_epoch + 1
    logger.log("Starting epoch #{}".format(crt_epoch))

    running_loss = 0.0
    for i, (input_batch, _) in enumerate(train_loader):
      
      input_batch = input_batch.permute(1, 0, 2, 3, 4, 5)

      optimizer.zero_grad()

      input_batch_i, input_batch_j = input_batch

      input_batch_i = input_batch_i.to(device)
      input_batch_j = input_batch_j.to(device)

      his, zis = model(input_batch_i)
      hjs, zjs = model(input_batch_j)

      zis = F.normalize(zis, dim=1)
      zjs = F.normalize(zjs, dim=1)

      loss = nt_xent(zis, zjs)
      loss.backward()
      optimizer.step()
        
      running_loss += loss.item()

    # linear warmup for the first 10 epochs
    if crt_epoch > 10:
      scheduler.step()
      print(optimizer.param_groups[0]['lr'])

    losses.append((crt_epoch, running_loss))
    logger.log("Finished epoch #{} with {} loss".format(crt_epoch, running_loss), show_time = True)
      
    if (crt_epoch - 1) % 5 == 0:
      logger.log("Saving epoch {} with loss {}".format(crt_epoch, running_loss))
      checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 
        'scheduler': scheduler.state_dict()}
      model_filename = "4CPV{}_{}b_{}_{}f_{}_{}{}_3d".format(base_model, batch_size, dataset_type,
        frame_size, sampling_method_str, temporal_transform_str, step)
      logger.save_model(checkpoint, model_filename, crt_epoch, running_loss)

  logger.log("Saving epoch {} with loss {}".format(crt_epoch, running_loss), show_time = True)
  checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 
    'scheduler': scheduler.state_dict()}
  model_filename = "4CPV{}_{}b_{}_{}f_{}_{}{}_3d".format(base_model, batch_size, dataset_type,
    frame_size, sampling_method_str, temporal_transform_str, step)
  logger.save_model(checkpoint, model_filename, crt_epoch, running_loss)

  loss_filename = logger.get_output_file(model_filename + "_loss.pkl")
  logger.log("Saving losses at {}".format(loss_filename))
  with open(loss_filename, "wb") as fp:
    pickle.dump(losses, fp)

  logger.log("Finished training in {:.2f}s".format(time.time() - start_time))


if __name__=='__main__':

  #print("Module file, not main file")
  logger = Logger(show = True, html_output = True, config_file = "config.txt")
  run(logger)
  
  