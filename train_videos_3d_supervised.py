import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms

from temporal_transform import TemporalRandomCrop
from resnet3d import generate_model

from ucf101 import get_ucf_dataset
from hmdb51 import get_hmdb_dataset

from logger import Logger

from tqdm import tqdm
import pickle
import time
import os

def run(logger):

  torch.manual_seed(0)

  frame_size = logger.config_dict['frame_resize']
  logger.log("Frames resized to {}x{}".format(frame_size, frame_size))

  spatial_transform  = transforms.Compose([transforms.Resize((frame_size, frame_size)), 
    transforms.ToTensor()])

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
      spatial_transform, temporal_transform = None,
      stack_clip = True, is_simclr_transform = False, apply_same_per_clip = True) 
  elif dataset_type == "hmdb51":
    train_dataset = get_hmdb_dataset(video_path, annotation_path, "training", sampling_method, 
      spatial_transform, temporal_transform = None,
      stack_clip = True, is_simclr_transform = False, apply_same_per_clip = True)
  
  train_loader  = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True, 
    num_workers = num_workers)
  logger.log("{} Train data loaded with {} clips".format(dataset_type.upper(), len(train_dataset)))

  base_model = logger.config_dict['base_convnet']
  num_classes = logger.config_dict['num_classes']
  model = generate_model(model_depth = 18, n_classes = num_classes, add_projection_layers = False)
  logger.log("Model {} 3D simCLR loaded {}".format(base_model, model))

  device = "cuda" if torch.cuda.is_available() else "cpu"
  logger.log("Training on: {}".format("GPU" if device == "cuda" else CPU))

  if torch.cuda.device_count() > 1:
    logger.log("Using {} GPUs".format(torch.cuda.device_count()))
    model = nn.DataParallel(model)

  criterion = torch.nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.01)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 10)
  
  checkpoint_epoch = 0
  if logger.config_dict['model_checkpoint_epoch'] != 0:
    checkpoint_epoch = logger.config_dict['model_checkpoint_epoch']
    checkpoint_file  = logger.config_dict['model_checkpoint_file']
    logger.log("Loading model from checkpoint at epoch {} from {}".format(checkpoint_epoch, checkpoint_file))

    checkpoint = torch.load(logger.get_model_file(checkpoint_file))
    model.load_state_dict(checkpoint['model'])

    optimizer = optim.Adam(model.parameters(), lr=0.1)
    optimizer.load_state_dict(checkpoint['optimizer'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 10)

    if device == "cuda":
      for state in optimizer.state.values():
        for k, v in state.items():
          if torch.is_tensor(v):
            state[k] = v.cuda()

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
    for i, (inputs, labels) in enumerate(train_loader):
      
      optimizer.zero_grad()

      inputs = inputs.to(device)
      labels = labels.to(device)
      
      outputs = model(inputs)
      loss = criterion(outputs, labels)

      loss.backward()
      optimizer.step()

      running_loss += loss.item()

    scheduler.step(running_loss)
   
    losses.append((crt_epoch, running_loss))
    logger.log("Finished epoch #{} with {} loss".format(crt_epoch, running_loss), show_time = True)
      
    if (crt_epoch - 1) % 5 == 0:
      logger.log("Saving epoch {} with loss {}".format(crt_epoch, running_loss), show_time = True)
      checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
      model_filename = "{}_sup_{}_{}f_{}_3d".format(base_model, dataset_type, frame_size, sampling_method_str)
      logger.save_model(checkpoint, model_filename, crt_epoch, running_loss)

  logger.log("Saving epoch {} with loss {}".format(crt_epoch, running_loss), show_time = True)
  checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
  model_filename = "{}_sup_{}_{}f_{}_3d".format(base_model, dataset_type, frame_size, sampling_method_str)
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
  