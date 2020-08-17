# :arrow_down: MScThesis :mortar_board: TU Delft :arrow_down:
## Contrastive Learning of Visual Representations from Unlabeled Videos :video_camera:

Code for my Master Thesis conducted inside the [Pattern Recognition & Bioinformatics research group](https://www.tudelft.nl/ewi/over-de-faculteit/afdelingen/intelligent-systems/pattern-recognition-bioinformatics/) from Delft University of Technology.

My supervisors:
* Head of the [Computer Vision Lab](https://www.tudelft.nl/ewi/over-de-faculteit/afdelingen/intelligent-systems/pattern-recognition-bioinformatics/computer-vision-lab/) and Associate Professor -  [Dr. Jan van Gemert](https://jvgemert.github.io/)
* PhD student - [Osman Semih Kayhan](https://scholar.google.com.hk/citations?user=IQd5igMAAAAJ&hl=en)

## Data :floppy_disk:

For pre-training and evaluation, two action recognition datasets needs to be downloaded: HMDB51 and UCF101.

### HMDB51 :movie_camera:
* Download the train/test splits from [here](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
* Convert from avi to jpg:
```shell
python utils/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory
```
* Generate n_frame files for each video:
```
python utils/n_frames_ucf101_hmdb51.py jpg_video_directory
```
* Generate json annotation files for each split, with `annotation_dir_path` containing \*.txt files:
```
python utils/hmdb51_json.py annotation_dir_path
```

### UCF101 :movie_camera:
* Download the train/test splits from [here](https://www.crcv.ucf.edu/data/UCF101.php)
* Convert from avi to jpg:
```shell
python utils/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory
```
* Generate n_frame files for each video:
```
python utils/n_frames_ucf101_hmdb51.py jpg_video_directory
```
* Generate json annotation files for each split, with `annotation_dir_path` containing \*.txt files:
```
python utils/ucf101_json.py annotation_dir_path
```

> :exclamation: After all of this steps, the data folder should have the following structure:
<pre>
data
│   
│   hmdb51_1.json
|   hmdb51_2.json
|   hmdb51_3.json
|   ucf101_01.json       
|   ucf101_02.json 			
|   ucf101_03.json 		
|
└───hmdb51_videos
|   └───jpg
│       └───brush_hair
|       | folders with jpg and n_frame file for each brush_hair video  
|       |
|       └─── ... 51 folders for each action class
|       |
|       └───wave
|       | folders with jpg and n_frame file for each wave video 
|       └───
|
└───ucf101_videos
|   └───jpg
│       └───ApplyEyeMakeup
|       | folders with jpg and n_frame file for each ApplyEyeMakeup video  
|       |
|       └─── ... 101 folders for each action class
|       |
|       └───YoYo
|       | folders with jpg and n_frame file for each YoYo video 
|       └───
└───
</pre>

## Installation :computer:
The scripts can be run in [Anaconda](https://www.anaconda.com/download/) Windows/Linux environment.

You need to create an Anaconda :snake: `python 3.6` environment named `nonexpert_video`.
Inside that environment some addition packages needs to be installed. Run the following commands inside Anaconda Prompt ⌨:
```shell
(base) conda create -n nonexpert_video python=3.6 anaconda
(base) conda activate nonexpert_video
(nonexpert_video) conda install -c pytorch pytorch
(nonexpert_video) conda install -c pytorch torchvision
(nonexpert_video) conda install -c anaconda cudatoolkit
(nonexpert_video) conda install -c conda-forge tqdm 
```

> :exclamation: For GPU support, NVIDIA CUDA compatible graphic card is needed with proper drivers installed.

## Config file :bookmark_tabs:

<pre>
{
	"data_folder": "data",
	"video_folder": "ucf101_videos",   <em>ucf101_videos or hmdb51_videos</em>
	"frame_folder": "jpg",
	"annotation_file": "ucf101_01.json",   <em>ucf101_01.json or hmdb51_1.json for the 1st split</em>
	"base_convnet": "resnet18",
	"simclr_out_dim": 256,
	"dataset_type": "ucf101",   <em>ucf101 or hmdb51</em>
 	"num_classes": 101,   <em>101 for UCF101 or 51 for HMDB51</em>
	"strength": 0.5,
	"temp": 0.5,
	"batch_size": 256,
	"frame_resize": 56,   <em>56 or 224</em>
  	"sampling_method": "rand32", 
	"temporal_transform_type": "shift",   <em>shift, drop, shuffle, reverse</em>
	"temporal_transform_step": 8,   <em>shift step size</em>
  	"same_per_clip": "True",   <em>False for Frame-mode and True for Chunk-mode</em>
	"model_checkpoint_epoch": 0,   <em> if !=0, load from checkpoint file</em>
	"model_checkpoint_file": ".ptm",   <em>PyTorch saved checkpoint for checkpoint epoch</em>
	"num_epochs": 100,
	"num_workers": 4,  <em>DataLoader number of workers, set accordingly to number of GPUs</em>
}
</pre>

## Usage :arrow_forward:

### Pre-training
* `train_videos_3d.py` for videoSimCLR pre-training
* `train_videos_3d_supervised.py` for fully supervised pre-training

### Linear Evaluation
* `Evaluation_ResNet18_3D_videos.ipynb` for videoSimCLR
* `Evaluation_ResNet18_3D_videos_kinetics.ipynb` for Kinetics pre-trained
* `Evaluation_ResNet18_3D_videos_supervised.ipynb` for supervised pre-trained

### Fine-tuning
* `fine_tune.py`


## Acknowledgements :wave:
* Part of this code is inspired by [Hara et al. 3D Resnets repo](https://github.com/kenshohara/3D-ResNets-PyTorch).
* Part of scripts from `videoMOCO_scripts` adapted from [He et al. Momentum Contrast repo](https://github.com/facebookresearch/moco)
