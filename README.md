# :arrow_down: MScThesis TU Delft :arrow_down:
## Contrastive Learning of Visual Representations from Unlabeled Videos 

Code for my Master Thesis conducted inside the [Pattern Recognition & Bioinformatics research group](https://www.tudelft.nl/ewi/over-de-faculteit/afdelingen/intelligent-systems/pattern-recognition-bioinformatics/) from Delft University of Technology.

My supervisors:
* Head of the [Computer Vision Lab](https://www.tudelft.nl/ewi/over-de-faculteit/afdelingen/intelligent-systems/pattern-recognition-bioinformatics/computer-vision-lab/) and Associate Professor -  [Dr. Jan van Gemert](https://jvgemert.github.io/)
* PhD student - [Osman Semih Kayhan](https://scholar.google.com.hk/citations?user=IQd5igMAAAAJ&hl=en)

## Data :floppy_disk:

For pre-training and evaluation, two action recognition datasets needs to be downloaded: HMDB51 and UCF101.

### HMDB51
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

### UCF101
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

After all of this steps, the data folder should have the following structure:
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
```

