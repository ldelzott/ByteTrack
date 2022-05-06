# Primitive swarm visualization tool on top of [ByteTrack](https://github.com/ifzhang/ByteTrack)
## Installation steps
One should first clone this repository and install the dependencies.  
```shell
git clone https://github.com/ldelzott/ByteTrack.git
pip3 install folium==0.2.1
pip3 install -r requirements.txt
pip install tinydb
pip install pysimplegui
pip uninstall -y torch torchvision torchaudio
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
python3 setup.py develop
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install cython_bbox

```
One should download a pre-trained YOLOX model. This model must be placed in ByteTrack/pretrained.
```shell
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth
```
The train.py utility is used to train the model. The training is done on the dataset provided in ByteTrack/datasets/epucks_dataset_300
```shell
python3 /content/ByteTrack/tools/train.py -f /content/ByteTrack/exps/example/mot/yolox_x_mix_det.py -d 1 -b 2 --fp16 -o -c /content/ByteTrack/pretrained/yolox_x.pth
```
The visualization tool can be launched on the provided input video sequence aggregation-8-30fps-cropped.mp4
```shell
python3 /content/ByteTrack/tools/demo_track.py video --path /content/ByteTrack/videos/aggregation-8-30fps-cropped.mp4 -f /content/ByteTrack/exps/example/mot/yolox_x_mix_det.py -c /content/ByteTrack/YOLOX_outputs/yolox_x_mix_det/latest_ckpt.pth.tar --fp16 --fuse --save_result
```
## Functionalities 
# Graphs
The graphs show the value of metrics computed in real-time on the tracked objects. The current graphs range from 1 to 40 frames.  

<img src="assets/graphvisualization1.gif" width="700"/>

# Configuration of the metrics
## Trails visualization
Shows the trails of the tracked objects. The maximum number of displayed points is 8000. The value can be adjusted during run time. 

<img src="assets/1_trails_settings.gif" width="900"/>


## Velocity vectors
Shows an approximation of a velocity vector on each tracked object. 

<img src="assets/2_velocity_vectors.gif" width="900"/>


## Fastest entity
Highlight the fastest moving entity, based on the computed velocity vectors.

<img src="assets/3_fastest_entity.gif" width="900"/>


## Combined visualizations
Any of the previous visualizations can be superposed during run time. 

<img src="assets/4_all_metrics.gif" width="900"/>


## Global position heatmap
The radius of the masks used to generate the heat of each tracked object is overemphasized. The second example shows the heatmap obtained for a smaller mask radius. 

<img src="assets/5_global_heatmap.gif" width="900"/>
<img src="assets/5_global_heatmap_2.gif" width="900"/>


## Individual position heatmap
The position heatmap of each tracked entity can be displayed. The mask radius can be adjusted. 
<img src="assets/6_individual_heatmap.gif" width="900"/>


## Networks map
Assuming objects can communicate with their neighbors under a given distance, the network map highlights the groups of objects in the same "communication cluster".      
<img src="assets/7_networks_map.gif" width="900"/>

# Recording
The process generate an output video. This output video shows the manipulations done in unpaused state in the visualization tool.
[video](https://github.com/ldelzott/ByteTrack/blob/main/assets/output_4_cropped_and_resized.mp4)


