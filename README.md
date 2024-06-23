# Maxillofacial Bone Movements-Aware Dual Graph Convolution Approach for Postoperative Facial Appearance Prediction
This is the repository of DGCFP, a postoperative facial appearance prediction model, for the orthognathic surgery planning. 

## Project Dependencies
Here are some important packages.
```
python>=3.7
pytorch>=1.13
torch_geometric==2.2.0
open3d==0.16.0
pytorch3d==0.7.4
pymeshlab==2022.2.post2
```
The more detailed dependencies can be checked in the ```requirments.txt```.

## Project Configuration
Preparing data.

Please prepairing cropped 3D facial and bony models from paired preoperative and postoperative CT scans of patients following the instructions provided in the paper.
All files need to be organized into a proper file structure.
Here is our directory structure.
```
|-/original_data_dir
|--/patient_id
|---/crop
|----pre_bone.ply
|----pre_bone_plan.ply
|----pre_face.ply
|----post_face.ply
|---post_face_landmarks.csv
|---pre_face_landmarks.csv
```

Processing the cropped data to generate datasets.
```shell
python dataprocess.py
```

Performing training with configuration ```xxxxx.json```.
```shell
CUDA_VISIBLE_DEVICES=x python main.py --is_train --config ./configuration/xxxxx.json
```

Generating predictions based on trained model.
```shell
CUDA_VISIBLE_DEVICES=x python main.py --config ./configuration/xxxxx.json
```
## Dataset
We are sorry, but due to our agreement with our partners, we are unable to provide the data. Please prepare your own data and organize them as mentioned above.