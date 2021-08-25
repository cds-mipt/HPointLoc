# HPointLoc
Framework for visual place recognition and localization. This project is a complete implementation of paper "HPointLoc: open dataset and framework for indoor visual localizationbased on synthetic RGB-D images"

This repository provides a novel framework **PNTR** for exploring new indoor datasets - **HPointLoc**, specially designed to explore detection and loop closure capabilities in Simultaneous Localization and Mapping (SLAM).

Our paper on deep learning-based visual place recognition contains detailed information about these datasets:
> BibteX: HPointLoc: open dataset and framework for indoor visual localizationbased on synthetic RGB-D images*

HPointLoc is based on the popular Habitat simulator from 49 photorealistic indoor scenes from the Matterport3D dataset and contains 76,000 frames.
<p align="center">
  <img src="https://user-images.githubusercontent.com/68793107/130797278-615f72c7-0528-4eff-af95-a7e07bf1fea3.png" />
</p> 


When forming the dataset, considerable attention was paid to the presence of instance segmentation of scene objects, which will allow it to be used in new emerging semantic methods for place recognition and localization
<p align="center">
  <img src="https://user-images.githubusercontent.com/68793107/130794869-ea0388e6-f19c-4c83-989a-64d79622db2a.png" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/68793107/130797488-9d1bfe09-180e-48c1-9900-f7342e0a771d.png" />
</p>





## Quick start to evaluate PNTR pipeline

```bash
git clone --recurse-submodules https://github.com/cds-mipt/HPointLoc
conda env create -f environment.yml
python pipelines/pipeline_evaluate.py --dataset_root /path/to/dataset --image-retrieval 'patchnetvlad' --keypoints-matching 'superpoint_superglue' --optimizer-cloud 'teaser' -f  
```

