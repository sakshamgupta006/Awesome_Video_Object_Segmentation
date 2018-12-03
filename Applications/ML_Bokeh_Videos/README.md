# Machine-Learned-Bokeh-Videos

This application tries to mimic bokeh effect on videos using Mask-RCNN and bluring. The application can be used for any video having single/multiple persons by initially selecting the Person of Interest.

## Setup
1. Download the model file from Mask-RCNN([mask_rcnn_coco.h5](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5))
2. Copy this file in the model folder.
3. Use `requirements.txt` file to setup the virtual or conda env.
4. Load/Setup cuda/9.0.xx and cudnn/7.xx 

## Running Instructions
1. Put your Input Video in the input folder.
2. Run the following command
```
python -input <Video Name> -output <Output Name> -model <Model Name> 
```

## Future Work