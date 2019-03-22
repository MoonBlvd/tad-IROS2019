# Unsupervised Traffic Accident Detection in First-Person Videos

*Yu Yao, Mingze Xu, Yuchen Wang, David Crandall and Ella Atkins*

This repo contains the code for our [paper](https://arxiv.org/pdf/1903.00618.pdf) on unsupervised traffic accident detection.

:boom: The full code will be released upon the acceptance of our paper.

:boom: So far we have released the pytorch implementation of our ICRA paper [*Egocentric Vision-based Future Vehicle Localization for Intelligent Driving Assistance Systems*](https://arxiv.org/pdf/1809.07408.pdf), which is an important building block for the traffic accident detection. The original project repo is https://github.com/MoonBlvd/fvl-ICRA2019

## Future Object Localization
To train the model, run:

	python train_fol.py --load_config YOUR_CONFIG_FILE

To test the model, run:

	python test_fol.py --load_config YOUR_CONFIG_FILE
 
 An example of the config file can be found in ```config/fol_ego_train.yaml```

#### evaluation result
Note that we have only evaluated the model performance with prediction horizon 0.5 seconds. We are working on proving the 1 second and 2 seconds results.

|     Model      | pred horizon | FDE  | ADE | FIOU |
|:--------------:|--------------|------|-----|------|
| FOL + Ego pred | 0.5 sec      | 10.9 | 6.6 | 0.95 |

## Run detection
Go to Mask-RCNN root directory run:

	python run_inference.py \
        -i /media/DATA/VAD_datasets/taiwan_sa/testing/frames \
        -o /media/DATA/VAD_datasets/taiwan_sa/testing/mask_rcnn_detections \
        --for_deepsort \
        --image_shape 1280 720 3 \
        -g=0

## Run tracking
Go to deep-sort root directory run:

	python deep_sort_app.py \
    --sequence_dir=/media/DATA/VAD_datasets/taiwan_sa/testing/frames \
    --detection_dir=/media/DATA/VAD_datasets/taiwan_sa/testing/mask_rcnn_detections \
    --min_confidence=0.3 \
    --nms_max_overlap=0.5 \
    --output_dir=/media/DATA/VAD_datasets/taiwan_sa/testing/deep_sort:w

## Run flownet2
Go to flownet2 root directory run:

	export CUDA_VISIBLE_DEVICES=0
	python main.py \
	    --skip_validation \
	    --skip_train \
	    --inference \
	    --no_loss \
	    --save_flow \
	    --model FlowNet2 \
	    --inference_dataset TaiwanSA \
	    --inference_dataset_root data/taiwan_sa/testing \
	    --inference_size 320 192 \
	    --resume checkpoints/FlowNet2_checkpoint.pth.tar

## Run ORB-SLAM2 for ego motion
Go to orb_slam2 root directory run:
	
	python rgb2gray.py --help
	
to make sure there is a 'time.txt' file in each video's folder. Make sure the time length is greather than or equal to the video length(!!)

Then run

	python run_all_videos.py --help

to generate odometry outputs.


## Train ego motion prediction

1. Run scripts/odo_to_ego_motion.py to convert the ```.txt``` odometry files to ```.npy``` files containing yaw, x, z. Note that training and validation data are separately created
2. Run ```python train_ego_pred.py``` to train a RNN-ED ego motion prediction model.

## Train FVL + ego motion preditcion

1. Make sure ego motion model has been pretrained
2. In config/fvl_config.yaml indicate the best checkpoint of the ego motion prediction model
3. run  ```python train.py``` to train the FVL-ego model
