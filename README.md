# Unsupervised Traffic Accident Detection in First-Person Videos

*Yu Yao, Mingze Xu, Yuchen Wang, David Crandall and Ella Atkins*

This repo contains the code for our [paper](https://arxiv.org/pdf/1903.00618.pdf) on unsupervised traffic accident detection.

:boom: The full code will be released upon the acceptance of our paper.

:boom: So far we have released the pytorch implementation of our ICRA paper [*Egocentric Vision-based Future Vehicle Localization for Intelligent Driving Assistance Systems*](https://arxiv.org/pdf/1809.07408.pdf), which is an important building block for the traffic accident detection. The original project repo is https://github.com/MoonBlvd/fvl-ICRA2019

<img src="figures/teaser.png" width="400">

## Requirements
To run the code on feature-ready HEV-I dataset or dataset prepared in HEV-I style:

	cuda9.0 or newer
	pytorch 1.0
	torchsummaryX
	tensorboardX
## Dataset and features
### HEV-I dataset
**Note:**  Honda Research Institute is still working on preparing the videos in HEV-I dataset. The planned release date will be around May 20 2019 during the ICRA.

However, we provide the newly generated features here in case you are interested in just using the input features to test your models:

[Training features](https://drive.google.com/open?id=1TE-smXm4dD2QgoCQHYmzLHqSsltoIxbe)

[Validation features](https://drive.google.com/open?id=1Vcu6NU7PwDOPTv6RU_7AuBfj6I0rj4dR)

Each feature file is name as "*VideoName*_*ObjectID*.pkl". Each .pkl file includes 4 attributes:.
* frame_id: the temporal location of the object in the video;
* bbox: the bounding box of the object from it appears to it disappears;
* flow: the corresponding optical flow features of the object obtained from the ROIPool;
* ego_motion: the corresponding [yaw, x, z] value of ego car odometry obtained from the orbslam2.


To prepare the features used in this work, we used:
* Detection: [MaskRCNN](https://github.com/matterport/Mask_RCNN)
* Tracking: [DeepSort](https://github.com/nwojke/deep_sort)
* Dense optical flow: [FlowNet2.0](https://github.com/NVIDIA/flownet2-pytorch)
* Ego motion: [ORBSLAM2](https://github.com/raulmur/ORB_SLAM2)

### A3D dataset

## Future Object Localization

<img src="figures/fol.png" width="400">

To train the model, run:

	python train_fol.py --load_config YOUR_CONFIG_FILE

To test the model, run:

	python test_fol.py --load_config YOUR_CONFIG_FILE
 
 An example of the config file can be found in ```config/fol_ego_train.yaml```

### Evaluation results on HEV-I dataset
We do not slipt the dataset into easy and challenge cases as we did in the original repo. Instead we evalute all cases together. We are still updating the following results table by changing the prediction horizon and the ablation models.

|     Model      | train seg length | pred horizon | FDE  | ADE  | FIOU |
| :------------: | ---------------- | ------------ | ---- | ---- | ---- |
| FOL + Ego pred | 1.6 sec          | 0.5 sec      | 11.0 | 6.7  | 0.85 |
| FOL + Ego pred | 1.6 sec          | 1.0 sec      | 24.7 | 12.6 | 0.73 |
| FOL + Ego pred | 1.6 sec          | 1.5 sec      | 44.1 | 20.4 | 0.61 |
| FOL + Ego pred | 3.2 sec          | 2.0 sec      | N/A  | N/A  | N/A  |
| FOL + Ego pred | 3.2 sec          | 2.5 sec      | N/A  | N/A  | N/A  |

**Note**: Due to the change of model structure, the above evaluation results can be different from the original paper. The users are encouraged to compare with the result listed in this repo since the new model structure is more efficient than the model proposed in the original paper.

## Citation
If you found the repo is useful, please feel free to cite our papers:

	@article{yao2018egocentric,
	title={Egocentric Vision-based Future Vehicle Localization for Intelligent Driving Assistance Systems},
	author={Yao, Yu and Xu, Mingze and Choi, Chiho and Crandall, David J and Atkins, Ella M and Dariush, Behzad},
	journal={arXiv preprint arXiv:1809.07408},
	year={2018}
	}



	@article{yao2019unsupervised,
	title={Unsupervised Traffic Accident Detection in First-Person Videos},
	author={Yao, Yu and Xu, Mingze and Wang, Yuchen and Crandall, David J and Atkins, Ella M},
	journal={arXiv preprint arXiv:1903.00618},
	year={2019}
	}
<!-- ## Run detection
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

to generate odometry outputs. -->


<!-- ## Train ego motion prediction

1. Run scripts/odo_to_ego_motion.py to convert the ```.txt``` odometry files to ```.npy``` files containing yaw, xhttps://github.com/NVIDIA/flownet2-pytorch, z. Note that trainingtensorboardXand validation data are separately createdhttps://github.com/NVIDIA/flownet2-pytorch
2. Run ```python train_https://github.com/NVIDIA/flownet2-pytorchego_pred.py``` to traintensorboardXa RNN-ED ego motion prediction model.
https://github.com/NVIDIA/flownet2-pytorch
## Train FVL + ego motihttps://github.com/NVIDIA/flownet2-pytorchon preditcion
https://github.com/NVIDIA/flownet2-pytorch
1. Make sure ego motionhttps://github.com/NVIDIA/flownet2-pytorch model has been pretrained
2. In config/fvl_config.yaml indicate the best checkpoint of the ego motion prediction model
3. run  ```python train.py``` to train the FVL-ego model -->
