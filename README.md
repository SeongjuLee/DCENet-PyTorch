# DCENet-PyTorch
PyTorch Implementation of DCENet (https://arxiv.org/abs/2010.16267) for Trajectory Forecasting


## Requirements
* `python 3.8`
* `pytorch 1.7.1`
* `matplotlib`
* `scipy`
* `neptune-client` if you need


## Dataset
Processed data (`./processed_data/train/train_merged0.npz`, `./processed_data/train/train_merged1.npz`, `./processed_data/train/train_merged2.npz`, `./processed_data/train/biwi_hotel.npz`) is requried.

You can obtain the processed data from the original repository (https://github.com/tanjatang/DCENet).

## Train
* Command for training

`python main.py --gpu $GPU_NUMS --config $CONFIG_FILENAME`

* Example

`python main.py --gpu 0 --config config.json`

## Test
* Command for evaluation

`python evaluate.py --gpu $GPU_NUMS --config $CONFIG_FILENAME --resume-name #CHECKPOINT_FILENAME`

* Example

`python evaluate.py --gpu 0 --config config.json --resume-name best_model.pth`

## Performance
### Evaluation Results @Top25
|      Criteria      	| Original Implementation (Tensorflow) 	| My Implementation (PyTorch) 	|
|:------------------:	|:------------------------------------:	|:---------------------------:	|
|         ADE        	|                0.37 m                	|            0.36 m           	|
|         FDE        	|                0.76 m                	|            0.67 m           	|
| Hausdorff Distance 	|                0.75 m                	|            0.67 m           	|
|   Speed Deviation  	|               0.06 m/s               	|           0.05 m/s          	|
|    Heading Error   	|                 25.60                	|            24.67            	|

### Evaluation Results for Most-likely Predictions
|      Criteria      	| Original Implementation (Tensorflow) 	| My Implementation (PyTorch) 	|
|:------------------:	|:------------------------------------:	|:---------------------------:	|
|         ADE        	|                0.39 m                	|            0.42 m           	|
|         FDE        	|                0.78 m                	|            0.79 m           	|
| Hausdorff Distance 	|                0.77 m                	|            0.78 m           	|
|   Speed Deviation  	|               0.06 m/s               	|           0.05 m/s          	|
|    Heading Error   	|                 30.98                	|            30.62            	|


## License
Model details and most of utility functions are from from the origianl DCENet repository (https://github.com/tanjatang/DCENet).

Codes for progress bar came from https://github.com/AaronHeee/MEAL.

Codes for early stopping came from https://github.com/Bjarten/early-stopping-pytorch.

## Who Am I?
I am on Ph.D course in Artificial Intelligence Lab. ([Homepage](https://ailab.gist.ac.kr/ailab/)), Gwangju Institute of Science and Technology (GIST, [Homepage](https://www.gist.ac.kr/kr/event_2st/index.html)), Korea.
