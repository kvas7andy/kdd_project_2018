Src files utilize PyTorch deep learning framework for model construction and learning process on GPU.


| File/Folder          | Description           | 
| ------------- |:-------------:| 
| ├── main.py | Main execution file: 1) _If not already computed and saved_ prepare centres from ImageNet: outputs `train_centres.txt` file in [../datasets/imagenet](../datasets/imagenet) 2) Compute the transformations of the layers from centres (which are all the way fixed) 3) Train network, with `active-batch`/`active-batch with class balancing`/`randomly` sampling. 4) Save the fine-tuned model after 100 batches | 
| ├── data_prepare.py  | File for dataset preparation: utilizes `train.txt` and/or `eval.txt` files in datasets folders (like [../datasets/imagenet])[../datasets/imagenet] to prepare ImageNet dataset loader into the model, as well as target/new dataset.| 
| ├── distinctiveness_and_uncertainty.py | Numpy matrix computations and Kendall tau-coefficient fast implementation (from `scipy.spatial.distance` and `scipy.stats.kendalltau` respectively) to compute distinctiveness and uncertainty. Uses precomputed feature transformations of centres for faster computation of score: `score = (1-\lambda t) \cdot distinctiveness + \lambda t \cdot uncertainty ` | 
| ├── model.py | The implementations of Alex & VGG16 networks used from `torchvision.models` and freezing first layers & taking features from corresponding layers A and B, described in paper | 
| ├── optimizer.py | Standard Adam optimizer |
|  ├── trainer.py | * Compute centers with `prepare_centers()` for 1) point of algorithm. * train algorithm with `train()` on training dataset * evaluate with `eval()` the fine-tuned model on evaluation dataset during training process   | 
|  └── utils.py | Specify all parameters for the algorithm, see [root folder README.md](../README.md) |

                        










