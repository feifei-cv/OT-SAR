
**Unsupervised Image Animation via Optimal Transport and
Spatial-Adaptive Refinement**

* YAML configs
There are several configuration (```config/dataset_name.yaml```) files one for each `dataset`. 

* Training and Evaluation together:
    ```
    # Training for each dataset
    CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --config config/dataset_name.yaml --device_ids 0,1,2,3

    # Evaluation for each dataset
    CUDA_VISIBLE_DEVICES=0 python run.py --config config/dataset_name.yaml --mode reconstruction --checkpoint path/to/checkpoint
    ```
## Acknowledgments
This code is heavily based on: [FOMM](https://github.com/AliaksandrSiarohin/first-order-model) and [3d-corenet](https://github.com/ChaoyueSong/3d-corenet). We thank the authors for making the source code publicly available.




