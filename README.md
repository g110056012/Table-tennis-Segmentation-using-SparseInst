# Table-tennis using SparseInst 

### 1. Refereeence sites:
1. SparseInst github: https://github.com/hustvl/SparseInst
2. ResNeSt github: https://github.com/chongruo/detectron2-ResNeSt

### 2. Enviornments:
#### SparseInst:
    git clone https://github.com/hustvl/SparseInst.git
#### Install the detectron2:
    git clone https://github.com/facebookresearch/detectron2.git
#### ResNeSt:
    pip install git+https://github.com/zhanghang1989/ResNeSt
#### (Environment yaml):
    conda env create -f /path/to/environment.yml  

### 3. Before Training
#### Parameters setting
    /path/to/SparseInst/sparseinst/config.py
    cfg.MODEL.SPARSE_INST.DECODER.NUM_MASKS = 100
    cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES = 80
#### Config files setting
    /path/to/SparseInst/table-tennis/ball_configs/

### 4. Commmand Line:
#### Train model 
    python tools/ball_train_net.py --config-file table-tennis/ball_configs/ttball_sparse_inst_r50vd_dcn_giam_aug.yaml --num-gpu 1
#### Test model 
    python tools/ball_test_net.py --config-file table-tennis/output/testing11_ResNet_ttball_sparse_inst_r50_giam/config.yaml MODEL.WEIGHTS table-tennis/output/testing11_ResNet_ttball_sparse_inst_r50_giam INPUT.MIN_SIZE_TEST 512
#### Demo (For example 正手拍)
    python demo.py --config-file table-tennis/output/testing11_ResNet_ttball_sparse_inst_r50_giam/config.yaml --video-input table-tennis/ball_data/0621/正手/IMG_7392_Trim.mp4 --output table-tennis/result/ --opt MODEL.WEIGHTS table-tennis/output/testing11_ResNet_ttball_sparse_inst_r50_giam/model_final.pth INPUT.MIN_SIZE_TEST 512
    
