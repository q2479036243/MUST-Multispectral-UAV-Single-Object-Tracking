## 设置环境
运行以下指令
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
其中output用于保存模型训练、测试、验证结果；data为数据集路径，也可以直接在下面文件内更改
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```


## 优化数据集
处理高光谱数据集，从npy格式另存为多张jpg图像，可以有效加速数据读取和降低CPU占用（非必要操作，可以直接np.load加载npy文件）
```
python preprocess_datasets/must.py
```


## 权重预处理
下载预训练模型 [MAE ViT-Base weights](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) 并把它放在 `$PROJECT_ROOT$/pretrained_networks` 文件夹下，并运行下列指令
```
python pretrained_networks/trans_model.py
```


## 训练模型
```
python tracking/train.py --script untrack --config baseline_must --save_dir ./output --mode multiple --nproc_per_node 3 --use_wandb 0
```


## 测试模型
```
python tracking/test.py untrack baseline_must --dataset MUSTHSI --runid 50 --threads 12 --num_gpus 3
python tracking/analysis_results.py
```

## 可视化结果
```
python -m visdom.server
CUDA_VISIBLE_DEVICES=2 python tracking/test.py untrack baseline --dataset MUSTHSI --runid 50 --threads 1 --num_gpus 1 --debug 1
```