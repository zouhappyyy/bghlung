# BHNet for HUST-Lung Dataset 

## Preparing
1. install Environment

```
pip install -e .
```


2. make dataset
```
python step2_make_nnunet_set_supervision.py
```

3. training model,read step3_train_models.sh and can run training model scripts as follows:
```bash
CUDA_VISIBLE_DEVICES=1 nohup /home/pubenv/anaconda3/envs/zyzbrain18/bin/python ./nnunet/run/run_training.py 3d_fullres BGHNetV4Trainer Task702_LungTumor 0 > ./BGHNetV4Trainer_Fold0.log 2>&1 &
```

4. evaluate model performance
```bash
python step4_pred_val_results_and_eval.py
```

5. predict unlabeled CT scans
```bash
python step5_20240401_pred_test_lung_data.py
python step5_20240511_pred_test_lung_data.py
```
