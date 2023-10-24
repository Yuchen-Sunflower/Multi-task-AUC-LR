# An_Adaptive_Low_Rank_Algorithm_for_Multi_Task_AUC_Learning
This repository is the official implementation of our paper "An_Adaptive_Low_Rank_Algorithm_for_Multi_Task_AUC_Learning"

## Datasets

We have provided the datasets we use. Some datasets could also be downloaded at the following websites:

MHC-I: http://tools.iedb.org/main/datasets/

USPS: http://gaussianprocess.org/gpml/data/

## Run codes

Before run our code, you need to modify the feature number and loading dataset. For example, if you want to run experiments on MHC-I, you should set 

```python
feat_num = 180
# 9 landmine
# 180 MHC-I
# 256 USPS
# 80 Simulate

training_data, validation_data, training_label, validation_label = read_data.mhc_split_valid(mhc_dir)
training_data, training_label, testing_data, testing_label = read_data.mhc_split(mhc_dir)
```

Select corresponding code of MTAUC-FNNFN in "\_\_main\_\_":

```python
for i in range(iter):
        auc_i, time_i = reAUC_test(params)
        auc.append(auc_i * 100)
        avg_time.append(time_i)
    print('Avg auc: {:.1f}'.format(np.sum(auc) / iter))
    print('Standard deviation: {:.2f}'.format(np.std(auc)))
    print('Each iteration time: {:.4f}'.format(sum(avg_time) / len(avg_time)))
```

Then run the code:

```python
python3 optimizer.py
```
