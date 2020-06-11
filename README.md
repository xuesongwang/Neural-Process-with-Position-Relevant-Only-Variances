
# NP-PROV: Neural Processes with Position-Relevant-Only Variances

This repository is the official implementation of [NP-PROV: Neural Processes with Position-Relevant-Only Variances](https://arxiv.org/abs/2030.12345). 

<p align="center">
<img src="demo_images/NP-PROV-MU.jpg" width="200"> <img src="demo_images/NP-PROV-SIGMA.jpg" width="200">
</p>

## Requirements
* Python 3.6 or higher.

* `gcc` and `gfortran`:
    On OS X, these are both installed with `brew install gcc`.
    On Linux, `gcc` is most likely already available,
    and `gfortran` can be installed with `apt-get install gfortran`.
    

Install the requirements and You should now be ready to go!

```bash
pip install -r requirements.txt
```


## Training

To train the model(s) for off-the-grid datasets, run this command:

```train
python train_1d.py --name EQ --epochs 200 --learning_rate 3e-4 --weight_decay 1e-5
```

The first argument, `name`(`default = EQ`), specifies the data that the model will be trained
on, and should be one of the following:
 
* `EQ`: samples from a GP with an exponentiated quadratic (EQ) kernel;
* `matern`: samples from a GP with a Matern-5/2 kernel;
* `period`: samples from a GP with a weakly-periodic kernel
* `smart_meter`: This dataset is referred from: https://github.com/3springs/attentive-neural-processes/tree/RANPfSD/data 
 To train on smart_meter, you need to change the argument `indir` in the function of `get_smartmeter_df` in `data/smart_meter.py` 
 to your own data path. 
           
           

## Evaluation

To evaluate my model on off-the-grid datasets, run:

```eval
python eval_1d.py --name EQ
```
The argument is the same as in train_1d.py. A model called `name` + `_model.pt`
will be loaded from the folder `saved_model`


## Results

Our model achieves the following log-likelihood with mean (variance) on the following off-the-grid datasets:



| Model name         | EQ              | Matern         |  Period       | Smart Meter   |
| ------------------ |---------------- | -------------- |-------------- | -------------- |
| NP-PROV            | 2.20 (0.02)     |    0.90 (0.03) |  -1.00 (0.02) | 2.32 (0.05)
