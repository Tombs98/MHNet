
## Training
- Download datasets from the google drive links and place them in this directory. Your directory structure should look something like this
  
  `Synthetic_Rain_Datasets` <br/>
  `├──`[train](https://drive.google.com/drive/folders/1Hnnlc5kI0v9_BtfMytC2LR5VpLAFZtVe?usp=sharing)  <br/>
  `└──`[test](https://drive.google.com/drive/folders/1PDWggNh8ylevFmrjo-JEvlmqsDlWWvZs?usp=sharing)  <br/>
      `├──Test100`   <br/>
      `├──Rain100H`  <br/>
      `├──Rain100L`  <br/>
      `├──Test1200`  <br/>
      


- Train the model with default arguments by running

```
python train.py
```


## Evaluation

1. Download the [model](https://drive.google.com/file/d/18P-lAXRXZa2gr7NW9NB_LrNrb4lsiyAg/view?usp=sharing) and place it in `./pretrained_models/`

2. Download test datasets (Test100, Rain100H, Rain100L, Test1200) from [here](https://drive.google.com/drive/folders/1PDWggNh8ylevFmrjo-JEvlmqsDlWWvZs?usp=sharing) and place them in `./Datasets/Synthetic_Rain_Datasets/test/`

3. Run
```
python test.py
```

#### To reproduce PSNR/SSIM scores of the paper, run
```
python eval.py
```
