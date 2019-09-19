# Keras_VGG
paper from https://arxiv.org/abs/1409.1556  
(only VGG16)
## Getting start
 - Keras 2.2.4  
 - tensorflow-gpu 1.9.0   
 - numpy 1.16.3 
## First. prepare data: 
### 1. revise config/data.ini 
[data]  
path = the folder absolute path where the data is.  
classes_path = the classes names of all data.  

[training]  
filename = save the training data path in train.txt.  
proportion = the proportion with training data.  

[validation]  
filename = save the validation data path in val.txt.  
proportion = the proportion with validation data.  

[testing]  
filename = save the testing data path in test.txt.  
proportion = the proportion with testing data.  

### 2. execution prepare_data.py
```
python prepare_data.py
```
---
## Second. train model:
### 1. revise config/model.ini
[data]  
train_set = train.txt path(same as config/data.ini training/filename).  
val_set = val.txt path(same as config/data.ini validation/filename).  

[model]  
input_size = model input shape is (input_size, input_size, 3).  
num_classes = number of classes names.  

[train]  
epochs =   
batch_size =   
learning_rate =   
save_path = final model save path.  
pretrained_path = if pre-trained model path, or do not fill in.  

[gpu]  
gpu = specified GPU to train model.  

### 2. execution train.py
```
python train.py
```
