# The target of this project
it's a simple demo to use deep learning to find pattern in trading ticker data 

# env preparation
followed the video on youtube:  https://www.youtube.com/watch?v=2C-B1VFMq58, still some bugs

'''
     conda install -c apple tensorflow-deps
'''

if that doesn't work, please download the package from conda official website, conda install XXXX

'''
    pip install tensorflow-macos
'''

please don't install tensorflow-metal, that will cause some problem in this example.


# data preparation h
The csv file is in data folder. Since the training data without time series, we trim the timestamp and the identifier to fit the data. Eventuall save it into numpy data for future usage. For data label, mark the point where you order, choose the location with intuitive selection, then let the machine have the clear logic. 

```
python prepare.py
```


# model training
data label is far from done, currently it's just price change ratio selection. Later on more sophisticated way should be there.
```
python model.py
```


# interpret
Connect the model with the real time data, to determine positions. That will be evetually goal. 
```
python interpret.py
```


currently 