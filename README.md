The project is about detecting sarcasm in the text and identifying whether the text is sarcastic or not.

### Dataset 
[Reddit dataset](https://www.kaggle.com/datasets/danofer/sarcasm/data?select=train-balanced-sarcasm.csv) was used for sarcasm training and for word embedding [crawl-300d-2M](https://fasttext.cc/docs/en/english-vectors.html) was used.

### File instructions

The main model is located in `sarcasm_detecion.py` whlile the corresponding `.h5` file saves its weight.

`app.py` file contains the flask code to integrate the frontend with backend and the frontend files for the site are located in `static` and `templates` folder. Please note that due to some constraints site is not live but can be seen locally by running the provided files.

The `SDE` notebook contains the code for different model which highlites the words based on the attention weight.

### Timing information

The main model takes around 60 mins for training if we train it for 40 epochs and the attention model takes 70 mins for the same epochs. The time required to show the output when input is provided on the website is negligible.
