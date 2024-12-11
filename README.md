Our dataset is the Soco finger dataset with 6000 fingerprint images from 600 subjects (so there are 10 fingerprints per subject).
The database also contains artificially modified fingerprints for each individual, which have been altered at easy, medium, and hard levels based on three methods. The three used methods are: obliteration, central rotation, and z-cut.
The dataset is publicly avaiable at: https://www.kaggle.com/datasets/ruizgara/socofing/data .
Further information can be found about the dataset here: https://arxiv.org/pdf/1807.10609



What is our plans?:
We would like to develop 3 models for detect the following things(if we have time we
would like to merge these three models into one model to calculate these things
simultaneously).:

-Fingerprint recognition: Categorize fingerprints that they identity is between the
list of acceptable persons or not.

-Gender predicting: Decide that the finger belong to a male of a female
   
-Finger detection: Decide which finger is which finger (thumb, index, middle, ring,
little)

Here you can see what the files does in the repository:
│<br>
├── **Data_preparatin.ipynb**              # In this notebook we transfrmedabd visualised images<br>
├── **Finger_functions.py**              # creating .csv from images.<br>
├── **plots.ipynb**            # Here are the final results plotted<br>
├── **requirements.txt**                   # install this file with pip install<br>
├── **vgg16.ipynb**      # You can run, train, and test a model<br>
