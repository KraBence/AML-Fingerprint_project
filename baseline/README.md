## Important Note
To run the code efficiently, a GPU is strongly recommended, and it may not run properly without one. Be careful not to execute the train_model() line, as this will retrain the model.

# Models
We have developed two models: one for predicting gender and another for predicting hand (left or right). Both models share the same architecture, consisting of three residual blocks followed by a fully connected layer for classification.

## Gender Prediction
To predict gender, load the model from the file 'gender_predict.pt'. Ensure that in both test_dataloader() and val_dataloader(), the target value is set to 'gender'.
The accuracy on the test set is %.

## Hand Prediction
To predict hand, load the model from the file 'hand_predict.pt'. Ensure that in both test_dataloader() and val_dataloader(), the target value is set to 'hand'.
The accuracy on the test set is %.

## Data Splitting
The dataset contains 600 subjects. Data is split based on the subjects' IDs: the first 500 subjects are used for training, while the remaining 100 subjects are evenly split between testing and validation.
