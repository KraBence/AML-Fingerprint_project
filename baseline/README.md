To run the code you may need GPU.
Watch to do not run the train_model() line, because it will retrain the model.
# Models
We have developed two models: one for predicting gender and another for predicting the hand (left or right). Both models share the same architecture, consisting of three residual blocks followed by a fully connected layer for classification.

## Gender Prediction
To predict gender, load the model using the file 'gender_predict.pt', and in the test_dataleader() and the val_dataloader set the predicted value to 'hand'. 

The accuracy on the test set is %.

## Hand Prediction
To predict the hand, load the model using the file 'hand_predict.pt', and in the test_dataleader() and the val_dataloader set the predicted value to 'hand'. 

The accuracy on the test set is %.

## Data Splitting
the data contains 600 subjects. We split the data based on the IDs of the subjects. The first 500 subjects are used for training, while the remaining data is evenly split between testing and validation.
