# Models
We have developed two models: one for predicting gender and another for predicting the hand (left or right). Both models share the same architecture, consisting of three residual blocks followed by a fully connected layer for classification.

## Gender Prediction
To predict gender, load the model using the file 'gender_predict_split_by_id_model2.pt'.

## Hand Prediction
To predict the hand, load the model using the file 'hand_predict_split_by_id_model2.pt'.

## Data Splitting
The dataset is split based on the IDs of the subjects. The first 500 subjects are used for training, while the remaining data is evenly split between testing and validation.
