A. Data Preprocessing
Text data from Shakespeare’s plays was tokenized into
individual words. Stopwords were removed to focus the training on meaningful words. The text was then converted into
sequences of tokens, which served as the input for the LSTM
model

B. Model Architecture
The LSTM model consisted of an embedding layer that
transformed words into a dense vector representation, followed
by LSTM layers that processed the text sequences. To predict
multiple future words, a RepeatVector layer was used, followed by another LSTM layer and a TimeDistributed dense
layer to output the sequence of predicted words.

C. Training Process
The LSTM model was trained to predict the next three
words in sequences derived from the plays, using categorical
cross-entropy as the loss function. Training included adjusting
the model’s learning rate and epoch count based on the
validation loss to prevent overfitting.
