# NLP Exercise 2: Aspect-Based Sentiment Analysis

## Team members

Pauline BERBERI, Philippe FORMONT, Rizwan NATO

## Resolution

### Project Structure

Here is the structure of our code.

```
│   README.md
│   
├───data
│       devdata.csv
│       traindata.csv
│   
├───resources  
└───src
        classifier.py           # define the classifier
        config.py               # define global variables and device
        create_dataset.py       # create dataloader for training and testing
        model.py                # define the model's architecture
        preprocessing.py        # preprocess data
        tester.py               # run the code and evaluate model
```

### Data Preprocessing

Our data features will be the concatenation of

- Tokenized sentences thanks to `BertTokenizer` from pretrained `bert-base-cased`, with a maximum sentence length of 64 caracters. This length was chosen by test-and-trial. We end up with 3 vectors : `input_ids`, `attention_mask` and ``aspect_id`` (telling which aspect are we evaluating for a given sentece).
- A vector corresponding to a mask of the tokenized aspect term.

Our model will output 3 scores corresponding to the probability of the Text being 'positive', 'negative', or 'neutral' for a given aspect.

### Model architecture

We use `BertModel` from pretrained `bert-base-cased` and we concatenate 6 fully connected layers of shape $1024\times1024$ with ReLU activation function, each one separated by a Drop Out layer with probabilities decreasing from $0.5$ to $0.3$. We also use two-by-two skipping layers. 

The last layer is a fully connected layer of size $12\times3$, which gives the model's polarity predictions for every aspect. We then return the polarity predictions for the aspect we are looking for. The crossentropy loss is computed only on those three scores (polarity score for our aspect). 

We tried to perform data augmentation by adding for each text a neutral polarity for aspects that are not labeled for this text (to compute the loss on more neurons in the last layer for each text) but this did'nt improve the results.

We use Drop Out because we noticed some overfitting, and skip connection (the intuition beeing that it would facilitate the use of deep models, keeping a gradient not too low to update the weights of the BERT model).

### Training

We set our learning rate to $10^{-5}$, otherwise the model converges within one epoch and stops learning. We put $50$ epochs, but the model is learning slower and slower, so we could have stopped the training earlier. We also used a linear learning rate schedule.

### Evaluation and results

We reach an accuracy of $90\%$ on the training set, and $83\%$ on the test set. We see that there are very few neutral polarity predicted.
