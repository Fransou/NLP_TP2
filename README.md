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
        classifier.py		# define the classifier
        config.py		# define global variables and device
        create_dataset.py	# create dataloader for training and testing
        model.py		# define the model's architecture
        preprocessing.py	# preprocess data
        tester.py		# run the code and evaluate model
```

### Data Preprocessing

Our data features will be the concatenation of

- Tokenized sentences thanks to `BertTokenizer` from pretrained `bert-base-cased`, with a maximum sentence length of 64 caracters. This length was chosen by test-and-trial. We end up with 3 vectors : `input_ids`, `attention_mask` and ``aspect_id``.
- A vector corresponding to a mask of the tokenized aspect term.

Our labels will be a matrix of shape $12 \times 3$, corresponding to the 12 aspect categories and the 3 polarities. It is a sparse vector, with 1 in the aspect category index with the corresponding polarity. By doing this, even if a sentence contains sentiments about multiple aspects, we do only one prediction per aspect.

### Model architecture

We use `BertModel` from pretrained `bert-base-cased` and we concatenate 6 fully connected layers of shape $1024\times1024$ with ReLU activation function, each one separated by a Drop Out layer with probabilities decreasing from $0.5$ to $0.3$. We also use two-by-two skipping layers. The last layer is a fully connected layer of size $12\times3$ with softmax.

We use Drop Out because we noticed some overfitting, and skip connection in order for the model to learn.

### Training

We set our learning rate to $10^{-5}$, otherwise the model converges within one epoch and stops learning. We put $ 100$ epochs, but the model is learning slower and slower, so we could have stopped the training earlier.

### Evaluation and results

We reach an accuracy of $87\%$ on the training set, and $84\%$ on the test set. We see that there are very few neutral polarity predicted.
