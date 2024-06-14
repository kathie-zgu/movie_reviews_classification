# Movie Reviews Classification (NLP) Project Summary
This project focuses on classifying movie reviews as positive or negative using the IMDB dataset. We employ deep learning algorithms to predict the sentiment of these reviews using three different models with various hyperparameters:

- SimpleNeuralNet (custom model)
- DeepNeuralNet (custom model)
- DistilBERT (fine-tuned distilbert-base-uncased)

The steps involved in this project are outlined below:

## Project Description
### Step 1: Setting Up the Environment
We installed the necessary libraries for natural language processing and deep learning, including transformers, datasets, nltk, evaluate, accelerate, and wandb.

### Step 2: Data Downloading and Preprocessing
Downloading the Dataset: We used the datasets library to download the IMDB dataset, which consists of 50,000 movie reviews.

Preprocessing: We expanded contractions, removed HTML tags, handled negations, tokenized the text, removed stop words, and lemmatized the words. This preprocessing was essential to clean the text data before feeding it into the models.
### Step 3: Tokenization and Embeddings
Tokenization: We tokenized the text data for the DistilBERT model using the distilbert-base-uncased tokenizer.

Word Embeddings: For SimpleNN and DeepNN models, we created word embeddings using the Word2Vec model.
### Step 4: Dataset and DataLoader Creation
We created custom datasets and data loaders for training and testing the models:

IMDBDataset Class: This class converts text sequences into embeddings and pairs them with their labels.

Data Loaders: We set up data loaders to facilitate batch processing during training and testing.
### Step 5: Model Definition and Training
We defined three models for this task:

SimpleNeuralNet: A basic neural network model with a single fully connected layer.

DeepNeuralNet: A more complex neural network with multiple layers and convolutional operations.

DistilBERT: A pre-trained transformer model fine-tuned for the sentiment classification task.
### Step 6: Training and Evaluation
Training Loop: We implemented a training loop with early stopping to prevent overfitting.

Evaluation: We evaluated the models using accuracy as the primary metric.
### Step 7: Hyperparameter Tuning and Model Comparison
We experimented with different hyperparameters, such as learning rates and model configurations, to identify the best-performing model. The performance of each model was logged and visualized using Weights & Biases (WandB).

## Conclusion
This project demonstrated the application of various deep learning techniques to the task of sentiment analysis on movie reviews. By comparing different models and hyperparameters, we were able to determine the most effective approach for classifying movie reviews as positive or negative.

The results indicate that pre-trained transformer models like DistilBERT significantly outperform simpler neural network architectures in this task:

- Fine-tuned DistilBERT showed its best accuracy at third epoch = 0.9050
- SimpleNN showed its best accuracy at sixth epoch = 0.7957
- DeepNN showed its best accuracy at tenth epoch = 0.8522

For detailed results and further analysis, refer to the [WandB report]( https://api.wandb.ai/links/e-v-zgurskaya/u4vehuno)
