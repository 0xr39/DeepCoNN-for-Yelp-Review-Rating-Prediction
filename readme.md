## DeepCoNN for Yelp Review Rating Prediction

An implementation of the model in [*Lei Zheng, Vahid Noroozi, and Philip S Yu. 2017. Joint deep modeling of users and items using reviews for recommendation*](https://arxiv.org/abs/1701.04783) 
on the [Yelp Review dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/data)

## Architecture

The model, in short, process the user review text data and the item review text data with 2 identical model only to combine them with a shared layer later. The text data is first processed by a word2Vec model to be turned into a vector, and later processed by a CNN network with a max-pooling layer to deal with the varying text length. After that, both the user review and item review is reconnnected by a shared layer to output the final prediction.

![architecture](https://github.com/ALEXdotR/DeepCoNN-for-Yelp-Review-Rating-Prediction/assets/72406898/1fa2c75e-05a4-4286-b9b4-c56313d25974)

*Zheng, Noroozi, et. al., 2017*

