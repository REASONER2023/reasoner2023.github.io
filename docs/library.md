---
title: Library
---

## Introduction

Along with REASONER, we develop an explainable recommendation library. This library provides two types of widely studied explainable recommender models. The first one are feature based explainable recommender models, and the second one are the models with natural language explanations.

## How to Obtain the Library

You can access our library through [GitHub](https://github.com/REASONER2023/reasoner2023.github.io/tree/main).

## Model

### Tag-aware models

|  Model   |                                                                                                                        Description                                                                                                                         |                                                                  Reference                                                                  | Year  |
| :------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------: | :---: |
|   EFM    |                                               EFM predicts the user preferences and generates explainable recommendations based on explicit product features and user opinions from the review information.                                                |     [Explicit Factor Models for Explainable Recommendation based on Phrase-level Sentiment Analysis](https://www.cs.cmu.edu/~glai1/papers/yongfeng-guokun-sigir14.pdf)  (Yongfeng Zhang et al.,  SIGIR2014)     | 2014  |
| TriRank  |                                 TriRank models the user-item-aspect ternary relation as a heterogeneous tripartite graph based on user ratings and reviews, and it devises a vertex ranking algorithm for recommendation.                                  |                     [TriRank: Review-aware Explainable Recommendation by Modeling Aspects](https://wing.comp.nus.edu.sg/wp-content/uploads/Publications/PDF/TriRank-%20Review-aware%20Explainable%20Recommendation%20by%20Modeling%20Aspects.pdf) (Xiangnan He et al., CIKM2015)                     | 2015  |
|  LRPPM   |                                                     A tensor-matrix factorization algorithm which captures the user preferences using ranking-based optimization objective over various item aspects.                                                      |                     [Learning to Rank Features for Recommendation over Multiple Categories](http://yongfeng.me/attach/sigir16-chen.pdf) (Xu Chen et al., SIGIR 2016)                      | 2016  |
|   SULM   |                                                            SULM enhances recommendations by recommending not only item but also the specific aspects by using aspect-level sentiment analysis.                                                             | [Aspect Based Recommendations: Recommending Items with the Most Valuable Aspects Based on User Reviews](https://www.researchgate.net/profile/Konstantin-Bauman/publication/318915371_Aspect_Based_Recommendations_Recommending_Items_with_the_Most_Valuable_Aspects_Based_on_User_Reviews/links/5f06007e92851c52d620bc9f/Aspect-Based-Recommendations-Recommending-Items-with-the-Most-Valuable-Aspects-Based-on-User-Reviews.pdf)  (Konstantin Bauman et al., KDD 2017) | 2017  |
|   MTER   |         MTER is a tensor factorization method which models the task of item recommendation using a three-way tensor over the users, items and features. We omit the modeling of the opinions in the original implementation for adapting our data.         |                  [Explainable Recommendation via Multi-Task Learning in Opinionated Text Data](https://dl.acm.org/doi/pdf/10.1145/3209978.3210010)  (Nan Wang et al.,  SIGIR2018)                  | 2018  |
|   AMF    |                                                                    AMF improves the recommendation accuracy by using the auxiliary information extracted from the user review aspects.                                                                     |                         [Explainable recommendation with fusion of aspect information](https://yneversky.github.io/Papers/Hou2019_Article_ExplainableRecommendationWithF.pdf) (Yunfeng Hou et al., WWW2019)                          | 2019  |
| DERM-MLP | DERM is a deep recommender model for jointly predicting the ratings and tags. The two tasks share the set of user/item/tag embeddings. The hidden states as well as the tag embeddings are put into different layers corresponding to the different tasks. |                                                                      -                                                                      |   -   |
| DERM-MF  |                                                DERM-MF firstly obtains a hidden state based on the user/item embeddings using matrix factorization, and then the outputs are computed by a neural network.                                                 |                                                                      -                                                                      |   -   |
|  DERM-C  |                                                  DERM-C combines matrix factorization and Multi-Layer Perceptron (MLP) to derive the hidden states, and the outputs are merged in a concatenated manner.                                                   |                                                                      -                                                                      |   -   |
|  DERM-H  |                                                       DERM-H leverages the tags to profile the users and items, and then use the same architecture as DERM-MLP for predicting the ratings and tags.                                                        |                                                                      -                                                                      |   -   |

### Review-aware models

|  Model  |                                                                      Description                                                                       |                                                Reference                                                 | Year  |
| :-----: | :----------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------: | :---: |
| Att2Seq |       A review generation model which uses LSTM as the decoder, and output the texts directly based on the user/item IDs and rating information.       |              [Learning to Generate Product Reviews from Attributes](https://aclanthology.org/E17-1059.pdf) (Li Dong et al. ACL2017)               | 2017  |
|   NRT   | NRT simultaneously predicts the reviews and ratings based on the input user-item pair, where the two tasks share the same embedding and hidden layers. | [Neural Rating Regression with Abstractive Tips Generation for Recommendation](https://arxiv.org/pdf/1708.00154.pdf) (Piji Li et al, SIGIR2017) | 2017  |
|  PETER  |                     PETER leverages Transformer to generate the user reviews, which is a state-of-the-art review generation model.                     |             [Personalized Transformer for Explainable Recommendation](https://arxiv.org/pdf/2105.11601.pdf) (Lei Li et al. ACL2021)              | 2021  |



## Framework

<div align=center>
<img
src={require('../static/img/library/structure.png').default}
style={{width: '80%'}}
/> 
</div>

The structure of our library is shown in the figure above. The configuration module is the base part of the library and responsible for initializing all the parameters. We support three methods to specify the parameters, that is, the command line, parameter dictionary and configuration file. Based on the configuration module, there are four upper-layer modules:

- **Data module.** This module aims to convert the raw data into the model inputs. There are two components: the first one is responsible for loading the data and building vocabularies for the user reviews. The second part aims to process the data into the formats required by the model inputs, and generate the sample batches for model optimization.
- **Model module.** This module aims to implement the explainable recommender models. There are two types of methods in our library. The first one includes the feature-based explainable recommender models, and the second one contains the models with natural language explanations. We delay the detailed introduction of these models in the next section.
- **Trainer module.** This module is leveraged to implement the training losses, such as the Bayesian Personalized Ranking (BPR) and Binary Cross Entropy (BCE). In addition, this module can also record the complete model training process.
- **Evaluation module.** This module is designed to evaluate different models, and there are three types of evaluation tasks, that is, rating prediction, top-k recommendation and review generation. Upon the above four modules, there is an execution module to run different recommendation tasks.

## 4、Quick start

Here is a quick-start example for our library. You can directly execute *tag_prediction.py* or *review_generate.py* to run a feature-based or review-based model, respectively. In each of these commends, you need to specify three parameters to indicate the names of the model, dataset and configuration file, respectively.

Run feature based models:

```bash
python tag_prediction.py --model=[model name] --dataset=[dataset] --config=[config_files]
```

Run natural language based models:

```bash
python tag_prediction.py --model=[model name] --dataset=[dataset] --config=[config_files]
```