# REASONER: An Explainable Recommendation Dataset with Multi-aspect Real User Labeled Ground Truths

[HomePage] | [Dataset] | [Library]

[HomePage]: https://reasoner2023.github.io/
[Dataset]: https://reasoner2023.github.io/docs/dataset
[Library]: https://reasoner2023.github.io/docs/library

<!-- [Paper]: https://arxiv.org/abs/2011.01731 -->

REASONER is an explainable recommendation dataset with multi-aspect real user labeled ground truths. The complete labeling process for each user is shown in following figure.
![generation](asset/steps.png)
In specific, we firstly develop a video recommendation platform, where a series of questions around the recommendation explainability are carefully designed. Then, we recruit about 3000 users with different backgrounds to use the system, and collect their behaviors and feedback to our questions.

## Data description

*REASONER* contains fifty thousand of user-item interactions as well as the side information including the video categories and user profile. Three files are included in the download data:

```plain
 REASONER
  ├── data
  │   ├── interaction.csv
  │   ├── user.csv
  │   ├── video.csv
```

### 1. Descriptions of the fields in `interaction.csv`

| Field Name:  | Description                                                                    | Type    | Example                                                                 |
| :----------- | :----------------------------------------------------------------------------- | :------ | :---------------------------------------------------------------------- |
| user_id      | The ID of the user.                                                            | int64   | 0                                                                       |
| video_id     | The ID of the viewed video.                                                    | int64   | 3650                                                                    |
| like         | Whether user like the video. 0 means no, 1 means yes                           | int64   | 0                                                                       |
| reason_tag   | Tags representing why the user likes/dislikes the video.                       | list    | [4728,2216,2523]                                                        |
| rating       | User rating for the video.                                                     | float64 | 3.0                                                                     |
| review       | User review for the video.                                                     | str     | This animation is very interesting, my friends and I like it very much. |
| video_tag    | Tags that reflect the content of the video.<br/>                               | list    | [2738,1216,2223]                                                        |
| interest_tag | Tags that reflect user interests.                                              | list    | [738,3226,1323]                                                         |
| watch_again  | Show only the interest tags, will the video be viewed. 0 means no, 1 means yes | int64   | 0                                                                       |

Note that if the user chooses to like the video, the `watch_again` item has no meaning and is set to 0.

### 2. Descriptions of the fields in `user.csv`

| Field Name: | Description                                                              | Type  | Example             |
| :---------- | :----------------------------------------------------------------------- | :---- | :------------------ |
| user_id     | The ID of the user.                                                      | int64 | 1005                |
| age         | User age.The mapping between id and content is shown below.              | int64 | 3                   |
| gender      | User gender. 0 means female, 1 menas male.                               | int64 | 0                   |
| education   | User education level. The mapping between id and content is shown below. | int64 | 3                   |
| career      | User occupation. The mapping between id and content is shown below.      | int64 | 20                  |
| income      | User income. The mapping between id and content is shown below.<br/>     | int64 | 3                   |
| address     | User income. The mapping between id and content is shown below.          | int64 | 23                  |
| hobby       | User hobby.                                                              | str   | drawing and soccer. |

The mappings between id and content are as below:

```plain
age={
0: "Under 15",
1: "15-20",
2: "20-25",
3: "25-30",
4: "30-35",
5: "35-40",
6: "40-45",
7: "45-50",
8: "Over 50"
},
education={
0: "Elementary School",
1: "Junior Middle School",
2: "Senior High School",
3: "Associate Degree",
4: "Bachelor's Degree",
5: "Master's Degree",
6: "Doctorate",
7: "Other"
},
career={
0: "Technology",
1: "Product",
2: "Design",
3: "Operations",
4: "Marketing",
5: "Human Resources/Finance/Administration",
6: "Senior Management",
7: "Sales",
8: "Media",
9: "Finance",
10: "Education and Training",
11: "Healthcare",
12: "Procurement/Trade",
13: "Supply Chain/Logistics",
14: "Real Estate/Construction",
15: "Agriculture/Forestry/Animal Husbandry/Fishing",
16: "Consulting/Translation/Law",
17: "Tourism",
18: "Service Industry",
19: "Manufacturing",
20: "Other"
},
income={
0: "0-5000",
1: "5000-10000",
2: "10000-15000",
3: "15000-20000",
4: "20000 and above"
}

```

### 3. Descriptions of the fields in `video.csv.`

| Field Name: | Description                                                                        | Type  | Example       |
| :---------- | :--------------------------------------------------------------------------------- | :---- | :------------ |
| video_id    | The ID of the video.                                                               | int64 | 1000          |
| title       | The title of the video.                                                            | str   | 18            |
| info        | The introduction of the video.                                                     | str   | 0             |
| tags        | The ID of the video tags.                                                          | list  | [112,33,1233] |
| duration    | The duration of the video in seconds.                                              | int64 | 120           |
| category    | The category of the video. The mapping between id and content is shown below.<br/> | int64 | 3             |

The mapping between categories and id is as follows:

```plain
category={
0: 'Music',
1: 'Gaming',
2: 'Comedy',
3: 'Lifestyle',
4: 'Movie & Montage',
5: 'Science & Technology',
6: 'Animation',
7: 'Other'
}
```

## Library

We developed a unified framework, which includes ten well-known explainable recommender models for rating prediction, top-k recommendation and review generation.

![图片](asset/structure.png)
The structure of our library is shown in the figure above. The configuration module is the base part of the library and responsible for initializing all the parameters. We support three methods to specify the parameters, that is, the commend line, parameter dictionary and configuration file. Based on the configuration module, there are four upper-layer modules:

- Data module. This module aims to convert the raw data into the model inputs. There are two components: the first one is responsible for loading the data and building vocabularies for the user reviews. The second part aims to process the data into the formats required by the model inputs, and generate the sample batches for model optimization.
- Model module. This module aims to implement the explainable recommender models. There are two types of methods in our library. The first one includes the feature-based explainable recommender models, and the second one contains the models with natural language explanations. We delay the detailed introduction of these models in the next section.
- Trainer module. This module is leveraged to implement the training losses, such as the Bayesian Personalized Ranking (BPR) and Binary Cross Entropy (BCE). In addition, this module can also record the complete model training process.
- Evaluation module. This module is designed to evaluate different models, and there are three types of evaluation tasks, that is, rating prediction, top-k recommendation and review generation. Upon the above four modules, there is an execution module to run different recommendation tasks.

### Requirements

```
python>=3.7.0
pytorch>=1.7.0
```

### Implemented Models

We implement several well-known explainable recommender models and list them according to category:

**Feature based models**:

* **[EFM](model/tag_aware_recommender/efm.py)** from Yongfeng Zhang *et al.*: [Explicit Factor Models for Explainable Recommendation based on Phrase-level Sentiment Analysis](https://www.cs.cmu.edu/~glai1/papers/yongfeng-guokun-sigir14.pdf) (SIGIR 2014).

* **[TriRank](model/tag_aware_recommender/trirank.py)** from Xiangnan He *et al.*: [TriRank: Review-aware Explainable Recommendation by Modeling Aspects](https://wing.comp.nus.edu.sg/wp-content/uploads/Publications/PDF/TriRank-%20Review-aware%20Explainable%20Recommendation%20by%20Modeling%20Aspects.pdf) (CIKM 2015).

* **[LRPPM](model/tag_aware_recommender/lrppm.py)** from Xu Chen *et al.*: [Learning to Rank Features for Recommendation over Multiple Categories](http://yongfeng.me/attach/sigir16-chen.pdf) (SIGIR 2016).

* **[SULM](model/tag_aware_recommender/sulm.py)** from Konstantin Bauman *et al.*: [Aspect Based Recommendations: Recommending Items with the Most Valuable Aspects Based on User Reviews.](https://www.researchgate.net/profile/Konstantin-Bauman/publication/318915371_Aspect_Based_Recommendations_Recommending_Items_with_the_Most_Valuable_Aspects_Based_on_User_Reviews/links/5f06007e92851c52d620bc9f/Aspect-Based-Recommendations-Recommending-Items-with-the-Most-Valuable-Aspects-Based-on-User-Reviews.pdf) (KDD 2017).

* **[MTER](model/tag_aware_recommender/mter.py)** from Nan Wang *et al.*: [Explainable Recommendation via Multi-Task Learning in Opinionated Text Data](https://dl.acm.org/doi/pdf/10.1145/3209978.3210010) (SIGIR 2018).

* **[AMF](model/tag_aware_recommender/amf.py)** from Yunfeng Hou *et al.*: [Explainable recommendation with fusion of aspect information](https://yneversky.github.io/Papers/Hou2019_Article_ExplainableRecommendationWithF.pdf) (WWW 2019).

* In addition to the above shallow models based on matrix factorization, we also implement the following deep feature-based explainable recommender models (called DERM for short).

**Natural Language based models**:

* **[Att2Seq](model/review_aware_recommender/att2seq.py)** from Li Dong *et al.*: [Learning to Generate Product Reviews from Attributes](https://aclanthology.org/E17-1059.pdf) (ACL 2017).

* **[NRT](model/review_aware_recommender/nrt.py)** from Piji Li *et al.*: [Neural Rating Regression with Abstractive Tips Generation for Recommendation](https://arxiv.org/pdf/1708.00154.pdf) (SIGIR 2017).

* **[PETER](model/review_aware_recommender/peter.py)** from Lei Li *et al.*: [Personalized Transformer for Explainable Recommendation](https://arxiv.org/pdf/2105.11601.pdf) (ACL 2021).

### Quick start

Here is a quick-start example for our library. You can directly execute _tag_prediction.py_ or _review_generate.py_ to run a feature-based or review-based model, respectively. In each of these commends, you need to specify three parameters to indicate the names of the model, dataset and configuration file, respectively.

Run feature based models:
```bash
python tag_prediction.py --model=[model name] --dataset=[dataset] --config=[config_files]
```

Run natural language based models:

```bash
python tag_prediction.py --model=[model name] --dataset=[dataset] --config=[config_files]
```


## How to Obtain?

Please provide us with your basic information including your name, institution, and purpose of use to request the dataset and library. You can email us at reasonerdataset@gmail.com.

## Acknowledgement

Please cite the following paper as the reference if you use our code or dataset.

```
@inproceedings{,
  title={},
  author={},
  booktitle={},
  year={}
}


