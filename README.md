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

We developed a unified framework, which includes ten well-known explainable recommender models for rating prediction, tag prediction and review generation.

![图片](asset/structure.png)
The structure of our library is shown in the figure above. The configuration module is the base part of the library and responsible for initializing all the parameters. We support three methods to specify the parameters, that is, the commend line, parameter dictionary and configuration file. Based on the configuration module, there are four upper-layer modules:

- Data module. This module aims to convert the raw data into the model inputs. There are two components: the first one is responsible for loading the data and building vocabularies for the user reviews. The second part aims to process the data into the formats required by the model inputs, and generate the sample batches for model optimization.
- Model module. This module aims to implement the explainable recommender models. There are two types of methods in our library. The first one includes the feature-based explainable recommender models, and the second one contains the models with natural language explanations. We delay the detailed introduction of these models in the next section.
- Trainer module. This module is leveraged to implement the training losses, such as the Bayesian Personalized Ranking (BPR) and Binary Cross Entropy (BCE). In addition, this module can also record the complete model training process.
- Evaluation module. This module is designed to evaluate different models, and there are three types of evaluation tasks, that is, rating prediction, top-k recommendation and review generation. Upon the above four modules, there is an execution module to run different recommendation tasks.

### Quick start

Here is a quick-start example for our library. You can directly execute _tag_prediction.py_ or _review_generate.py_ to run a feature-based or review-based model, respectively. In each of these commends, you need to specify three parameters to indicate the names of the model, dataset and configuration file, respectively.

```plain
python tag_prediction.py --model=[model] --datatset=[dataset] --config=[config_files]
python review_prediction.py --model=[model] --datatset=[dataset] --config=[config_files]
```

## How to Obtain?

Please provide us with your basic information including your name, institution, and purpose of use to request the dataset and library. You can email us at reasonerdataset@gmail.com.
