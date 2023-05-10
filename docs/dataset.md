---
title: Dataset
---

## Introduction

REASONER is an explainable recommendation dataset. It contains the ground truths for multiple explanation purposes, for example, enhancing the recommendation persuasiveness, informativeness and so on. In this dataset, the ground truth annotators are exactly the people who produce the user-item interactions, and they can make selections from the explanation candidates with multi-modalities. This dataset can be widely used for explainable recommendation, unbiased recommendation, psychology-informed recommendation and so on. Please see our paper for details.  

## How to Obtain the Dataset

You can obtain all the data in the REASONER from [Google Drive](https://drive.google.com/drive/folders/1dARhorIUu-ajc5ZsWiG_XY36slRX_wgL?usp=sharing).

## Data description

*REASONER* contains fifty thousand of user-item interactions as well as the side information including the video categories and user profile. Three files are included in the dataset:

```plain
 REASONER
  ├── data
  │   ├── interaction.csv
  │   ├── user.csv
  │   ├── video.csv
  │   ├── bigfive.csv 
```

### 1. Descriptions of the fields in `interaction.csv`

| Field Name:  | Description                                                                    | Type    | Example                                                                 |
| :----------- | :----------------------------------------------------------------------------- | :------ | :---------------------------------------------------------------------- |
| user_id      | ID of the user                                                                | int64   | 0                                                                       |
| video_id     | ID of the viewed video                                                        | int64   | 3650                                                                    |
| like         | Whether user like the video: 0 means no, 1 means yes                           | int64   | 0                                                                       |
| persuasiveness_tag   |The user selected tags for the question "Which tags are the reasons that you would like to watch this video?" before watching the video                        | list    | [4728,2216,2523]                                                        |
| rating       | User rating for the video                                                     | float64 | 3.0                                                                     |
| review       | User review for the video                                                     | str     | This animation is very interesting, my friends and I like it very much. |
| informativeness_tag    | The user selected tags for the question "Which features are most informative for this video?" after watching the video                             | list    | [2738,1216,2223]                                                        |
| satisfaction_tag | The user selected tags for the question "Which features are you most satisfied with?" after watching the video.                                              | list    | [738,3226,1323]                                                         |
| watch_again  |  If the system only show the satisfaction_tag to the user, whether the she would like to watch this video? | int64   | 0                                                                       |

Note that if the user chooses to like the video, the `watch_again` item has no meaning and is set to 0.

### 2. Descriptions of the fields in `user.csv`

| Field Name: | Description                                | Type  | Example             |
| :---------- | :----------------------------------------- | :---- | :------------------ |
| user_id     | ID of the user                            | int64 | 1005                |
| age         | User age (indicated by ID)                | int64 | 3                   |
| gender      | User gender: 0 means female, 1 means male | int64 | 0                   |
| education   | User education level (indicated by ID)    | int64 | 3                   |
| career      | User occupation (indicated by ID)         | int64 | 20                  |
| income      | User income (indicated by ID)             | int64 | 3                   |
| address     | User address (indicated by ID)            | int64 | 23                  |
| hobby       | User hobbies                              | str   | drawing and soccer. |

### 3. Descriptions of the fields in `video.csv`

| Field Name: | Description                              | Type  | Example                                   |
| :---------- | :--------------------------------------- | :---- | :---------------------------------------- |
| video_id    | ID of the video                         | int64 | 1                                         |
| title       | Title of the video                      | str   | Take it once a day to prevent depression. |
| info        | Introduction of the video               | str   | Just like it, once a day                  |
| tags        | ID of the video tags                    | list  | [112,33,1233]                             |
| duration    | Duration of the video in seconds        | int64 | 120                                       |
| category    | Category of the video (indicated by ID) | int64 | 3                                         |

### 4. Descriptions of the fields in `bigfive.csv`

We have the annotators take the [Big Five Personality Test](https://www.psytoolkit.org/survey-library/big5-bfi-s.html), and `bigfive.csv` contains the answers of the annotators to 15 questions, where [0, 1, 2, 3, 4, 5] correspond to [strongly disagree, disagree, somewhat disagree, somewhat agree, agree, strongly agree]. The file also includes a user_id column.

## Statistics

### 1. The basic statistics of REASONER

We have collected the basic information of the REASONER dataset and listed it in the table below. "u-v" represents the number of interactions between users and videos, "u-t" represents the number of tags clicked by users, and "Q1, Q2, Q3" respectively represent the persuasiveness, informativeness, and satisfaction of the tags.

| #User | #Video | #Tag  | #u-v   | #u-t (Q1) | #u-t (Q2) | #u-t (Q3) |
| ----- | ------ | ----- | ------ | --------- | --------- | --------- |
| 2,997 | 4,672  | 6,115 | 58,497 | 263,885   | 271,456   | 256,079   |

### 2. Statistics on the users

<div style={{textAlign: 'center'}}>
<img
src={require('../static/img/dataset/user.png').default}
style={{width: '80%'}}
/>
</div>

### 3. Statistics on the videos

<div style={{textAlign: 'center'}}>
<img
src={require('../static/img/dataset/video.png').default}
style={{width: '80%'}}
/>
</div>

## Quick View

We provide two ways to quickly view the data content and take the first ten lines of *interaction.csv* for example.

### 1. Command Line

````
cd dataset
head -n 10 interaction.csv 
````

### 2. Pandas

````python
import pandas as pd

df = pd.read_csv('interaction.csv', sep='\t')
print(df.head(10))
````
