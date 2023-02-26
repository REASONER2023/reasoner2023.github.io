---
title: Dataset
---

## Introduction

REASONER is an explainable recommendation dataset. It contains the ground truths for multiple explanation purposes, for example, enhancing the recommendation persuasiveness, informativeness and so on. In this dataset, the ground truth annotators are exactly the people who produce the user-item interactions, and they can make selections from the explanation candidates with multi-modalities. This dataset can be widely used for explainable recommendation, unbiased recommendation, psychology-informed recommendation and so on.


## How to Obtain the Dataset

Please provide us with your basic information including your name, institution, and purpose of use to request the dataset. You can email us at reasonerdataset@gmail.com.

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
| user_id      | ID of the user.                                                                | int64   | 0                                                                       |
| video_id     | ID of the viewed video.                                                        | int64   | 3650                                                                    |
| like         | Whether user like the video. 0 means no, 1 means yes                           | int64   | 0                                                                       |
| reason_tag   | Tags that reflect why the user likes/dislikes the video.                       | list    | [4728,2216,2523]                                                        |
| rating       | User rating for the video.                                                     | float64 | 3.0                                                                     |
| review       | User review for the video.                                                     | str     | This animation is very interesting, my friends and I like it very much. |
| video_tag    | Tags that reflect the content of the video.<br/>                               | list    | [2738,1216,2223]                                                        |
| interest_tag | Tags that reflect user interests.                                              | list    | [738,3226,1323]                                                         |
| watch_again  | Show only the interest tags, will the video be viewed. 0 means no, 1 means yes | int64   | 0                                                                       |

Note that if the user chooses to like the video, the `watch_again` item has no meaning and is set to 0.

### 2. Descriptions of the fields in `user.csv`

| Field Name: | Description                                | Type  | Example             |
| :---------- | :----------------------------------------- | :---- | :------------------ |
| user_id     | ID of the user.                            | int64 | 1005                |
| age         | User age (indicated by ID).                | int64 | 3                   |
| gender      | User gender. 0 means female, 1 menas male. | int64 | 0                   |
| education   | User education level (indicated by ID).    | int64 | 3                   |
| career      | User occupation (indicated by ID).         | int64 | 20                  |
| income      | User income (indicated by ID).             | int64 | 3                   |
| address     | User address (indicated by ID).            | int64 | 23                  |
| hobby       | User hobbies.                              | str   | drawing and soccer. |

### 3. Descriptions of the fields in `video.csv.`

| Field Name: | Description                              | Type  | Example                                   |
| :---------- | :--------------------------------------- | :---- | :---------------------------------------- |
| video_id    | ID of the video.                         | int64 | 1                                         |
| title       | Title of the video.                      | str   | Take it once a day to prevent depression. |
| info        | Introduction of the video.               | str   | Just like it, once a day                  |
| tags        | ID of the video tags.                    | list  | [112,33,1233]                             |
| duration    | Duration of the video in seconds.        | int64 | 120                                       |
| category    | Category of the video (indicated by ID). | int64 | 3                                         |


## Statistics

### 1. The basic statistics of REASONER.
We have collected the basic information of the REASONER dataset and listed it in the table below. "u-v" represents the number of interactions between users and videos, "u-t" represents the number of tags clicked by users, and "Q1, Q2, Q3" respectively represent the persuasiveness, informativeness, and satisfaction of the tags.
| #User | #Video | #Tag  | #u-v   | #u-t (Q1) | #u-t (Q2) | #u-t (Q3) |
| ----- | ------ | ----- | ------ | --------- | --------- | --------- |
| 2,997 | 4,672  | 6,115 | 58,497 | 263,885   | 271,456   | 256,079   |

### 2. Statistics on the users

<img
src={require('../static/img/dataset/user.png').default}
style={{width: '80%'}}
/>

### 3. Statistics on the videos

<img
src={require('../static/img/dataset/video.png').default}
style={{width: '80%'}}
/>

