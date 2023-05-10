---
title: Dataset
---

## Introduction

**REASONER** is an explainable recommendation dataset. It contains the ground truths for multiple explanation purposes, for example, enhancing the recommendation persuasiveness, informativeness and satisfaction. In this dataset, the ground truth annotators are exactly the people who produce the user-item interactions, and they can make selections from the explanation candidates with multi-modalities. This dataset can be widely used for explainable recommendation, unbiased recommendation, psychology-informed recommendation and so on. Please see our paper for more details.

The dataset contains the following files.

```plain
 REASONER-Dataset
  │── dataset
  │   ├── interaction.csv
  │   ├── user.csv
  │   ├── video.csv
  │   ├── bigfive.csv 
  │   ├── tag_map.csv 
  │   ├── video_map.csv 
  │── preview
  │── README.md
```

## How to Obtain the Dataset


You can directly download the REASONER dataset through the following three links:

- [![Google Drive](https://img.shields.io/badge/-Google%20Drive-yellow)](https://drive.google.com/drive/folders/1dARhorIUu-ajc5ZsWiG_XY36slRX_wgL?usp=share_link)

- [![Baidu Netdisk](https://img.shields.io/badge/-Baidu%20Netdisk-lightgrey)](https://pan.baidu.com/s/1L9AzPe0MkRbMwk6yeDj4QA?pwd=ipxd)

- [![OneDrive](https://img.shields.io/badge/-OneDrive-blue)]([REASONER-Dataset](https://1drv.ms/f/s!AiuzqR3lP02KbCZOY3c8bfb3ZWg?e=jWTuc1))

## Data description

### 1. interaction.csv

This file contains the user's annotation records on the video, including the following fields:

| Field Name:         | Description                                                  | Type    | Example                                                      |
| :------------------ | :----------------------------------------------------------- | :------ | :----------------------------------------------------------- |
| user_id             | ID of the user                                               | int64   | 0                                                            |
| video_id            | ID of the viewed video                                       | int64   | 3650                                                         |
| like                | Whether user like the video: 0 means no, 1 means yes         | int64   | 0                                                            |
| persuasiveness_tag  | The user selected tags for the question "Which tags are the reasons that you would like to watch this video?" before watching the video | list    | [4728,2216,2523]                                             |
| rating              | User rating for the video, the range is 1.0~5.0              | float64 | 3.0                                                          |
| review              | User review for the video                                    | str     | This animation is very interesting, my friends and I like it very much. |
| informativeness_tag | The user selected tags for the question "Which features are most informative for this video?" after watching the video | list    | [2738,1216,2223]                                             |
| satisfaction_tag    | The user selected tags for the question "Which features are you most satisfied with?" after watching the video. | list    | [738,3226,1323]                                              |
| watch_again         | If the system only show the satisfaction_tag to the user, whether the she would like to watch this video? 0 means no, 1 means yes | int64   | 0                                                            |

Note that if the user chooses to like the video, the `watch_again` item has no meaning and is set to 0.

### 2. user.csv

This file contains user profiles.

| Field Name: | Description                               | Type  | Example             |
| :---------- | :---------------------------------------- | :---- | :------------------ |
| user_id     | ID of the user                            | int64 | 1005                |
| age         | User age (indicated by ID)                | int64 | 3                   |
| gender      | User gender: 0 means female, 1 means male | int64 | 0                   |
| education   | User education level (indicated by ID)    | int64 | 3                   |
| career      | User occupation (indicated by ID)         | int64 | 20                  |
| income      | User income (indicated by ID)             | int64 | 3                   |
| address     | User address (indicated by ID)            | int64 | 23                  |
| hobby       | User hobbies                              | str   | drawing and soccer. |

### 3. video.csv

This file contains information of videos.

| Field Name: | Description                             | Type  | Example                                   |
| :---------- | :-------------------------------------- | :---- | :---------------------------------------- |
| video_id    | ID of the video                         | int64 | 1                                         |
| title       | Title of the video                      | str   | Take it once a day to prevent depression. |
| info        | Introduction of the video               | str   | Just like it, once a day                  |
| tags        | ID of the video tags                    | list  | [112,33,1233]                             |
| duration    | Duration of the video in seconds        | int64 | 120                                       |
| category    | Category of the video (indicated by ID) | int64 | 3                                         |

### 4. bigfive.csv

We have the annotators take the [Big Five Personality Test](https://www.psytoolkit.org/survey-library/big5-bfi-s.html), and `bigfive.csv` contains the answers of the annotators to 15 questions, where [0, 1, 2, 3, 4, 5] correspond to [strongly disagree, disagree, somewhat disagree, somewhat agree, agree, strongly agree]. This file also includes a  `user_id` column.

The questions are described as follows:

| Question | Description                                                  |
| :------- | :----------------------------------------------------------- |
| Q1       | I think most people are basically well-intentioned           |
| Q2       | I get bored with crowded parties                             |
| Q3       | I'm a person who takes risks and breaks the rules            |
| Q4       | i like adventure                                             |
| Q5       | I try to avoid crowded parties and noisy environments        |
| Q6       | I like to plan things out at the beginning                   |
| Q7       | I worry about things that don't matter                       |
| Q8       | I work or study hard                                         |
| Q9       | Although there are some liars in the society, I think most people are still credible |
| Q10      | I have a spirit of adventure that no one else has            |
| Q11      | I often feel uneasy                                          |
| Q12      | I'm always worried that something bad is going to happen     |
| Q13      | Although there are some dark things in human society (such as war, crime, fraud), I still believe that human nature is generally good |
| Q14      | I enjoy going to social and entertainment gatherings         |
| Q15      | It is one of my characteristics to pay attention to logic and order in doing things |

### 5. tag_map.csv

Mapping relationship between the tag ID and the tag content. We add 7 additional tags that all videos contain, namely "preview 1, preview 2, preview 3, preview 4, preview 5, title, content".

| Field Name:         | Description                                                  | Type    | Example                                                      |
| :------------------ | :----------------------------------------------------------- | :------ | :----------------------------------------------------------- |
| tag_id              | ID of the tag                                                | int64   | 1409                                                         |
| tag_content         | The content corresponding to the tag                         | str     | cute baby                                                    |

### 6. video_map.csv

Mapping relationship between the video ID and the folder name in `preview`.

| Field Name:         | Description                                                  | Type    | Example                                                      |
| :------------------ | :----------------------------------------------------------- | :------ | :----------------------------------------------------------- |
| video_id            | ID of the video                                              | int64   | 1                                                            |
| folder_name         | The folder name corresponding to the video                   | str     | 83062078                                                     |

### 7. preview

Each video contains 5 image previews.

The mapping relationship between the folder name and the video ID is in `video_map.csv`.


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

## Codes for accessing our data

We provide code to read the data into data frame with *pandas*. 

```python
import pandas as pd

# access interaction.csv
interaction_df = pd.read_csv('interaction.csv', sep='\t', header=0)
# get the first ten lines
print(interaction_df.head(10))
# get each column 
# ['user_id', 'video_id', 'like', 'persuasiveness_tag', 'rating', 'review', 'informativeness_tag', 'satisfaction_tag', 'watch_again', ]
for col in interaction_df.columns:
	print(interaction_df[col][:10])

# access user.csv
user_df = pd.read_csv('user.csv', sep='\t', header=0)
print(user_df.head(10))
# ['user_id', 'age', 'gender', 'education', 'career', 'income', 'address', 'hobby']
for col in user_df.columns:
	print(user_df[col][:10])
  
# access video.csv
video_df = pd.read_csv('video.csv', sep='\t', header=0)
print(video_df.head(10))
# ['video_id', 'title', 'info', 'tags', 'duration', 'category']
for col in video_df.columns:
	print(video_df[col][:10])

# access bigfive.csv
bigfive_df = pd.read_csv('bigfive.csv', sep='\t', header=0)
print(bigfive_df.head(10))
# ['user_id', 'Q1', ..., 'Q15']
for col in bigfive_df.columns:
	print(bigfive_df[col][:10])

# access tag_map.csv
tag_map_df = pd.read_csv('tag_map.csv', sep='\t', header=0)
print(tag_map_df.head(10))
# ['tag_id', 'tag_content']
for col in tag_map_df.columns:
	print(tag_map_df[col][:10])
  
# access video_map.csv
video_map_df = pd.read_csv('video_map.csv', sep='\t', header=0)
print(video_map_df.head(10))
# ['video_id', 'folder_name']
for col in video_map_df.columns:
	print(video_map_df[col][:10])
```

