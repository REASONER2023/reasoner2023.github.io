"use strict";(self.webpackChunkreasoner=self.webpackChunkreasoner||[]).push([[598],{3905:(t,e,a)=>{a.d(e,{Zo:()=>p,kt:()=>k});var n=a(7294);function r(t,e,a){return e in t?Object.defineProperty(t,e,{value:a,enumerable:!0,configurable:!0,writable:!0}):t[e]=a,t}function i(t,e){var a=Object.keys(t);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(t);e&&(n=n.filter((function(e){return Object.getOwnPropertyDescriptor(t,e).enumerable}))),a.push.apply(a,n)}return a}function l(t){for(var e=1;e<arguments.length;e++){var a=null!=arguments[e]?arguments[e]:{};e%2?i(Object(a),!0).forEach((function(e){r(t,e,a[e])})):Object.getOwnPropertyDescriptors?Object.defineProperties(t,Object.getOwnPropertyDescriptors(a)):i(Object(a)).forEach((function(e){Object.defineProperty(t,e,Object.getOwnPropertyDescriptor(a,e))}))}return t}function d(t,e){if(null==t)return{};var a,n,r=function(t,e){if(null==t)return{};var a,n,r={},i=Object.keys(t);for(n=0;n<i.length;n++)a=i[n],e.indexOf(a)>=0||(r[a]=t[a]);return r}(t,e);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(t);for(n=0;n<i.length;n++)a=i[n],e.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(t,a)&&(r[a]=t[a])}return r}var o=n.createContext({}),s=function(t){var e=n.useContext(o),a=e;return t&&(a="function"==typeof t?t(e):l(l({},e),t)),a},p=function(t){var e=s(t.components);return n.createElement(o.Provider,{value:e},t.children)},m="mdxType",g={inlineCode:"code",wrapper:function(t){var e=t.children;return n.createElement(n.Fragment,{},e)}},f=n.forwardRef((function(t,e){var a=t.components,r=t.mdxType,i=t.originalType,o=t.parentName,p=d(t,["components","mdxType","originalType","parentName"]),m=s(a),f=r,k=m["".concat(o,".").concat(f)]||m[f]||g[f]||i;return a?n.createElement(k,l(l({ref:e},p),{},{components:a})):n.createElement(k,l({ref:e},p))}));function k(t,e){var a=arguments,r=e&&e.mdxType;if("string"==typeof t||r){var i=a.length,l=new Array(i);l[0]=f;var d={};for(var o in e)hasOwnProperty.call(e,o)&&(d[o]=e[o]);d.originalType=t,d[m]="string"==typeof t?t:r,l[1]=d;for(var s=2;s<i;s++)l[s]=a[s];return n.createElement.apply(null,l)}return n.createElement.apply(null,a)}f.displayName="MDXCreateElement"},4528:(t,e,a)=>{a.r(e),a.d(e,{assets:()=>o,contentTitle:()=>l,default:()=>g,frontMatter:()=>i,metadata:()=>d,toc:()=>s});var n=a(7462),r=(a(7294),a(3905));const i={title:"Dataset"},l=void 0,d={unversionedId:"dataset",id:"dataset",title:"Dataset",description:"Introduction",source:"@site/docs/dataset.md",sourceDirName:".",slug:"/dataset",permalink:"/docs/dataset",draft:!1,tags:[],version:"current",frontMatter:{title:"Dataset"}},o={},s=[{value:"Introduction",id:"introduction",level:2},{value:"How to Obtain the Dataset",id:"how-to-obtain-the-dataset",level:2},{value:"Data description",id:"data-description",level:2},{value:"1. interaction.csv",id:"1-interactioncsv",level:3},{value:"2. user.csv",id:"2-usercsv",level:3},{value:"3. video.csv",id:"3-videocsv",level:3},{value:"4. bigfive.csv",id:"4-bigfivecsv",level:3},{value:"5. tag_map.csv",id:"5-tag_mapcsv",level:3},{value:"6. video_map.csv",id:"6-video_mapcsv",level:3},{value:"7. preview",id:"7-preview",level:3},{value:"Statistics",id:"statistics",level:2},{value:"1. The basic statistics of REASONER",id:"1-the-basic-statistics-of-reasoner",level:3},{value:"2. Statistics on the users",id:"2-statistics-on-the-users",level:3},{value:"3. Statistics on the videos",id:"3-statistics-on-the-videos",level:3},{value:"Codes for accessing our data",id:"codes-for-accessing-our-data",level:2}],p={toc:s},m="wrapper";function g(t){let{components:e,...i}=t;return(0,r.kt)(m,(0,n.Z)({},p,i,{components:e,mdxType:"MDXLayout"}),(0,r.kt)("h2",{id:"introduction"},"Introduction"),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"REASONER")," is an explainable recommendation dataset. It contains the ground truths for multiple explanation purposes, for example, enhancing the recommendation persuasiveness, informativeness and satisfaction. In this dataset, the ground truth annotators are exactly the people who produce the user-item interactions, and they can make selections from the explanation candidates with multi-modalities. This dataset can be widely used for explainable recommendation, unbiased recommendation, psychology-informed recommendation and so on. Please see our paper for more details."),(0,r.kt)("p",null,"The dataset contains the following files."),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-plain"}," REASONER-Dataset\n  \u2502\u2500\u2500 dataset\n  \u2502   \u251c\u2500\u2500 interaction.csv\n  \u2502   \u251c\u2500\u2500 user.csv\n  \u2502   \u251c\u2500\u2500 video.csv\n  \u2502   \u251c\u2500\u2500 bigfive.csv \n  \u2502   \u251c\u2500\u2500 tag_map.csv \n  \u2502   \u251c\u2500\u2500 video_map.csv \n  \u2502\u2500\u2500 preview\n  \u2502\u2500\u2500 README.md\n")),(0,r.kt)("h2",{id:"how-to-obtain-the-dataset"},"How to Obtain the Dataset"),(0,r.kt)("p",null,"You can directly download the REASONER dataset through the following three links:"),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("p",{parentName:"li"},(0,r.kt)("a",{parentName:"p",href:"https://drive.google.com/drive/folders/1dARhorIUu-ajc5ZsWiG_XY36slRX_wgL?usp=share_link"},(0,r.kt)("img",{parentName:"a",src:"https://img.shields.io/badge/-Google%20Drive-yellow",alt:"Google Drive"})))),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("p",{parentName:"li"},(0,r.kt)("a",{parentName:"p",href:"https://pan.baidu.com/s/1L9AzPe0MkRbMwk6yeDj4QA?pwd=ipxd"},(0,r.kt)("img",{parentName:"a",src:"https://img.shields.io/badge/-Baidu%20Netdisk-lightgrey",alt:"Baidu Netdisk"})))),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("p",{parentName:"li"},(0,r.kt)("a",{parentName:"p",href:"https://1drv.ms/f/s!AiuzqR3lP02KbCZOY3c8bfb3ZWg?e=jWTuc1"},(0,r.kt)("img",{parentName:"a",src:"https://img.shields.io/badge/-OneDrive-blue",alt:"OneDrive"}))))),(0,r.kt)("h2",{id:"data-description"},"Data description"),(0,r.kt)("h3",{id:"1-interactioncsv"},"1. interaction.csv"),(0,r.kt)("p",null,"This file contains the user's annotation records on the video, including the following fields:"),(0,r.kt)("table",null,(0,r.kt)("thead",{parentName:"table"},(0,r.kt)("tr",{parentName:"thead"},(0,r.kt)("th",{parentName:"tr",align:"left"},"Field Name:"),(0,r.kt)("th",{parentName:"tr",align:"left"},"Description"),(0,r.kt)("th",{parentName:"tr",align:"left"},"Type"),(0,r.kt)("th",{parentName:"tr",align:"left"},"Example"))),(0,r.kt)("tbody",{parentName:"table"},(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"user_id"),(0,r.kt)("td",{parentName:"tr",align:"left"},"ID of the user"),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"0")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"video_id"),(0,r.kt)("td",{parentName:"tr",align:"left"},"ID of the viewed video"),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"3650")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"like"),(0,r.kt)("td",{parentName:"tr",align:"left"},"Whether user like the video: 0 means no, 1 means yes"),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"0")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"persuasiveness_tag"),(0,r.kt)("td",{parentName:"tr",align:"left"},'The user selected tags for the question "Which tags are the reasons that you would like to watch this video?" before watching the video'),(0,r.kt)("td",{parentName:"tr",align:"left"},"list"),(0,r.kt)("td",{parentName:"tr",align:"left"},"[4728,2216,2523]")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"rating"),(0,r.kt)("td",{parentName:"tr",align:"left"},"User rating for the video, the range is 1.0~5.0"),(0,r.kt)("td",{parentName:"tr",align:"left"},"float64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"3.0")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"review"),(0,r.kt)("td",{parentName:"tr",align:"left"},"User review for the video"),(0,r.kt)("td",{parentName:"tr",align:"left"},"str"),(0,r.kt)("td",{parentName:"tr",align:"left"},"This animation is very interesting, my friends and I like it very much.")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"informativeness_tag"),(0,r.kt)("td",{parentName:"tr",align:"left"},'The user selected tags for the question "Which features are most informative for this video?" after watching the video'),(0,r.kt)("td",{parentName:"tr",align:"left"},"list"),(0,r.kt)("td",{parentName:"tr",align:"left"},"[2738,1216,2223]")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"satisfaction_tag"),(0,r.kt)("td",{parentName:"tr",align:"left"},'The user selected tags for the question "Which features are you most satisfied with?" after watching the video.'),(0,r.kt)("td",{parentName:"tr",align:"left"},"list"),(0,r.kt)("td",{parentName:"tr",align:"left"},"[738,3226,1323]")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"watch_again"),(0,r.kt)("td",{parentName:"tr",align:"left"},"If the system only show the satisfaction_tag to the user, whether the she would like to watch this video? 0 means no, 1 means yes"),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"0")))),(0,r.kt)("p",null,"Note that if the user chooses to like the video, the ",(0,r.kt)("inlineCode",{parentName:"p"},"watch_again")," item has no meaning and is set to 0."),(0,r.kt)("h3",{id:"2-usercsv"},"2. user.csv"),(0,r.kt)("p",null,"This file contains user profiles."),(0,r.kt)("table",null,(0,r.kt)("thead",{parentName:"table"},(0,r.kt)("tr",{parentName:"thead"},(0,r.kt)("th",{parentName:"tr",align:"left"},"Field Name:"),(0,r.kt)("th",{parentName:"tr",align:"left"},"Description"),(0,r.kt)("th",{parentName:"tr",align:"left"},"Type"),(0,r.kt)("th",{parentName:"tr",align:"left"},"Example"))),(0,r.kt)("tbody",{parentName:"table"},(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"user_id"),(0,r.kt)("td",{parentName:"tr",align:"left"},"ID of the user"),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"1005")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"age"),(0,r.kt)("td",{parentName:"tr",align:"left"},"User age (indicated by ID)"),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"3")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"gender"),(0,r.kt)("td",{parentName:"tr",align:"left"},"User gender: 0 means female, 1 means male"),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"0")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"education"),(0,r.kt)("td",{parentName:"tr",align:"left"},"User education level (indicated by ID)"),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"3")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"career"),(0,r.kt)("td",{parentName:"tr",align:"left"},"User occupation (indicated by ID)"),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"20")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"income"),(0,r.kt)("td",{parentName:"tr",align:"left"},"User income (indicated by ID)"),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"3")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"address"),(0,r.kt)("td",{parentName:"tr",align:"left"},"User address (indicated by ID)"),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"23")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"hobby"),(0,r.kt)("td",{parentName:"tr",align:"left"},"User hobbies"),(0,r.kt)("td",{parentName:"tr",align:"left"},"str"),(0,r.kt)("td",{parentName:"tr",align:"left"},"drawing and soccer.")))),(0,r.kt)("h3",{id:"3-videocsv"},"3. video.csv"),(0,r.kt)("p",null,"This file contains information of videos."),(0,r.kt)("table",null,(0,r.kt)("thead",{parentName:"table"},(0,r.kt)("tr",{parentName:"thead"},(0,r.kt)("th",{parentName:"tr",align:"left"},"Field Name:"),(0,r.kt)("th",{parentName:"tr",align:"left"},"Description"),(0,r.kt)("th",{parentName:"tr",align:"left"},"Type"),(0,r.kt)("th",{parentName:"tr",align:"left"},"Example"))),(0,r.kt)("tbody",{parentName:"table"},(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"video_id"),(0,r.kt)("td",{parentName:"tr",align:"left"},"ID of the video"),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"1")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"title"),(0,r.kt)("td",{parentName:"tr",align:"left"},"Title of the video"),(0,r.kt)("td",{parentName:"tr",align:"left"},"str"),(0,r.kt)("td",{parentName:"tr",align:"left"},"Take it once a day to prevent depression.")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"info"),(0,r.kt)("td",{parentName:"tr",align:"left"},"Introduction of the video"),(0,r.kt)("td",{parentName:"tr",align:"left"},"str"),(0,r.kt)("td",{parentName:"tr",align:"left"},"Just like it, once a day")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"tags"),(0,r.kt)("td",{parentName:"tr",align:"left"},"ID of the video tags"),(0,r.kt)("td",{parentName:"tr",align:"left"},"list"),(0,r.kt)("td",{parentName:"tr",align:"left"},"[112,33,1233]")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"duration"),(0,r.kt)("td",{parentName:"tr",align:"left"},"Duration of the video in seconds"),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"120")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"category"),(0,r.kt)("td",{parentName:"tr",align:"left"},"Category of the video (indicated by ID)"),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"3")))),(0,r.kt)("h3",{id:"4-bigfivecsv"},"4. bigfive.csv"),(0,r.kt)("p",null,"We have the annotators take the ",(0,r.kt)("a",{parentName:"p",href:"https://www.psytoolkit.org/survey-library/big5-bfi-s.html"},"Big Five Personality Test"),", and ",(0,r.kt)("inlineCode",{parentName:"p"},"bigfive.csv")," contains the answers of the annotators to 15 questions, where ","[0, 1, 2, 3, 4, 5]"," correspond to ","[strongly disagree, disagree, somewhat disagree, somewhat agree, agree, strongly agree]",". This file also includes a  ",(0,r.kt)("inlineCode",{parentName:"p"},"user_id")," column."),(0,r.kt)("p",null,"The questions are described as follows:"),(0,r.kt)("table",null,(0,r.kt)("thead",{parentName:"table"},(0,r.kt)("tr",{parentName:"thead"},(0,r.kt)("th",{parentName:"tr",align:"left"},"Question"),(0,r.kt)("th",{parentName:"tr",align:"left"},"Description"))),(0,r.kt)("tbody",{parentName:"table"},(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"Q1"),(0,r.kt)("td",{parentName:"tr",align:"left"},"I think most people are basically well-intentioned")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"Q2"),(0,r.kt)("td",{parentName:"tr",align:"left"},"I get bored with crowded parties")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"Q3"),(0,r.kt)("td",{parentName:"tr",align:"left"},"I'm a person who takes risks and breaks the rules")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"Q4"),(0,r.kt)("td",{parentName:"tr",align:"left"},"i like adventure")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"Q5"),(0,r.kt)("td",{parentName:"tr",align:"left"},"I try to avoid crowded parties and noisy environments")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"Q6"),(0,r.kt)("td",{parentName:"tr",align:"left"},"I like to plan things out at the beginning")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"Q7"),(0,r.kt)("td",{parentName:"tr",align:"left"},"I worry about things that don't matter")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"Q8"),(0,r.kt)("td",{parentName:"tr",align:"left"},"I work or study hard")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"Q9"),(0,r.kt)("td",{parentName:"tr",align:"left"},"Although there are some liars in the society, I think most people are still credible")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"Q10"),(0,r.kt)("td",{parentName:"tr",align:"left"},"I have a spirit of adventure that no one else has")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"Q11"),(0,r.kt)("td",{parentName:"tr",align:"left"},"I often feel uneasy")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"Q12"),(0,r.kt)("td",{parentName:"tr",align:"left"},"I'm always worried that something bad is going to happen")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"Q13"),(0,r.kt)("td",{parentName:"tr",align:"left"},"Although there are some dark things in human society (such as war, crime, fraud), I still believe that human nature is generally good")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"Q14"),(0,r.kt)("td",{parentName:"tr",align:"left"},"I enjoy going to social and entertainment gatherings")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"Q15"),(0,r.kt)("td",{parentName:"tr",align:"left"},"It is one of my characteristics to pay attention to logic and order in doing things")))),(0,r.kt)("h3",{id:"5-tag_mapcsv"},"5. tag_map.csv"),(0,r.kt)("p",null,'Mapping relationship between the tag ID and the tag content. We add 7 additional tags that all videos contain, namely "preview 1, preview 2, preview 3, preview 4, preview 5, title, content".'),(0,r.kt)("table",null,(0,r.kt)("thead",{parentName:"table"},(0,r.kt)("tr",{parentName:"thead"},(0,r.kt)("th",{parentName:"tr",align:"left"},"Field Name:"),(0,r.kt)("th",{parentName:"tr",align:"left"},"Description"),(0,r.kt)("th",{parentName:"tr",align:"left"},"Type"),(0,r.kt)("th",{parentName:"tr",align:"left"},"Example"))),(0,r.kt)("tbody",{parentName:"table"},(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"tag_id"),(0,r.kt)("td",{parentName:"tr",align:"left"},"ID of the tag"),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"1409")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"tag_content"),(0,r.kt)("td",{parentName:"tr",align:"left"},"The content corresponding to the tag"),(0,r.kt)("td",{parentName:"tr",align:"left"},"str"),(0,r.kt)("td",{parentName:"tr",align:"left"},"cute baby")))),(0,r.kt)("h3",{id:"6-video_mapcsv"},"6. video_map.csv"),(0,r.kt)("p",null,"Mapping relationship between the video ID and the folder name in ",(0,r.kt)("inlineCode",{parentName:"p"},"preview"),"."),(0,r.kt)("table",null,(0,r.kt)("thead",{parentName:"table"},(0,r.kt)("tr",{parentName:"thead"},(0,r.kt)("th",{parentName:"tr",align:"left"},"Field Name:"),(0,r.kt)("th",{parentName:"tr",align:"left"},"Description"),(0,r.kt)("th",{parentName:"tr",align:"left"},"Type"),(0,r.kt)("th",{parentName:"tr",align:"left"},"Example"))),(0,r.kt)("tbody",{parentName:"table"},(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"video_id"),(0,r.kt)("td",{parentName:"tr",align:"left"},"ID of the video"),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"1")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"folder_name"),(0,r.kt)("td",{parentName:"tr",align:"left"},"The folder name corresponding to the video"),(0,r.kt)("td",{parentName:"tr",align:"left"},"str"),(0,r.kt)("td",{parentName:"tr",align:"left"},"83062078")))),(0,r.kt)("h3",{id:"7-preview"},"7. preview"),(0,r.kt)("p",null,"Each video contains 5 image previews."),(0,r.kt)("p",null,"The mapping relationship between the folder name and the video ID is in ",(0,r.kt)("inlineCode",{parentName:"p"},"video_map.csv"),"."),(0,r.kt)("h2",{id:"statistics"},"Statistics"),(0,r.kt)("h3",{id:"1-the-basic-statistics-of-reasoner"},"1. The basic statistics of REASONER"),(0,r.kt)("p",null,'We have collected the basic information of the REASONER dataset and listed it in the table below. "u-v" represents the number of interactions between users and videos, "u-t" represents the number of tags clicked by users, and "Q1, Q2, Q3" respectively represent the persuasiveness, informativeness, and satisfaction of the tags.'),(0,r.kt)("table",null,(0,r.kt)("thead",{parentName:"table"},(0,r.kt)("tr",{parentName:"thead"},(0,r.kt)("th",{parentName:"tr",align:null},"#User"),(0,r.kt)("th",{parentName:"tr",align:null},"#Video"),(0,r.kt)("th",{parentName:"tr",align:null},"#Tag"),(0,r.kt)("th",{parentName:"tr",align:null},"#u-v"),(0,r.kt)("th",{parentName:"tr",align:null},"#u-t (Q1)"),(0,r.kt)("th",{parentName:"tr",align:null},"#u-t (Q2)"),(0,r.kt)("th",{parentName:"tr",align:null},"#u-t (Q3)"))),(0,r.kt)("tbody",{parentName:"table"},(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:null},"2,997"),(0,r.kt)("td",{parentName:"tr",align:null},"4,672"),(0,r.kt)("td",{parentName:"tr",align:null},"6,115"),(0,r.kt)("td",{parentName:"tr",align:null},"58,497"),(0,r.kt)("td",{parentName:"tr",align:null},"263,885"),(0,r.kt)("td",{parentName:"tr",align:null},"271,456"),(0,r.kt)("td",{parentName:"tr",align:null},"256,079")))),(0,r.kt)("h3",{id:"2-statistics-on-the-users"},"2. Statistics on the users"),(0,r.kt)("div",{style:{textAlign:"center"}},(0,r.kt)("img",{src:a(7396).Z,style:{width:"80%"}})),(0,r.kt)("h3",{id:"3-statistics-on-the-videos"},"3. Statistics on the videos"),(0,r.kt)("div",{style:{textAlign:"center"}},(0,r.kt)("img",{src:a(6323).Z,style:{width:"80%"}})),(0,r.kt)("h2",{id:"codes-for-accessing-our-data"},"Codes for accessing our data"),(0,r.kt)("p",null,"We provide code to read the data into data frame with ",(0,r.kt)("em",{parentName:"p"},"pandas"),". "),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},"import pandas as pd\n\n# access interaction.csv\ninteraction_df = pd.read_csv('interaction.csv', sep='\\t', header=0)\n# get the first ten lines\nprint(interaction_df.head(10))\n# get each column \n# ['user_id', 'video_id', 'like', 'persuasiveness_tag', 'rating', 'review', 'informativeness_tag', 'satisfaction_tag', 'watch_again', ]\nfor col in interaction_df.columns:\n    print(interaction_df[col][:10])\n\n# access user.csv\nuser_df = pd.read_csv('user.csv', sep='\\t', header=0)\nprint(user_df.head(10))\n# ['user_id', 'age', 'gender', 'education', 'career', 'income', 'address', 'hobby']\nfor col in user_df.columns:\n    print(user_df[col][:10])\n  \n# access video.csv\nvideo_df = pd.read_csv('video.csv', sep='\\t', header=0)\nprint(video_df.head(10))\n# ['video_id', 'title', 'info', 'tags', 'duration', 'category']\nfor col in video_df.columns:\n    print(video_df[col][:10])\n\n# access bigfive.csv\nbigfive_df = pd.read_csv('bigfive.csv', sep='\\t', header=0)\nprint(bigfive_df.head(10))\n# ['user_id', 'Q1', ..., 'Q15']\nfor col in bigfive_df.columns:\n    print(bigfive_df[col][:10])\n\n# access tag_map.csv\ntag_map_df = pd.read_csv('tag_map.csv', sep='\\t', header=0)\nprint(tag_map_df.head(10))\n# ['tag_id', 'tag_content']\nfor col in tag_map_df.columns:\n    print(tag_map_df[col][:10])\n  \n# access video_map.csv\nvideo_map_df = pd.read_csv('video_map.csv', sep='\\t', header=0)\nprint(video_map_df.head(10))\n# ['video_id', 'folder_name']\nfor col in video_map_df.columns:\n    print(video_map_df[col][:10])\n")))}g.isMDXComponent=!0},7396:(t,e,a)=>{a.d(e,{Z:()=>n});const n=a.p+"assets/images/user-0107da845ccde6b3fdbf309bd3e20f0a.png"},6323:(t,e,a)=>{a.d(e,{Z:()=>n});const n=a.p+"assets/images/video-83e6c5af375680a01831e51b9ded0091.png"}}]);