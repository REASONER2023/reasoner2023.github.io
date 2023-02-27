"use strict";(self.webpackChunkreasoner=self.webpackChunkreasoner||[]).push([[598],{3905:(t,e,a)=>{a.d(e,{Zo:()=>p,kt:()=>k});var n=a(7294);function r(t,e,a){return e in t?Object.defineProperty(t,e,{value:a,enumerable:!0,configurable:!0,writable:!0}):t[e]=a,t}function i(t,e){var a=Object.keys(t);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(t);e&&(n=n.filter((function(e){return Object.getOwnPropertyDescriptor(t,e).enumerable}))),a.push.apply(a,n)}return a}function l(t){for(var e=1;e<arguments.length;e++){var a=null!=arguments[e]?arguments[e]:{};e%2?i(Object(a),!0).forEach((function(e){r(t,e,a[e])})):Object.getOwnPropertyDescriptors?Object.defineProperties(t,Object.getOwnPropertyDescriptors(a)):i(Object(a)).forEach((function(e){Object.defineProperty(t,e,Object.getOwnPropertyDescriptor(a,e))}))}return t}function d(t,e){if(null==t)return{};var a,n,r=function(t,e){if(null==t)return{};var a,n,r={},i=Object.keys(t);for(n=0;n<i.length;n++)a=i[n],e.indexOf(a)>=0||(r[a]=t[a]);return r}(t,e);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(t);for(n=0;n<i.length;n++)a=i[n],e.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(t,a)&&(r[a]=t[a])}return r}var o=n.createContext({}),s=function(t){var e=n.useContext(o),a=e;return t&&(a="function"==typeof t?t(e):l(l({},e),t)),a},p=function(t){var e=s(t.components);return n.createElement(o.Provider,{value:e},t.children)},m="mdxType",f={inlineCode:"code",wrapper:function(t){var e=t.children;return n.createElement(n.Fragment,{},e)}},c=n.forwardRef((function(t,e){var a=t.components,r=t.mdxType,i=t.originalType,o=t.parentName,p=d(t,["components","mdxType","originalType","parentName"]),m=s(a),c=r,k=m["".concat(o,".").concat(c)]||m[c]||f[c]||i;return a?n.createElement(k,l(l({ref:e},p),{},{components:a})):n.createElement(k,l({ref:e},p))}));function k(t,e){var a=arguments,r=e&&e.mdxType;if("string"==typeof t||r){var i=a.length,l=new Array(i);l[0]=c;var d={};for(var o in e)hasOwnProperty.call(e,o)&&(d[o]=e[o]);d.originalType=t,d[m]="string"==typeof t?t:r,l[1]=d;for(var s=2;s<i;s++)l[s]=a[s];return n.createElement.apply(null,l)}return n.createElement.apply(null,a)}c.displayName="MDXCreateElement"},4528:(t,e,a)=>{a.r(e),a.d(e,{assets:()=>o,contentTitle:()=>l,default:()=>f,frontMatter:()=>i,metadata:()=>d,toc:()=>s});var n=a(7462),r=(a(7294),a(3905));const i={title:"Dataset"},l=void 0,d={unversionedId:"dataset",id:"dataset",title:"Dataset",description:"Introduction",source:"@site/docs/dataset.md",sourceDirName:".",slug:"/dataset",permalink:"/docs/dataset",draft:!1,tags:[],version:"current",frontMatter:{title:"Dataset"}},o={},s=[{value:"Introduction",id:"introduction",level:2},{value:"How to Obtain the Dataset",id:"how-to-obtain-the-dataset",level:2},{value:"Data description",id:"data-description",level:2},{value:"1. Descriptions of the fields in\xa0<code>interaction.csv</code>",id:"1-descriptions-of-the-fields-ininteractioncsv",level:3},{value:"2. Descriptions of the fields in\xa0<code>user.csv</code>",id:"2-descriptions-of-the-fields-inusercsv",level:3},{value:"3. Descriptions of the fields in\xa0<code>video.csv.</code>",id:"3-descriptions-of-the-fields-invideocsv",level:3},{value:"Statistics",id:"statistics",level:2},{value:"1. The basic statistics of REASONER",id:"1-the-basic-statistics-of-reasoner",level:3},{value:"2. Statistics on the users",id:"2-statistics-on-the-users",level:3},{value:"3. Statistics on the videos",id:"3-statistics-on-the-videos",level:3}],p={toc:s},m="wrapper";function f(t){let{components:e,...i}=t;return(0,r.kt)(m,(0,n.Z)({},p,i,{components:e,mdxType:"MDXLayout"}),(0,r.kt)("h2",{id:"introduction"},"Introduction"),(0,r.kt)("p",null,"REASONER is an explainable recommendation dataset. It contains the ground truths for multiple explanation purposes, for example, enhancing the recommendation persuasiveness, informativeness and so on. In this dataset, the ground truth annotators are exactly the people who produce the user-item interactions, and they can make selections from the explanation candidates with multi-modalities. This dataset can be widely used for explainable recommendation, unbiased recommendation, psychology-informed recommendation and so on. Please see our paper for details.  "),(0,r.kt)("h2",{id:"how-to-obtain-the-dataset"},"How to Obtain the Dataset"),(0,r.kt)("p",null,"Please provide us with your basic information including your name, institution, and purpose of use to request the dataset. You can email us at ",(0,r.kt)("a",{parentName:"p",href:"mailto:reasonerdataset@gmail.com"},"reasonerdataset@gmail.com"),"."),(0,r.kt)("h2",{id:"data-description"},"Data description"),(0,r.kt)("p",null,(0,r.kt)("em",{parentName:"p"},"REASONER"),"\xa0contains fifty thousand of user-item interactions as well as the side information including the video categories and user profile. Three files are included in the dataset:"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-plain"}," REASONER\n  \u251c\u2500\u2500 data\n  \u2502\xa0\xa0 \u251c\u2500\u2500 interaction.csv\n  \u2502\xa0\xa0 \u251c\u2500\u2500 user.csv\n  \u2502\xa0\xa0 \u251c\u2500\u2500 video.csv\n")),(0,r.kt)("h3",{id:"1-descriptions-of-the-fields-ininteractioncsv"},"1. Descriptions of the fields in\xa0",(0,r.kt)("inlineCode",{parentName:"h3"},"interaction.csv")),(0,r.kt)("table",null,(0,r.kt)("thead",{parentName:"table"},(0,r.kt)("tr",{parentName:"thead"},(0,r.kt)("th",{parentName:"tr",align:"left"},"Field Name:"),(0,r.kt)("th",{parentName:"tr",align:"left"},"Description"),(0,r.kt)("th",{parentName:"tr",align:"left"},"Type"),(0,r.kt)("th",{parentName:"tr",align:"left"},"Example"))),(0,r.kt)("tbody",{parentName:"table"},(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"user_id"),(0,r.kt)("td",{parentName:"tr",align:"left"},"ID of the user."),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"0")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"video_id"),(0,r.kt)("td",{parentName:"tr",align:"left"},"ID of the viewed video."),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"3650")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"like"),(0,r.kt)("td",{parentName:"tr",align:"left"},"Whether user like the video. 0 means no, 1 means yes"),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"0")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"reason_tag"),(0,r.kt)("td",{parentName:"tr",align:"left"},"Tags that reflect why the user likes/dislikes the video."),(0,r.kt)("td",{parentName:"tr",align:"left"},"list"),(0,r.kt)("td",{parentName:"tr",align:"left"},"[4728,2216,2523]")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"rating"),(0,r.kt)("td",{parentName:"tr",align:"left"},"User rating for the video."),(0,r.kt)("td",{parentName:"tr",align:"left"},"float64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"3.0")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"review"),(0,r.kt)("td",{parentName:"tr",align:"left"},"User review for the video."),(0,r.kt)("td",{parentName:"tr",align:"left"},"str"),(0,r.kt)("td",{parentName:"tr",align:"left"},"This animation is very interesting, my friends and I like it very much.")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"video_tag"),(0,r.kt)("td",{parentName:"tr",align:"left"},"Tags that reflect the content of the video.",(0,r.kt)("br",null)),(0,r.kt)("td",{parentName:"tr",align:"left"},"list"),(0,r.kt)("td",{parentName:"tr",align:"left"},"[2738,1216,2223]")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"interest_tag"),(0,r.kt)("td",{parentName:"tr",align:"left"},"Tags that reflect user interests."),(0,r.kt)("td",{parentName:"tr",align:"left"},"list"),(0,r.kt)("td",{parentName:"tr",align:"left"},"[738,3226,1323]")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"watch_again"),(0,r.kt)("td",{parentName:"tr",align:"left"},"Show only the interest tags, will the video be viewed. 0 means no, 1 means yes"),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"0")))),(0,r.kt)("p",null,"Note that if the user chooses to like the video, the ",(0,r.kt)("inlineCode",{parentName:"p"},"watch_again")," item has no meaning and is set to 0."),(0,r.kt)("h3",{id:"2-descriptions-of-the-fields-inusercsv"},"2. Descriptions of the fields in\xa0",(0,r.kt)("inlineCode",{parentName:"h3"},"user.csv")),(0,r.kt)("table",null,(0,r.kt)("thead",{parentName:"table"},(0,r.kt)("tr",{parentName:"thead"},(0,r.kt)("th",{parentName:"tr",align:"left"},"Field Name:"),(0,r.kt)("th",{parentName:"tr",align:"left"},"Description"),(0,r.kt)("th",{parentName:"tr",align:"left"},"Type"),(0,r.kt)("th",{parentName:"tr",align:"left"},"Example"))),(0,r.kt)("tbody",{parentName:"table"},(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"user_id"),(0,r.kt)("td",{parentName:"tr",align:"left"},"ID of the user."),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"1005")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"age"),(0,r.kt)("td",{parentName:"tr",align:"left"},"User age (indicated by ID)."),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"3")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"gender"),(0,r.kt)("td",{parentName:"tr",align:"left"},"User gender. 0 means female, 1 means male."),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"0")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"education"),(0,r.kt)("td",{parentName:"tr",align:"left"},"User education level (indicated by ID)."),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"3")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"career"),(0,r.kt)("td",{parentName:"tr",align:"left"},"User occupation (indicated by ID)."),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"20")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"income"),(0,r.kt)("td",{parentName:"tr",align:"left"},"User income (indicated by ID)."),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"3")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"address"),(0,r.kt)("td",{parentName:"tr",align:"left"},"User address (indicated by ID)."),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"23")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"hobby"),(0,r.kt)("td",{parentName:"tr",align:"left"},"User hobbies."),(0,r.kt)("td",{parentName:"tr",align:"left"},"str"),(0,r.kt)("td",{parentName:"tr",align:"left"},"drawing and soccer.")))),(0,r.kt)("h3",{id:"3-descriptions-of-the-fields-invideocsv"},"3. Descriptions of the fields in\xa0",(0,r.kt)("inlineCode",{parentName:"h3"},"video.csv.")),(0,r.kt)("table",null,(0,r.kt)("thead",{parentName:"table"},(0,r.kt)("tr",{parentName:"thead"},(0,r.kt)("th",{parentName:"tr",align:"left"},"Field Name:"),(0,r.kt)("th",{parentName:"tr",align:"left"},"Description"),(0,r.kt)("th",{parentName:"tr",align:"left"},"Type"),(0,r.kt)("th",{parentName:"tr",align:"left"},"Example"))),(0,r.kt)("tbody",{parentName:"table"},(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"video_id"),(0,r.kt)("td",{parentName:"tr",align:"left"},"ID of the video."),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"1")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"title"),(0,r.kt)("td",{parentName:"tr",align:"left"},"Title of the video."),(0,r.kt)("td",{parentName:"tr",align:"left"},"str"),(0,r.kt)("td",{parentName:"tr",align:"left"},"Take it once a day to prevent depression.")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"info"),(0,r.kt)("td",{parentName:"tr",align:"left"},"Introduction of the video."),(0,r.kt)("td",{parentName:"tr",align:"left"},"str"),(0,r.kt)("td",{parentName:"tr",align:"left"},"Just like it, once a day")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"tags"),(0,r.kt)("td",{parentName:"tr",align:"left"},"ID of the video tags."),(0,r.kt)("td",{parentName:"tr",align:"left"},"list"),(0,r.kt)("td",{parentName:"tr",align:"left"},"[112, 33,1233]")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"duration"),(0,r.kt)("td",{parentName:"tr",align:"left"},"Duration of the video in seconds."),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"120")),(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"left"},"category"),(0,r.kt)("td",{parentName:"tr",align:"left"},"Category of the video (indicated by ID)."),(0,r.kt)("td",{parentName:"tr",align:"left"},"int64"),(0,r.kt)("td",{parentName:"tr",align:"left"},"3")))),(0,r.kt)("h2",{id:"statistics"},"Statistics"),(0,r.kt)("h3",{id:"1-the-basic-statistics-of-reasoner"},"1. The basic statistics of REASONER"),(0,r.kt)("p",null,'We have collected the basic information of the REASONER dataset and listed it in the table below. "u-v" represents the number of interactions between users and videos, "u-t" represents the number of tags clicked by users, and "Q1, Q2, Q3" respectively represent the persuasiveness, informativeness, and satisfaction of the tags.'),(0,r.kt)("table",null,(0,r.kt)("thead",{parentName:"table"},(0,r.kt)("tr",{parentName:"thead"},(0,r.kt)("th",{parentName:"tr",align:null},"#User"),(0,r.kt)("th",{parentName:"tr",align:null},"#Video"),(0,r.kt)("th",{parentName:"tr",align:null},"#Tag"),(0,r.kt)("th",{parentName:"tr",align:null},"#u-v"),(0,r.kt)("th",{parentName:"tr",align:null},"#u-t (Q1)"),(0,r.kt)("th",{parentName:"tr",align:null},"#u-t (Q2)"),(0,r.kt)("th",{parentName:"tr",align:null},"#u-t (Q3)"))),(0,r.kt)("tbody",{parentName:"table"},(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:null},"2,997"),(0,r.kt)("td",{parentName:"tr",align:null},"4,672"),(0,r.kt)("td",{parentName:"tr",align:null},"6,115"),(0,r.kt)("td",{parentName:"tr",align:null},"58,497"),(0,r.kt)("td",{parentName:"tr",align:null},"263,885"),(0,r.kt)("td",{parentName:"tr",align:null},"271,456"),(0,r.kt)("td",{parentName:"tr",align:null},"256,079")))),(0,r.kt)("h3",{id:"2-statistics-on-the-users"},"2. Statistics on the users"),(0,r.kt)("div",{style:{textAlign:"center"}},(0,r.kt)("img",{src:a(7396).Z,style:{width:"80%"}})),(0,r.kt)("h3",{id:"3-statistics-on-the-videos"},"3. Statistics on the videos"),(0,r.kt)("div",{style:{textAlign:"center"}},(0,r.kt)("img",{src:a(6323).Z,style:{width:"80%"}})))}f.isMDXComponent=!0},7396:(t,e,a)=>{a.d(e,{Z:()=>n});const n=a.p+"assets/images/user-0107da845ccde6b3fdbf309bd3e20f0a.png"},6323:(t,e,a)=>{a.d(e,{Z:()=>n});const n=a.p+"assets/images/video-83e6c5af375680a01831e51b9ded0091.png"}}]);