"use strict";(self.webpackChunkreasoner=self.webpackChunkreasoner||[]).push([[535],{3905:(e,n,t)=>{t.d(n,{Zo:()=>l,kt:()=>h});var a=t(7294);function r(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function i(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);n&&(a=a.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,a)}return t}function o(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?i(Object(t),!0).forEach((function(n){r(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):i(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function s(e,n){if(null==e)return{};var t,a,r=function(e,n){if(null==e)return{};var t,a,r={},i=Object.keys(e);for(a=0;a<i.length;a++)t=i[a],n.indexOf(t)>=0||(r[t]=e[t]);return r}(e,n);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(a=0;a<i.length;a++)t=i[a],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(r[t]=e[t])}return r}var c=a.createContext({}),u=function(e){var n=a.useContext(c),t=n;return e&&(t="function"==typeof e?e(n):o(o({},n),e)),t},l=function(e){var n=u(e.components);return a.createElement(c.Provider,{value:n},e.children)},p="mdxType",m={inlineCode:"code",wrapper:function(e){var n=e.children;return a.createElement(a.Fragment,{},n)}},d=a.forwardRef((function(e,n){var t=e.components,r=e.mdxType,i=e.originalType,c=e.parentName,l=s(e,["components","mdxType","originalType","parentName"]),p=u(t),d=r,h=p["".concat(c,".").concat(d)]||p[d]||m[d]||i;return t?a.createElement(h,o(o({ref:n},l),{},{components:t})):a.createElement(h,o({ref:n},l))}));function h(e,n){var t=arguments,r=n&&n.mdxType;if("string"==typeof e||r){var i=t.length,o=new Array(i);o[0]=d;var s={};for(var c in n)hasOwnProperty.call(n,c)&&(s[c]=n[c]);s.originalType=e,s[p]="string"==typeof e?e:r,o[1]=s;for(var u=2;u<i;u++)o[u]=t[u];return a.createElement.apply(null,o)}return a.createElement.apply(null,t)}d.displayName="MDXCreateElement"},1171:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>c,contentTitle:()=>o,default:()=>m,frontMatter:()=>i,metadata:()=>s,toc:()=>u});var a=t(7462),r=(t(7294),t(3905));const i={},o="About",s={unversionedId:"about",id:"about",title:"About",description:"Organization Team",source:"@site/docs/about.md",sourceDirName:".",slug:"/about",permalink:"/docs/about",draft:!1,tags:[],version:"current",frontMatter:{}},c={},u=[{value:"Organization Team",id:"organization-team",level:2},{value:"Contact us",id:"contact-us",level:2},{value:"Acknowledgements",id:"acknowledgements",level:2},{value:"License",id:"license",level:2},{value:"Cite",id:"cite",level:2}],l={toc:u},p="wrapper";function m(e){let{components:n,...t}=e;return(0,r.kt)(p,(0,a.Z)({},l,t,{components:n,mdxType:"MDXLayout"}),(0,r.kt)("h1",{id:"about"},"About"),(0,r.kt)("h2",{id:"organization-team"},"Organization Team"),(0,r.kt)("p",null,"Xu Chen (Renmin University of China), Jingsen Zhang (Renmin University of China), Lei Wang (Renmin University of China), Quanyu Dai (Huawei Noah's Ark Lab), Zhenhua Dong (Huawei Noah's Ark Lab), Ruiming Tang (Huawei Noah's Ark Lab), Rui Zhang (Huawei Noah's Ark Lab), Li Chen (Hong Kong Baptist University), Ji-Ron Wen (Renmin University of China)."),(0,r.kt)("h2",{id:"contact-us"},"Contact us"),(0,r.kt)("p",null,"If you have any question about the dataset, you may contact us through ",(0,r.kt)("a",{parentName:"p",href:"mailto:reasonerdataset@gmail.com"},"reasonerdataset@gmail.com")),(0,r.kt)("h2",{id:"acknowledgements"},"Acknowledgements"),(0,r.kt)("p",null,"We sincerely thank the contributions of Zhenlei Wang, Rui Zhou, Kun Lin, Zeyu Zhang, Jiakai Tang and Hao Yang from Renmin University of China for checking the unreasonable results in the dataset construction process."),(0,r.kt)("h2",{id:"license"},"License"),(0,r.kt)("p",null,"Our licensing for the dataset is under a CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0), with the additional terms included herein. See official instructions ",(0,r.kt)("a",{parentName:"p",href:"https://creativecommons.org/licenses/by-nc/4.0/"},"here"),"."),(0,r.kt)("h2",{id:"cite"},"Cite"),(0,r.kt)("p",null,"Please cite the following paper as the reference if you use our code or dataset.",(0,r.kt)("a",{parentName:"p",href:"https://arxiv.org/abs/2303.00168"},(0,r.kt)("img",{parentName:"a",src:"https://img.shields.io/badge/-Paper%20Link-lightgrey",alt:"LINK"}))," ",(0,r.kt)("a",{parentName:"p",href:"https://arxiv.org/pdf/2303.00168.pdf"},(0,r.kt)("img",{parentName:"a",src:"https://img.shields.io/badge/-PDF-red",alt:"PDF"}))),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre"},"@misc{chen2023reasoner,\n      title={REASONER: An Explainable Recommendation Dataset with Multi-aspect Real User Labeled Ground Truths Towards more Measurable Explainable Recommendation}, \n      author={Xu Chen and Jingsen Zhang and Lei Wang and Quanyu Dai and Zhenhua Dong and Ruiming Tang and Rui Zhang and Li Chen and Ji-Rong Wen},\n      year={2023},\n      eprint={2303.00168},\n      archivePrefix={arXiv},\n      primaryClass={cs.IR}\n}\n")))}m.isMDXComponent=!0}}]);