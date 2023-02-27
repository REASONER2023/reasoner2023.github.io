import React from "react";
import clsx from "clsx";
import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import useBaseUrl from "@docusaurus/useBaseUrl";
import styles from "./index.module.css";
import Translate from "@docusaurus/Translate";

const features = [
  {
    title: "Multi-modal Explanations",
    imageUrl: "img/explanation.svg",
    description:
      "In our dataset, the users can select " +
      "both textual and visual explanations according to their preferences for each video. ",
  },
  {
    title: "Multi-aspect Explanation Ground Truth",
    imageUrl: "img/truth.svg",
    description:
      "In our dataset, we provide the explanation ground truths " +
      "from the perspectives of persuasiveness, informativeness and satisfaction. ",
  },
  {
    title: "Real User Labeled Dataset",
    imageUrl: "img/toolkit.svg",
    description:
      "In our dataset, the ground truths annotators are " +
      "exactly the people who produce the user-item interactions in the dataset. ",
  },
  {
    title: "Comprehensive User Information",
    imageUrl: "img/feedback.svg",
    description:
      "In our dataset, we have collected user profiles like the age, " +
      "gender, income, hobby and physiology information.",
  },
];

function Feature({ imageUrl, title, description }) {
  const imgUrl = useBaseUrl(imageUrl);
  return (
    <div className={clsx("col col--3", styles.feature)}>
      {imgUrl && (
        <div className="text--center">
          <img className={styles.featureImage} src={imgUrl} alt={title} />
        </div>
      )}
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  );
}

function Home() {
  const context = useDocusaurusContext();
  const { siteConfig: { customFields = {}, tagline } = {} } = context;
  return (
    <Layout title={tagline} description={customFields.description as string}>
      <div className={styles.hero}>
        <div className={styles.heroInner}>
          <h1 className={styles.heroProjectTagline}>
            <img
              alt="Reasoner logo"
              className={styles.heroLogo}
              src={useBaseUrl("img/logo.png")}
            />
            <span className={styles.heroTitleTextHtml}>
              <Translate
                id="homepage.hero.title"
                description="Home page hero title, can contain simple html tags"
                values={{
                  explainable: <b>Explainable</b>,
                  multi_aspect: <b>Multi-aspect</b>,
                  ground_truth: <b>Ground Truth</b>,
                }}
              >
                {`An {explainable} Recommendation Dataset with {multi_aspect} Real User Labeled {ground_truth}`}
              </Translate>
            </span>
          </h1>
          <div className={styles.indexCtas}>
            <Link
              className={clsx("margin-left--md", styles.indexTryMeButton)}
              to="https://github.com/REASONER2023/reasoner2023.github.io"
            >
              View on GitHub
            </Link>
          </div>
        </div>
      </div>
      <main>
        {features && features.length > 0 && (
          <section className={styles.features}>
            <div className="container">
              <div className="row">
                {features.map((props, idx) => (
                  <Feature key={idx} {...props} />
                ))}
              </div>
            </div>
          </section>
        )}
      </main>
    </Layout>
  );
}

export default Home;
