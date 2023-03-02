/* eslint-disable @typescript-eslint/no-var-requires */
// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require("prism-react-renderer/themes/github");
const darkCodeTheme = require("prism-react-renderer/themes/dracula");

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "REASONER",
  favicon: "img/favicon.ico",

  // Set the production url of your site here
  url: "https://reasoner2023.github.io/",
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: "/",

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: "REASONER2023", // Usually your GitHub org/user name.
  projectName: "reasoner2023.github.io", // Usually your repo name.

  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "warn",

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: false,
        },
        blog: false,
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: "REASONER",
        logo: {
          alt: "REASONER Logo",
          src: "img/logo.png",
        },
        items: [
          {
            type: "doc",
            docId: "dataset",
            position: "left",
            label: "Dataset",
          },
          {
            type: "doc",
            docId: "library",
            position: "left",
            label: "Library",
          },
          {
            to: "https://arxiv.org/abs/2303.00168v1",
            position: "left",
            label: "Paper",
          },
          {
            type: "doc",
            docId: "about",
            position: "left",
            label: "About",
          },
          {
            href: "https://github.com/REASONER2023/reasoner2023.github.io",
            label: "GitHub",
            position: "right",
          },
        ],
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
};

module.exports = config;
