/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  overviewSidebar: [
    {
      type: 'html',
      value: '<a class="navbar__brand custom-top-header" href="/deeppast"><div class="navbar__logo"><img src="./img/social/favicon-64x64.png" alt="Deep Past Challenge Logo" class="themedImage_ToTc themedImage--dark_i4oU"></div><b class="navbar__title text--truncate">Deep Past Challenge</b></a>'
    },
    {
      type: 'category',
      label: 'Overview',
      link: { type: 'doc', id: 'landing' },
      items: [
        { type: 'doc', id: 'get_started' },
      ]
    },
    {
      type: 'category',
      label: 'Competition',
      link: { type: 'doc', id: 'challenges' },
      items: [
        { type: 'doc', id: 'challenges' },
      ],
    },
    {
      type: 'category',
      label: 'Data',
      collapsible: true,
      collapsed: true,
      link: { type: 'doc', id: 'data' },
      items: [
        { type: 'doc', id: 'data_akkadian' },
        { type: 'doc', id: 'data_english' },
        { type: 'doc', id: 'data_unicode' },
        { type: 'doc', id: 'data_segments' },
      ],
    },
    {
      type: 'category',
      label: 'Tutorials',
      collapsible: true,
      collapsed: true,
      link: { type: 'doc', id: 'tutorial' },
      items: [
        { type: 'doc', id: 'tutorial_accessibility' },
        { type: 'doc', id: 'tutorial_conversion' },
        { type: 'doc', id: 'tutorial_linguistics' },
      ],
    },
    { type: 'doc', id: 'faq' },
    {
      type: 'link',
      label: 'Discord',
      href: 'https://discord.gg/XtCMyTrVCF',
    }
    ],
  archiveSidebar: [
    {
      type: 'html',
      value: '<a class="navbar__brand custom-top-header" href="/"><div class="navbar__logo"><img src="/img/social/favicon-64x64.png" alt="Vesuvius Challenge Logo" class="themedImage_ToTc themedImage--dark_i4oU"></div><b class="navbar__title text--truncate">Vesuvius Challenge</b></a>'
    },
    { type: 'doc', id: 'background' },
    { type: 'doc', id: 'firstletters' },
    { type: 'doc', id: 'grandprize' },
    { type: 'doc', id: 'livestream' },
  ],
};

module.exports = sidebars;
