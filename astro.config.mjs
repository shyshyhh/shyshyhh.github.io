import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  // Change this if you later buy a custom domain (e.g. 'https://hanyusong.com')
  site: 'https://shyshyhh.github.io',
  redirects: {
    '/writing/small-models-can-learn-what-they-cannot-judge':
      '/writing/lesson-sensitivity-without-validated-lesson-selection',
  },
  integrations: [sitemap()],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark' },
    },
  },
});
