# Personal site

Minimal Astro site: markdown writing with math (KaTeX) + code highlighting, a papers page, RSS, sitemap, dark mode.

## Run locally

```sh
npm install
npm run dev        # http://localhost:4321
```

## Add a post

1. Copy `src/content/blog/_template.md` to a new file in the same folder.
   The filename becomes the URL: `grpo-notes.md` → `/writing/grpo-notes/`.
2. Fill in `title`, `description`, `date`, and set `draft: false`.
3. That's it — it appears on the homepage, `/writing/`, and the RSS feed.

Files starting with `_` are ignored, and posts with `draft: true` never build.

## Edit your info

- `src/consts.ts` — name, tagline, links (GitHub/Scholar are TODO).
- `src/pages/index.astro` — the bio paragraphs.
- `src/pages/papers.astro` — the publications list (add new entries to the array).
- `astro.config.mjs` — set `site` to your real domain before deploying (RSS/sitemap use it).

## Deploy (pick one)

**Cloudflare Pages (recommended — free, fast, easy custom domain):**
1. Push this folder to a GitHub repo.
2. Cloudflare dashboard → Workers & Pages → Create → connect the repo.
3. Framework preset: Astro. Build command `npm run build`, output `dist`. Done.
4. Add your custom domain under the project's Custom Domains tab.

**GitHub Pages:**
1. Push to a GitHub repo, then repo Settings → Pages → Source: GitHub Actions.
2. The included workflow (`.github/workflows/deploy.yml`) builds and deploys on
   every push to `main`.
3. Set `site` in `astro.config.mjs` to `https://<username>.github.io` (and
   `base` if the repo isn't named `<username>.github.io` — see Astro docs).
