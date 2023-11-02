# 🌲 Forest Cover Classification

NUS-ISS, PRS Practice Module, WebUI & Client

## Project Structure

* `package.json` stores all packages the project has used
* `lib` folder places all components
* `routes` folder stores all pages with `/` home route and `analysis` analysis page route

```
Deforestation_Detection_Frontend
├── README.md
├── bun.lockb
├── jsconfig.json
├── node_modules
├── package.json
├── postcss.config.cjs
├── src
│   ├── app.d.ts
│   ├── app.html
│   ├── app.postcss
│   ├── lib
│   │   ├── Image.svelte
│   │   ├── Submit.svelte
│   │   ├── analyze.js
│   │   └── stores.js
│   └── routes
│       ├── +layout.svelte
│       ├── +page.server.js
│       ├── +page.svelte
│       └── analysis
│           ├── +layout.svelte
│           ├── +page.server.js
│           └── +page.svelte
├── static
│   ├── favicon.png
│   └── forest.webp
├── svelte.config.js
├── tailwind.config.cjs
└── vite.config.js
```

## Developing

1. Download Node JS from here: https://nodejs.org/en/download
   
2. Check installation using command ```npm -v```. (Note: In windows, do not make use of anaconda command prompt here). 

4. Install dependencies with `npm install` (or `pnpm install` or `yarn` or `bun install`), 

5. Run either of the following commands to start a development server:

```bash
# Using npm
npm run dev
# Using bun
bun --bun run dev
```
