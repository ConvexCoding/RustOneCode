import adapter from '@sveltejs/adapter-static'
import preprocess from 'svelte-preprocess'
import ViteRsw from 'vite-plugin-rsw'
import path from 'path'

/** @type {import('@sveltejs/kit').Config} */
const config = {
  // Consult https://github.com/sveltejs/svelte-preprocess
  // for more information about preprocessors
  preprocess: [
    preprocess({
      postcss: true,
    }),
  ],

  kit: {
    adapter: adapter({ fallback: 'app.html' }),
    vite: {
      plugins: [
        ViteRsw.default({
          crates: [
            {
              name: 'asplib',
              outDir: path.resolve('./src/lib/asplib'),
              unwatch: [
                '../src/lib',
                '../src/routes',
                '../src/app.css',
                '../src/app.html',
                '../src/hooks.ts',
                '../src/app.d.ts',
              ],
            },
          ],
        }),
      ],
    },
  },
}

export default config
