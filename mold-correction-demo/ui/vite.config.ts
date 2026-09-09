import { sites } from '@openai/sites-vite-plugin';
import tailwindcss from '@tailwindcss/postcss';
import vinext from 'vinext';
import { defineConfig, type HmrContext } from 'vite';
import hostingConfig from './.openai/hosting.json';

const SITE_CREATOR_PLACEHOLDER_DATABASE_ID =
  '00000000-0000-4000-8000-000000000000';

const { d1, r2 } = hostingConfig;

// macOS Seatbelt blocks FSEvents, so Codex previews need polling for HMR.
const isCodexSeatbeltSandbox = process.env.CODEX_SANDBOX === 'seatbelt';

const localBindingConfig = {
  main: 'vinext/server/app-router-entry',
  compatibility_flags: ['nodejs_compat'],
  d1_databases: d1
    ? [
        {
          binding: d1,
          database_name: 'site-creator-d1',
          database_id: SITE_CREATOR_PLACEHOLDER_DATABASE_ID,
        },
      ]
    : [],
  r2_buckets: r2
    ? [
        {
          binding: r2,
          bucket_name: 'site-creator-r2',
        },
      ]
    : [],
};

export default defineConfig(async () => {
  // Keep Wrangler and Miniflare state project-local. These are non-secret tool
  // settings; application environment belongs in ignored `.env*` files.
  process.env.WRANGLER_WRITE_LOGS ??= 'false';
  process.env.WRANGLER_LOG_PATH ??= '.wrangler/logs';
  process.env.MINIFLARE_REGISTRY_PATH ??= '.wrangler/registry';

  // Wrangler snapshots its log path while the Cloudflare plugin is imported.
  const { cloudflare } = await import('@cloudflare/vite-plugin');

  return {
    css: { postcss: { plugins: [tailwindcss()] } },
    // RSC/SSR/client 환경이 각각 React 사본을 해석하면 HMR 도중 같은
    // Context Provider를 여러 렌더러가 동시에 잡는 경고가 발생한다.
    resolve: {
      dedupe: ['react', 'react-dom', 'react-server-dom-webpack'],
    },
    server: {
      host: '127.0.0.1',
      port: 3000,
      strictPort: true,
      // Vite 8 can automatically forward browser errors when it detects an
      // AI coding environment. During the initial HMR connection that error
      // forwarding can try to send before the socket is ready and create the
      // misleading "send was called before connect" rejection.
      forwardConsole: false,
      ...(isCodexSeatbeltSandbox
        ? { watch: { useFsEvents: false, usePolling: true } }
        : {}),
    },
    plugins: [
      {
        name: 'ajindie-rsc-safe-reload',
        handleHotUpdate(context: HmrContext) {
          // Vinext의 RSC 엔트리를 부분 교체하면 이전 client renderer가
          // 남아 Context 충돌이 난다. 앱 소스는 안전한 전체 reload로 바꾼다.
          if (/[\\/]app[\\/].*\.(?:ts|tsx|css)$/.test(context.file)) {
            context.server.ws.send({ type: 'full-reload' });
            return [];
          }
        },
      },
      vinext(),
      sites(),
      cloudflare({
        viteEnvironment: { name: 'rsc', childEnvironments: ['ssr'] },
        config: localBindingConfig,
      }),
    ],
  };
});
