import assert from 'node:assert/strict'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'

const server = await createServer({ server: { middlewareMode: true } })
try {
  const { Compare } = await server.ssrLoadModule('/src/App.jsx')
  const html = renderToStaticMarkup(createElement(Compare, { before: 'original.png', after: 'enhanced.png' }))
  assert.match(html, /<img src="original.png" alt="Original"/)
  // Clip the enhanced layer's LEFT half: original stays left, enhanced stays right.
  assert.match(html, /class="after" style="clip-path:inset\(0 0 0 50%\)"><img src="enhanced.png" alt="Enhanced"/)
  assert.match(html, /compare-label left">Original/)
  assert.match(html, /compare-label right">Enhanced/)
  console.log('Before/After image-layer and label checks passed')
} finally {
  await server.close()
}
