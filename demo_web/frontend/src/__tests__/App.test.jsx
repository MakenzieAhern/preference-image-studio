// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import App, { Compare } from '../App.jsx'

const healthy = {cnn_ready:true, cuda_available:false}
const result = (url = '/outputs/enhanced.png') => ({enhanced_image_url:url, total_processing_seconds:.12})
const response = (data, status = 200) => ({ok:status >= 200 && status < 300, status, json:async () => data})
const deferred = () => {
  let resolve, reject
  const promise = new Promise((yes, no) => { resolve = yes; reject = no })
  return {promise, resolve, reject}
}
const regions = {segment_id:'source-hash', dimensions:[100,50], regions:[
  {region_id:2, label:'sky', coverage:.6, overlay_url:'/outputs/sky.png'},
  {region_id:3, label:'tiny', coverage:.001, overlay_url:'/outputs/tiny.png'},
]}
let fetchMock
let blobNumber

function upload(name = 'source.png', kind = 'source') {
  const label = kind === 'source' ? '01 · ORIGINAL IMAGE file' : 'REFERENCE IMAGE file'
  const file = new File(['image contents'], name, {type:/tiff?$/.test(name) ? 'image/tiff' : 'image/png'})
  fireEvent.change(screen.getByLabelText(label), {target:{files:[file]}})
  return file
}
function apply() { fireEvent.click(screen.getByRole('button', {name:/^Apply Enhancement/})) }
function requests(endpoint) { return fetchMock.mock.calls.filter(([url]) => url === endpoint) }
async function settle() { await act(async () => {}) }

beforeEach(() => {
  blobNumber = 0
  vi.stubGlobal('Image', class {
    naturalWidth = 100
    naturalHeight = 50
    set src(value) { this.url = value; queueMicrotask(() => this.onload?.()) }
  })
  URL.createObjectURL = vi.fn(() => `blob:image-${++blobNumber}`)
  URL.revokeObjectURL = vi.fn()
  fetchMock = vi.fn(async url => {
    if (url === '/api/health') return response(healthy)
    if (url === '/api/enhance/manual') return response(result())
    throw new Error(`Unexpected request: ${url}`)
  })
  vi.stubGlobal('fetch', fetchMock)
})
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

describe('manual-first workflow', () => {
  it('submits a manual request without automatically requesting segmentation', async () => {
    render(<App />)
    const file = upload()
    await settle()
    expect(screen.getByRole('button', {name:'Fine Tune'}).className).toBe('active')
    expect(requests('/api/segment')).toHaveLength(0)
    apply()
    await screen.findByAltText('Enhanced result')
    const form = requests('/api/enhance/manual')[0][1].body
    expect(form.get('image')).toBe(file)
    expect(form.get('strength')).toBe('1')
    expect(requests('/api/segment')).toHaveLength(0)
  })

  it('detects regions only after the explicit action and hides negligible regions', async () => {
    fetchMock.mockImplementation(async url => response(url === '/api/segment' ? regions : healthy))
    render(<App />)
    expect(screen.getByRole('button', {name:'Detect Regions'}).disabled).toBe(true)
    upload()
    await settle()
    fireEvent.click(screen.getByRole('button', {name:'Detect Regions'}))
    await screen.findByRole('button', {name:/sky\s*60\s*%/})
    expect(requests('/api/segment')).toHaveLength(1)
    expect(screen.queryByRole('button', {name:/tiny/})).toBeNull()
  })

  it('maps clicks through image letterboxing and ignores padding', async () => {
    fetchMock.mockImplementation(async url => {
      if (url === '/api/segment') return response(regions)
      if (url.includes('/region?')) return response({region:{region_id:2,label:'sky'}})
      return response(healthy)
    })
    render(<App />)
    upload()
    await settle()
    fireEvent.click(screen.getByRole('button', {name:'Detect Regions'}))
    await screen.findByRole('button', {name:/sky\s*60\s*%/})
    const preview = screen.getByAltText('01 · ORIGINAL IMAGE preview').parentElement
    preview.getBoundingClientRect = () => ({left:0,top:0,width:400,height:400})
    fireEvent.click(preview, {clientX:200,clientY:50})
    expect(fetchMock.mock.calls.some(([url]) => url.includes('/region?'))).toBe(false)
    fireEvent.click(preview, {clientX:200,clientY:200})
    await screen.findByText('sky selected from image')
    expect(requests('/api/segment/source-hash/region?x=50&y=25')).toHaveLength(1)
    expect(screen.getByRole('button', {name:/sky\s*60\s*%/}).className).toBe('active')
  })
})

describe('reference and text handoff', () => {
  it('requests global reference matching by default without detecting regions', async () => {
    fetchMock.mockImplementation(async url => response(url === '/api/enhance/reference' ? {
      ...result(), mode_used:'global-fallback', matched_regions:[],
      final_global_adjustments:{red:.1,green:0,blue:-.1,contrast:0},
    } : healthy))
    render(<App />)
    const source = upload()
    await settle()
    fireEvent.click(screen.getByRole('button', {name:/Reference Image/}))
    const reference = upload('reference.png', 'reference')
    await settle()
    fireEvent.click(screen.getByRole('button', {name:/^Apply Reference Style/}))
    await screen.findByAltText('Enhanced result')
    const form = requests('/api/enhance/reference')[0][1].body
    expect(form.get('source')).toBe(source)
    expect(form.get('reference')).toBe(reference)
    expect(form.get('strength')).toBe('0.7')
    expect(form.get('region_aware')).toBe('false')
    expect(requests('/api/segment')).toHaveLength(0)
  })

  it('hands text-plan values to regional fine tune with unit strength', async () => {
    fetchMock.mockImplementation(async url => response(
      url === '/api/enhance/text' ? {...result(), region_json:{global:{r:.05,contrast:.15},sky:{b:.1}}} :
      url === '/api/segment' ? regions :
      url === '/api/enhance/manual-regions' ? result('/outputs/refined.png') : healthy
    ))
    render(<App />)
    upload()
    await settle()
    fireEvent.change(screen.getByRole('slider', {name:/Overall Strength/}), {target:{value:'.2'}})
    fireEvent.click(screen.getByRole('button', {name:'Text Preference'}))
    fireEvent.click(screen.getByRole('button', {name:/^Enhance Image/}))
    await screen.findByAltText('Enhanced result')
    fireEvent.click(screen.getByRole('button', {name:/Fine Tune AI Result/}))
    await screen.findByRole('button', {name:/sky\s*60\s*%/})
    expect(screen.getByRole('slider', {name:/Overall Strength/}).value).toBe('1')
    expect(screen.getByRole('slider', {name:/Contrast/}).value).toBe('0.15')
    apply()
    await settle()
    const form = requests('/api/enhance/manual-regions')[0][1].body
    expect(form.get('global_red')).toBe('0.05')
    expect(form.get('global_contrast')).toBe('0.15')
    expect(form.get('global_strength')).toBe('1')
    expect(JSON.parse(form.get('regions_json'))).toEqual([{region_id:'2',r:0,g:0,b:.1,contrast:0,strength:1}])
  })
})

describe('out-of-order work and replacement', () => {
  it('ignores a TIFF preview that completes after a new source was selected', async () => {
    const pending = deferred()
    fetchMock.mockImplementation(url => url === '/api/preview' ? pending.promise : Promise.resolve(response(healthy)))
    render(<App />)
    upload('older.tiff')
    upload('newer.png')
    await settle()
    await act(async () => pending.resolve(response({enhanced_image_url:'/outputs/old-preview.jpg',dimensions:[200,400]})))
    expect(screen.getByAltText('01 · ORIGINAL IMAGE preview').getAttribute('src')).toBe('blob:image-1')
    expect(screen.getByText('newer.png')).toBeTruthy()
    expect(screen.queryByText('200 × 400')).toBeNull()
    expect(requests('/api/preview')[0][1].signal.aborted).toBe(true)
  })

  it('discards enhancement results for a replaced source, even if the server finishes later', async () => {
    const old = deferred()
    let enhancement = 0
    fetchMock.mockImplementation(url => {
      if (url === '/api/enhance/manual') return ++enhancement === 1 ? old.promise : Promise.resolve(response(result('/outputs/new.png')))
      return Promise.resolve(response(healthy))
    })
    render(<App />)
    upload('old.png')
    await settle()
    apply()
    upload('new.png')
    await settle()
    apply()
    await screen.findByAltText('Enhanced result')
    await act(async () => old.resolve(response(result('/outputs/old.png'))))
    expect(screen.getByAltText('Enhanced result').getAttribute('src')).toBe('/outputs/new.png')
    expect(requests('/api/enhance/manual')[0][1].signal.aborted).toBe(true)
  })

  it('does not attach an old segmentation to a new image', async () => {
    const pending = deferred()
    fetchMock.mockImplementation(url => url === '/api/segment' ? pending.promise : Promise.resolve(response(healthy)))
    render(<App />)
    upload('old.png')
    await settle()
    fireEvent.click(screen.getByRole('button', {name:'Detect Regions'}))
    upload('new.png')
    await act(async () => pending.resolve(response(regions)))
    expect(screen.queryByRole('button', {name:/sky\s*60\s*%/})).toBeNull()
    expect(screen.getByRole('button', {name:'Detect Regions'}).disabled).toBe(false)
    expect(requests('/api/segment')[0][1].signal.aborted).toBe(true)
  })

  it('cancels in-flight enhancement when its control mode is changed', async () => {
    const pending = deferred()
    fetchMock.mockImplementation(url => url === '/api/enhance/manual' ? pending.promise : Promise.resolve(response(healthy)))
    render(<App />)
    upload()
    await settle()
    apply()
    fireEvent.click(screen.getByRole('button', {name:'Text Preference'}))
    await act(async () => pending.resolve(response(result())))
    expect(screen.queryByAltText('Enhanced result')).toBeNull()
    expect(requests('/api/enhance/manual')[0][1].signal.aborted).toBe(true)
    expect(screen.getByRole('button', {name:/^Enhance Image/}).disabled).toBe(false)
  })

  it('replacing a reference revokes only that reference blob, not the original', async () => {
    render(<App />)
    upload()
    await settle()
    fireEvent.click(screen.getByRole('button', {name:/Reference Image/}))
    upload('reference.png', 'reference')
    await settle()
    upload('reference-2.png', 'reference')
    await settle()
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:image-2')
    expect(URL.revokeObjectURL).not.toHaveBeenCalledWith('blob:image-1')
    expect(screen.getByAltText('01 · ORIGINAL IMAGE preview').getAttribute('src')).toBe('blob:image-1')
  })
})

describe('errors and reset', () => {
  it('reports a TIFF network error and clears the unusable selection', async () => {
    fetchMock.mockImplementation(async url => {
      if (url === '/api/preview') throw new TypeError('Failed to fetch')
      return response(healthy)
    })
    render(<App />)
    upload('source.tiff')
    expect((await screen.findByRole('alert')).textContent).toContain('Preview failed')
    expect(screen.queryByAltText('01 · ORIGINAL IMAGE preview')).toBeNull()
    apply()
    expect(screen.getByRole('alert').textContent).toContain('Please upload an image first')
  })

  it('keeps a successful result when the following health check fails', async () => {
    let healthRequests = 0
    fetchMock.mockImplementation(async url => {
      if (url === '/api/health' && ++healthRequests > 1) throw new TypeError('Failed to fetch')
      return response(url === '/api/health' ? healthy : result())
    })
    render(<App />)
    upload()
    await settle()
    apply()
    await screen.findByAltText('Enhanced result')
    await settle()
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('renders structured API validation errors as readable text without crashing', async () => {
    fetchMock.mockImplementation(async url => url === '/api/health' ? response(healthy) : response({detail:[{msg:'invalid'}]},422))
    render(<App />)
    upload()
    await settle()
    apply()
    expect((await screen.findByRole('alert')).textContent).toContain('Enhancement failed. Please try again.')
  })

  it('resets strengths, reference options, prompt, and all image/result state', async () => {
    render(<App />)
    upload()
    await settle()
    fireEvent.change(screen.getByRole('slider', {name:/Overall Strength/}), {target:{value:'.25'}})
    fireEvent.click(screen.getByRole('button', {name:/Reference Image/}))
    fireEvent.change(screen.getByRole('slider', {name:/Reference Strength/}), {target:{value:'.2'}})
    fireEvent.click(screen.getByRole('checkbox', {name:/Semantic region matching/}))
    fireEvent.click(screen.getByRole('button', {name:'Text Preference'}))
    fireEvent.change(screen.getByRole('textbox'), {target:{value:'A different prompt'}})
    fireEvent.click(screen.getByRole('button', {name:'Fine Tune'}))
    apply()
    await screen.findByAltText('Enhanced result')
    fireEvent.click(screen.getByRole('button', {name:'Reset',exact:true}))
    expect(screen.queryByAltText('Enhanced result')).toBeNull()
    expect(screen.queryByAltText('01 · ORIGINAL IMAGE preview')).toBeNull()
    expect(screen.getByRole('slider', {name:/Overall Strength/}).value).toBe('1')
    fireEvent.click(screen.getByRole('button', {name:/Reference Image/}))
    expect(screen.getByRole('slider', {name:/Reference Strength/}).value).toBe('0.7')
    expect(screen.getByRole('checkbox', {name:/Semantic region matching/}).checked).toBe(false)
    fireEvent.click(screen.getByRole('button', {name:'Text Preference'}))
    expect(screen.getByRole('textbox').value).toBe('Make the image slightly warmer.')
  })
})

it('moves the enhanced clipping edge with the comparison slider while retaining image labels', () => {
  render(<Compare before="original.png" after="enhanced.png" />)
  fireEvent.change(screen.getByRole('slider', {name:'Before and after position'}), {target:{value:'80'}})
  expect(screen.getByAltText('Enhanced').parentElement.style.clipPath).toBe('inset(0 0 0 80%)')
  expect(screen.getByAltText('Original').getAttribute('src')).toBe('original.png')
  expect(screen.getByText('Original')).toBeTruthy()
  expect(screen.getByText('Enhanced')).toBeTruthy()
})
