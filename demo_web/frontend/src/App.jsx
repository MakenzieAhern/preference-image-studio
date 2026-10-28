import { useEffect, useRef, useState } from 'react'

const prompts = [
  'Make the image slightly warmer.',
  'Brighten the image slightly.',
  'Increase contrast while keeping the image natural.'
]
const channels = [['red', 'Red'], ['green', 'Green'], ['blue', 'Blue'], ['contrast', 'Contrast']]
const emptyControl = () => ({red:0, green:0, blue:0, contrast:0, strength:1})

function Upload({ label, file, preview, dimensions, onFile, onRemove, compact = false, overlay = '', onPreviewClick }) {
  const [dragging, setDragging] = useState(false)
  const accept = file => file && onFile(file)
  return <section className={`upload ${dragging ? 'dragging' : ''} ${compact ? 'compact' : ''}`}
    onDragOver={e => { e.preventDefault(); setDragging(true) }}
    onDragLeave={() => setDragging(false)}
    onDrop={e => { e.preventDefault(); setDragging(false); accept(e.dataTransfer.files[0]) }}>
    <div className="eyebrow">{label}</div>
    {preview ? <>
      <div className={`preview-stack ${onPreviewClick?'selectable':''}`} onClick={onPreviewClick}><img src={preview} alt={`${label} preview`} />{overlay&&<img className="region-overlay" src={overlay} alt="Selected region highlight"/>}</div>
      <div className="fileline"><span>{file?.name}</span><span>{dimensions || 'Reading dimensions…'}</span></div>
      <div className="row"><label className="text-button">Replace<input type="file" aria-label={`${label} file`} accept=".png,.jpg,.jpeg,.tif,.tiff" onChange={e => accept(e.target.files[0])}/></label><button className="text-button" onClick={onRemove}>Remove</button></div>
    </> : <label className="drop"><span className="upload-icon">↥</span><strong>Drop an image here</strong><small>or click to browse · PNG, JPG, TIFF</small><input type="file" aria-label={`${label} file`} accept=".png,.jpg,.jpeg,.tif,.tiff" onChange={e => accept(e.target.files[0])}/></label>}
  </section>
}

function Slider({ label, value, onChange, min = -.3, max = .3, step = .01, percent = false }) {
  const shown = Number(value)
  return <label className="slider-row"><span>{label}<b>{percent ? `${Math.round(shown * 100)}%` : `${shown > 0 ? '+' : ''}${shown.toFixed(2)}`}</b></span>
    <input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(Number(e.target.value))}/>
  </label>
}

export function Compare({ before, after }) {
  const [position, setPosition] = useState(50)
  if (!before || !after) return null
  return <div className="compare">
    <img src={before} alt="Original" />
    <div className="after" style={{clipPath:`inset(0 0 0 ${position}%)`}}><img src={after} alt="Enhanced" /></div>
    <div className="divider" style={{left:`${position}%`}}><span>↔</span></div>
    <input aria-label="Before and after position" type="range" min="0" max="100" value={position} onChange={e => setPosition(e.target.value)}/>
    <span className="compare-label left">Original</span><span className="compare-label right">Enhanced</span>
  </div>
}

function Interpretation({ result }) {
  if (!result) return <div className="empty-note">Your parsed enhancement plan will appear here.</div>
  const entries = Object.entries(result.region_json || {}).filter(([, values]) => Object.values(values).some(v => v))
  if (!entries.length) return <div className="empty-note">No non-zero regional controls were returned.</div>
  const describe = values => {
    const notes = [], brightness = ((values.r||0)+(values.g||0)+(values.b||0))/3
    if (brightness > .025) notes.push('Brighter'); if (brightness < -.025) notes.push('Darker')
    if ((values.r||0)-(values.b||0) > .05) notes.push('Warmer'); if ((values.b||0)-(values.r||0) > .05) notes.push('Cooler')
    if ((values.contrast||0) > 0) notes.push('Contrast increased'); if ((values.contrast||0) < 0) notes.push('Contrast softened')
    return notes.length ? notes.join(' · ') : 'Colour adjusted'
  }
  return <div className="interpretation friendly">{entries.map(([region, values]) => <div key={region}><strong>{region}</strong><span>{describe(values)}</span></div>)}<details><summary>Actual RGBC values</summary>{entries.map(([region, values])=><p key={region}>{region}: {Object.entries(values).filter(([,v])=>v).map(([k,v])=>`${k.toUpperCase()} ${v>0?'+':''}${v.toFixed(2)}`).join(' · ')}</p>)}</details></div>
}

async function responseData(response, fallback) {
  let data
  try { data = await response.json() }
  catch { throw new Error(fallback) }
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : fallback)
  return data
}

export default function App() {
  const [health, setHealth] = useState(null)
  const [tab, setTab] = useState('manual')
  const [source, setSource] = useState(null)
  const [sourcePreview, setSourcePreview] = useState('')
  const [sourceSize, setSourceSize] = useState('')
  const [reference, setReference] = useState(null)
  const [referencePreview, setReferencePreview] = useState('')
  const [referenceSize, setReferenceSize] = useState('')
  const [prompt, setPrompt] = useState(prompts[0])
  const [referenceStrength, setReferenceStrength] = useState(.7)
  const [regionAware, setRegionAware] = useState(false)
  const [manual, setManual] = useState({red:0, green:0, blue:0, contrast:0})
  const [manualStrength, setManualStrength] = useState(1)
  const [segmentation, setSegmentation] = useState(null)
  const [segmentationLoading, setSegmentationLoading] = useState(false)
  const [selectedScope, setSelectedScope] = useState('global')
  const [regional, setRegional] = useState({})
  const [handoff, setHandoff] = useState(false)
  const [regionMessage, setRegionMessage] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState('')
  const jobs = useRef({})

  function cancelJob(name) {
    jobs.current[name]?.abort()
    delete jobs.current[name]
  }
  function beginJob(name) {
    cancelJob(name)
    const controller = new AbortController()
    jobs.current[name] = controller
    return controller
  }
  function isCurrent(name, controller) {
    return jobs.current[name] === controller && !controller.signal.aborted
  }
  function invalidateResult() {
    cancelJob('enhance')
    setLoading(false)
    setElapsed(0)
    setResult(null)
    setHandoff(false)
  }
  function clearRegions() {
    cancelJob('segment')
    cancelJob('region')
    setSegmentation(null)
    setSegmentationLoading(false)
    setRegional({})
    setSelectedScope('global')
    setRegionMessage('')
  }

  useEffect(() => {
    const controller = new AbortController()
    fetch('/api/health', {signal: controller.signal})
      .then(response => responseData(response, 'Backend is unavailable.'))
      .then(setHealth).catch(() => {})
    return () => controller.abort()
  }, [])
  useEffect(() => () => Object.values(jobs.current).forEach(controller => controller.abort()), [])
  useEffect(() => {
    if (!loading) return
    const started = Date.now(), timer = setInterval(() => setElapsed((Date.now() - started) / 1000), 100)
    return () => clearInterval(timer)
  }, [loading])
  useEffect(() => () => { if (sourcePreview.startsWith('blob:')) URL.revokeObjectURL(sourcePreview) }, [sourcePreview])
  useEffect(() => () => { if (referencePreview.startsWith('blob:')) URL.revokeObjectURL(referencePreview) }, [referencePreview])
  useEffect(() => {
    if (!handoff || !segmentation || result?.method !== 'text') return
    const plan = result.region_json || {}, global = plan.global || {}
    setManual({red:global.r||0, green:global.g||0, blue:global.b||0, contrast:global.contrast||0})
    setManualStrength(1)
    setRegional(Object.fromEntries(segmentation.regions.filter(region => region.coverage >= .005).map(region => {
      const values = plan[region.label] || {}
      return [region.region_id, {red:values.r||0, green:values.g||0, blue:values.b||0, contrast:values.contrast||0, strength:1}]
    })))
    setHandoff(false)
  }, [handoff, segmentation, result])

  async function selectFile(file, kind) {
    if (!/\.(png|jpe?g|tiff?)$/i.test(file.name)) return setError('Please upload a PNG, JPG/JPEG, or TIF/TIFF image.')
    const controller = beginJob(kind)
    cancelJob('demo')
    invalidateResult()
    setError('')
    const setFile = kind === 'source' ? setSource : setReference
    const setPreview = kind === 'source' ? setSourcePreview : setReferencePreview
    const setSize = kind === 'source' ? setSourceSize : setReferenceSize
    setFile(file)
    setPreview('')
    setSize('')
    if (kind === 'source') clearRegions()
    try {
      if (/\.tiff?$/i.test(file.name)) {
        const form = new FormData()
        form.append('image', file)
        const response = await fetch('/api/preview', {method:'POST', body:form, signal:controller.signal})
        const data = await responseData(response, 'This TIFF image could not be previewed.')
        if (!isCurrent(kind, controller)) return
        setPreview(data.enhanced_image_url)
        setSize(data.dimensions.join(' × '))
      } else {
        const url = URL.createObjectURL(file)
        setPreview(url)
        const image = new Image()
        image.onload = () => {
          if (isCurrent(kind, controller)) setSize(`${image.naturalWidth} × ${image.naturalHeight}`)
        }
        image.onerror = () => {
          if (!isCurrent(kind, controller)) return
          setFile(null)
          setPreview('')
          setSize('')
          setError('This image could not be decoded. Please choose a valid image.')
        }
        image.src = url
      }
    } catch (failure) {
      if (!isCurrent(kind, controller)) return
      setFile(null)
      setPreview('')
      setSize('')
      setError(failure instanceof TypeError ? 'Preview failed. Check the backend connection and try again.' : failure.message)
    }
  }

  function removeFile(kind) {
    cancelJob(kind)
    cancelJob('demo')
    invalidateResult()
    setError('')
    if (kind === 'source') {
      setSource(null); setSourcePreview(''); setSourceSize(''); clearRegions()
    } else {
      setReference(null); setReferencePreview(''); setReferenceSize('')
    }
  }

  async function tryDemo() {
    const controller = beginJob('demo')
    setError('')
    try {
      const response = await fetch('/examples/demo-coast.png', {signal:controller.signal})
      if (!response.ok) throw new Error('The demo image could not be loaded. Please upload an image.')
      const blob = await response.blob()
      if (!isCurrent('demo', controller)) return
      await selectFile(new File([blob], 'demo-coast.png', {type:'image/png'}), 'source')
    } catch (failure) {
      if (isCurrent('demo', controller)) setError('The demo image could not be loaded. Please upload an image.')
    }
  }

  async function segmentSource() {
    if (!source || segmentationLoading) return
    const controller = beginJob('segment')
    setSegmentationLoading(true)
    setError('')
    const form = new FormData()
    form.append('image', source)
    try {
      const response = await fetch('/api/segment', {method:'POST', body:form, signal:controller.signal})
      const data = await responseData(response, 'Region detection failed. Please try again.')
      if (!isCurrent('segment', controller)) return
      setSegmentation(data)
      setRegionMessage(`Regions ready · ${data.regions.filter(region => region.coverage >= .005).length} editable`)
    } catch (failure) {
      if (isCurrent('segment', controller)) setError(failure instanceof TypeError ? 'Region detection failed. Check the backend connection.' : failure.message)
    } finally {
      if (isCurrent('segment', controller)) setSegmentationLoading(false)
    }
  }

  const visibleRegions = segmentation?.regions.filter(region => region.coverage >= .005) || []
  const selectedControl = selectedScope === 'global' ? {...manual, strength:manualStrength} : (regional[selectedScope] || emptyControl())
  const selectedOverlay = selectedScope === 'global' ? '' : segmentation?.regions.find(region => String(region.region_id) === String(selectedScope))?.overlay_url || ''

  function updateSelected(key, value) {
    if (selectedScope === 'global') {
      if (key === 'strength') setManualStrength(value)
      else setManual(current => ({...current, [key]:value}))
    } else setRegional(current => ({...current, [selectedScope]:{...(current[selectedScope] || emptyControl()), [key]:value}}))
  }
  function resetRegion() {
    if (selectedScope === 'global') { setManual({red:0,green:0,blue:0,contrast:0}); setManualStrength(1) }
    else setRegional(current => ({...current, [selectedScope]:emptyControl()}))
  }
  function resetAllControls() {
    setManual({red:0,green:0,blue:0,contrast:0}); setManualStrength(1); setRegional({})
  }
  function fineTuneAiPlan() {
    setHandoff(true)
    setSelectedScope('global')
    setTab('manual')
    if (!segmentation) segmentSource()
  }
  function changeTab(next) {
    if (next === tab) return
    invalidateResult()
    setError('')
    setTab(next)
  }

  async function clickRegion(event) {
    if (!segmentation) return
    const rect = event.currentTarget.getBoundingClientRect(), [imageWidth,imageHeight] = segmentation.dimensions
    const scale = Math.min(rect.width / imageWidth, rect.height / imageHeight)
    const left = (rect.width - imageWidth * scale) / 2, top = (rect.height - imageHeight * scale) / 2
    const x = (event.clientX - rect.left - left) / scale, y = (event.clientY - rect.top - top) / scale
    if (x < 0 || y < 0 || x >= imageWidth || y >= imageHeight) return setRegionMessage('Click inside the visible photo to select a region.')
    const controller = beginJob('region')
    try {
      const response = await fetch(`/api/segment/${segmentation.segment_id}/region?x=${Math.floor(x)}&y=${Math.floor(y)}`, {signal:controller.signal})
      const data = await responseData(response, 'Region selection is temporarily unavailable.')
      if (!isCurrent('region', controller)) return
      const match = visibleRegions.find(region => region.region_id === data.region?.region_id)
      if (match) { setSelectedScope(String(match.region_id)); setRegionMessage(`${match.label} selected from image`) }
      else setRegionMessage('That point is not a meaningful editable region; selection unchanged.')
    } catch {
      if (isCurrent('region', controller)) setRegionMessage('Region selection is temporarily unavailable.')
    }
  }

  const stage = tab === 'text' ? 'Processing text preference…' : tab === 'reference' ? 'Applying reference appearance…' : 'Applying fine tune…'

  async function enhance() {
    if (loading) return
    if (!source) return setError('Please upload an image first.')
    if (!sourcePreview) return setError('Please wait for the image preview to finish.')
    if (tab === 'reference' && (!reference || !referencePreview)) return setError('Please upload a reference image and wait for its preview.')
    if (tab === 'text' && !prompt.trim()) return setError('Please describe your preference first.')
    const controller = beginJob('enhance')
    const method = tab
    setError(''); setLoading(true); setElapsed(0)
    const form = new FormData()
    let endpoint = `/api/enhance/${method === 'manual' ? 'manual' : method}`
    form.append(method === 'reference' ? 'source' : 'image', source)
    if (method === 'text') form.append('preference', prompt)
    if (method === 'reference') { form.append('reference', reference); form.append('strength', referenceStrength); form.append('region_aware', regionAware) }
    if (method === 'manual' && segmentation) {
      endpoint = '/api/enhance/manual-regions'
      form.append('segment_id', segmentation.segment_id)
      Object.entries(manual).forEach(([key,value]) => form.append(`global_${key}`,value))
      form.append('global_strength', manualStrength)
      form.append('regions_json', JSON.stringify(Object.entries(regional).map(([region_id, control]) => ({region_id,r:control.red,g:control.green,b:control.blue,contrast:control.contrast,strength:control.strength}))))
    } else if (method === 'manual') { Object.entries(manual).forEach(([key,value]) => form.append(key,value)); form.append('strength', manualStrength) }
    try {
      const response = await fetch(endpoint, {method:'POST', body:form, signal:controller.signal})
      const data = await responseData(response, 'Enhancement failed. Please try again.')
      if (!isCurrent('enhance', controller)) return
      setResult({...data, method})
      // A later health-check outage must not turn a successful enhancement into an error.
      fetch('/api/health').then(response => responseData(response, 'Health check failed.')).then(data => {
        if (isCurrent('enhance', controller)) setHealth(data)
      }).catch(() => {})
    } catch (failure) {
      if (isCurrent('enhance', controller)) setError(failure instanceof TypeError ? 'Backend is unavailable. Start the demo and try again.' : failure.message)
    } finally {
      if (isCurrent('enhance', controller)) setLoading(false)
    }
  }

  function reset() {
    Object.keys(jobs.current).forEach(cancelJob)
    setSource(null); setSourcePreview(''); setSourceSize('')
    setReference(null); setReferencePreview(''); setReferenceSize('')
    invalidateResult(); clearRegions(); resetAllControls()
    setReferenceStrength(.7); setRegionAware(false); setPrompt(prompts[0]); setError(''); setTab('manual')
  }

  return <main>
    <header><div className="brand"><span className="mark">P</span><div><strong>Preference Image Studio</strong><small>Your image. Your preference. Your style.</small></div></div>
      <div className="status"><span className={health?.cnn_ready?'ok':''}>● Model {health?.cnn_ready?'Ready':'Offline'}</span><span className={health?.cuda_available?'ok':''}>● {health ? (health.cuda_available?'GPU Ready':'CPU Mode') : 'Backend unavailable'}</span></div>
    </header>

    <section className="hero"><div><span className="kicker">INTELLIGENT PHOTO WORKSPACE</span><h1>Shape the image<br/><em>you imagined.</em></h1></div><div className="hero-copy"><p>Natural language, visual references, or precise manual control—each path drives the same real preference-conditioned CNN.</p><button className="text-button" onClick={tryDemo}>Try Demo Image</button></div></section>

    <section className="workspace">
      <Upload label="01 · ORIGINAL IMAGE" file={source} preview={sourcePreview} dimensions={sourceSize} overlay={tab==='manual'?selectedOverlay:''} onPreviewClick={tab==='manual'?clickRegion:null} onFile={f=>selectFile(f,'source')} onRemove={()=>removeFile('source')}/>
      <section className="controls">
        <div className="tabs">{[['text','Text Preference'],['reference','Reference Image'],['manual','Fine Tune']].map(([id,label])=><button className={tab===id?'active':''} onClick={()=>changeTab(id)} key={id}>{label}{id==='reference'&&<i>Experimental</i>}</button>)}</div>
        {tab === 'text' && <div className="panel"><label className="field-label" htmlFor="preference">Describe your preference</label><p className="model-note">Text mode uses optional language and segmentation models. The first run may download model weights.</p><textarea id="preference" value={prompt} onChange={e=>setPrompt(e.target.value)} placeholder="Make the image slightly warmer."/><div className="chips">{prompts.map(p=><button onClick={()=>setPrompt(p)} key={p}>{p}</button>)}</div><div className="analysis"><div className="field-label">AI Interpretation</div><Interpretation result={result?.method==='text'?result:null}/>{result?.method==='text'&&<><p className="handoff-note">AI created the initial plan. Refine individual regions yourself.</p><button className="subtle handoff" onClick={fineTuneAiPlan}>Fine Tune AI Result →</button></>}</div></div>}
        {tab === 'reference' && <div className="panel"><Upload compact label="REFERENCE IMAGE" file={reference} preview={referencePreview} dimensions={referenceSize} onFile={f=>selectFile(f,'reference')} onRemove={()=>removeFile('reference')}/><Slider label="Reference Strength" value={referenceStrength} onChange={setReferenceStrength} min={0} max={1} step={.01} percent/><label className="check"><input type="checkbox" checked={regionAware} onChange={e=>setRegionAware(e.target.checked)}/> Semantic region matching (loads segmentation model)</label>{result?.method==='reference'&&<div className="mini-reference"><b>{result.mode_used==='region-aware'?'Region-aware':'Global fallback'}</b>{channels.map(([key,label])=><span key={key}>{label} {result.final_global_adjustments[key]>0?'+':''}{result.final_global_adjustments[key].toFixed(3)}</span>)}<small>Matched: {result.matched_regions.map(x=>x.name).join(', ')||'none'}</small></div>}<p className="honesty" title="Reference mode converts reference-image appearance statistics and semantic region matches into the CNN conditioning space.">Uses robust image statistics and optional semantic-region matching to convert appearance into the CNN’s real R/G/B/Contrast conditioning space.</p></div>}
        {tab === 'manual' && <div className="panel manual"><div className="scope-head"><span className="field-label">Edit scope</span>{segmentation&&<span className="regions-ready">● Regions ready</span>}</div><div className="scope-chips"><button className={selectedScope==='global'?'active':''} onClick={()=>setSelectedScope('global')}>Global</button>{visibleRegions.map(region=><button className={String(selectedScope)===String(region.region_id)?'active':''} key={region.region_id} onClick={()=>{setSelectedScope(String(region.region_id));setRegionMessage('')}}>{region.label}<small>{(region.coverage*100).toFixed(0)}%</small></button>)}</div>{!segmentation&&<div className="region-opt-in"><button className="subtle" disabled={!source||segmentationLoading||loading} onClick={segmentSource}>{segmentationLoading?'Detecting regions…':'Detect Regions'}</button><p className="model-note">Global controls work without region detection. Detecting regions loads the optional segmentation model.</p></div>}{(segmentationLoading||regionMessage)&&<div className="region-message" role="status">{segmentationLoading?'Detecting semantic regions…':regionMessage}</div>}<div className="selected-region">Selected Region <b>{selectedScope==='global'?'Global':segmentation?.regions.find(r=>String(r.region_id)===String(selectedScope))?.label}</b></div>{channels.map(([key,label])=><Slider key={key} label={label} value={selectedControl[key]} onChange={value=>updateSelected(key,value)}/>)}<Slider label={selectedScope==='global'?'Overall Strength':'Region Strength'} value={selectedControl.strength} onChange={value=>updateSelected('strength',value)} min={0} max={1} step={.01} percent/><div className="row"><button className="subtle" onClick={resetRegion}>Reset Region</button><button className="subtle" onClick={resetAllControls}>Reset All</button></div><div className="region-summary"><span className="field-label">Enhancement Plan</span>{[['global',{...manual,strength:manualStrength}],...Object.entries(regional)].map(([id,control])=>{const changes=channels.filter(([key])=>control[key]).map(([key,label])=>`${label} ${control[key]>0?'+':''}${control[key].toFixed(2)}`);if(!changes.length)return null;const name=id==='global'?'Global':segmentation?.regions.find(r=>String(r.region_id)===String(id))?.label;return <div key={id}><b>{name}</b><span>{changes.join(' · ')}</span></div>})}</div></div>}
        {error && <div className="error" role="alert">{error}</div>}
        <button className="enhance" disabled={loading||segmentationLoading} onClick={enhance}>{loading?<><span className="spinner"/>{stage} <b>{elapsed.toFixed(1)}s</b></>:`${tab==='reference'?'Apply Reference Style':tab==='manual'?'Apply Enhancement':'Enhance Image'}  →`}</button>
      </section>
      <section className="output"><div className="eyebrow">02 · ENHANCED IMAGE</div>{result ? <img src={result.enhanced_image_url} alt="Enhanced result"/> : <div className="output-empty"><span>✦</span><strong>Your result appears here</strong><small>Choose a control method and enhance</small></div>}{result&&<div className="fileline"><span>{result.method==='text'?'Text AI':result.method==='reference'?'Reference Transfer':'Manual Fine Tune'}</span><span>{result.total_processing_seconds.toFixed(2)}s</span></div>}</section>
    </section>

    {result && <section className="results"><div className="result-head"><div><span className="kicker">RESULT</span><h2>Before / After</h2></div><div className="row"><a className="text-button" href={result.enhanced_image_url} download="enhanced.png">Download</a><button className="text-button" disabled={loading||segmentationLoading} onClick={enhance}>Enhance Again</button><button className="text-button" onClick={reset}>Reset</button></div></div><Compare before={sourcePreview} after={result.enhanced_image_url}/>
      {result.method==='reference'&&<div className="reference-analysis"><div><span>Mode used</span><b>{result.mode_used}</b></div>{channels.map(([key,label])=><div key={key}><span>{label}</span><b>{result.final_global_adjustments[key]>=0?'+':''}{result.final_global_adjustments[key].toFixed(3)}</b></div>)}<div><span>Matched regions</span><b>{result.matched_regions.map(x=>x.name).join(', ')||'Global fallback'}</b></div></div>}
      <div className="side-by-side"><figure><img src={sourcePreview} alt="Original side-by-side"/><figcaption>Original</figcaption></figure><figure><img src={result.enhanced_image_url} alt="Enhanced side-by-side"/><figcaption>Enhanced</figcaption></figure></div>
    </section>}
    <footer><span>Local research demo · Images are processed by your backend</span><span>{health?.gpu_name || (health ? 'CPU inference' : 'Connect the local backend to begin')}</span></footer>
  </main>
}
