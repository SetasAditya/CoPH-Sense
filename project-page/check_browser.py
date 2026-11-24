"""Optional Playwright QA using installed Chrome; does not train/render scenarios."""
from pathlib import Path
import argparse,json
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--base',default='http://127.0.0.1:18765/');args=ap.parse_args();base=args.base.rstrip('/')+'/'
result={'base':base,'scenarios':{},'console_errors':[]}
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
    page=browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
    page.on('pageerror',lambda e:result['console_errors'].append(str(e)))
    page.goto(base,wait_until='networkidle')
    provenance=page.evaluate('async()=>await (await fetch("assets/provenance.json")).json()')
    for name in ['dynamics','hazards','reciprocity','encounters','uncertainty']:
        print('Checking',name,flush=True)
        page.locator('#tab-'+name).click()
        page.wait_for_function('(name)=>{const v=document.querySelector("video");return v.readyState>=2&&v.currentSrc.endsWith(`/assets/${name}.mp4`)&&!document.querySelector("#event-buttons button").disabled}',arg=name)
        duration=page.locator('video').evaluate('(v)=>v.duration');assert duration>0
        # Every scenario's second jump target is predetermined, not selected by outcome.
        target={'dynamics':20,'hazards':25,'reciprocity':4.8,'encounters':4.1,'uncertainty':3}[name]
        key='query_index' if name=='uncertainty' else 'mission_seconds'
        expected=min(provenance['frames'][name],key=lambda f:abs(f[key]-target))['video_seconds']
        page.locator('#event-buttons button').nth(1).click()
        print('Seek',name,'expected',expected,'state',page.locator('video').evaluate('(v)=>({time:v.currentTime,seeking:v.seeking,ready:v.readyState,seekable:Array.from({length:v.seekable.length},(_,i)=>[v.seekable.start(i),v.seekable.end(i)]),buffered:Array.from({length:v.buffered.length},(_,i)=>[v.buffered.start(i),v.buffered.end(i)])})'),flush=True)
        page.wait_for_function('(expected)=>{const v=document.querySelector("video");return !v.seeking&&v.readyState>=2&&Math.abs(v.currentTime-expected)<.05}',arg=expected)
        pos=page.locator('video').evaluate('(v)=>v.currentTime');assert abs(pos-expected)<.05
        assert page.locator('video').evaluate('(v)=>v.error ? v.error.message : null') is None
        result['scenarios'][name]={'duration':duration,'expected_event_seek':expected,'actual_event_seek':pos,'decoded_frame_ready':True,'video_error':None}
        if name=='reciprocity':page.locator('.stage').screenshot(path='/tmp/coph-gallery-player.png')
    page.locator('#tab-dynamics').focus();page.keyboard.press('ArrowRight');assert page.locator('#tab-hazards').get_attribute('aria-selected')=='true';result['keyboard_navigation']=True
    page.screenshot(path='/tmp/coph-gallery-desktop.png',full_page=True)
    page.set_viewport_size({'width':390,'height':844});page.goto(base,wait_until='networkidle');page.wait_for_function('document.querySelector("video").readyState>=2');page.screenshot(path='/tmp/coph-gallery-mobile.png',full_page=True)
    result['mobile_document_width']=page.evaluate('document.documentElement.scrollWidth');assert result['mobile_document_width']<=390
    assert not result['console_errors'],result['console_errors'];browser.close()
(ROOT/'docs/presentation/browser_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
