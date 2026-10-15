"""Real-browser layout, contrast, diagnostics and delayed-response feedback checks."""
from playwright.sync_api import sync_playwright
import os, base64


def screenshot_log(page):
    if os.environ.get('GRID_LAB_SCREENSHOT_LOG') == '1':
        data = page.screenshot(type='jpeg', quality=65, timeout=10000)
        print('BROWSER_SCREENSHOT:'+base64.b64encode(data).decode())


def check_contrast(page, mode):
    failures = page.evaluate('''() => {
        const rgb=s=>(s.match(/[\\d.]+/g)||[]).map(Number);
        const lum=c=>c.slice(0,3).map(v=>v/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4)
            .reduce((s,v,i)=>s+v*[.2126,.7152,.0722][i],0);
        const selected=document.querySelectorAll('#controls input:not([type="range"]),#controls label,#controls .info,#new-scenario');
        return [...selected].filter(e=>e.getBoundingClientRect().height>0).map(e=>{
            const style=getComputedStyle(e);const fg=rgb(style.color);
            let p=e,bg=[255,255,255,1];
            while(p) {const c=rgb(getComputedStyle(p).backgroundColor);if(c.length===3||c[3]>.99){bg=c;break;}p=p.parentElement;}
            const a=lum(fg),b=lum(bg);return {text:e.getAttribute('aria-label')||e.textContent||e.value,
                ratio:(Math.max(a,b)+.05)/(Math.min(a,b)+.05),fg:style.color,bg};
        }).filter(x=>x.ratio<4.5);
    }''')
    assert not failures, (mode, failures)
    print(f'Chromium {mode}: sidebar labels, gains, values and button contrast >=4.5:1: PASS')


def check_browser(url):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width':1440, 'height':900})
        page.on('pageerror', lambda error: print('Browser JS error:', error))
        page.goto(url, wait_until='domcontentloaded')
        page.wait_for_function('''() => {
            const gd=document.querySelector('#comparison .js-plotly-plot');
            return gd?.data?.[0]?.name==='Uncontrolled' && gd.data[2]?.name==='MLP + MPC'
                && document.querySelectorAll('.voltage-table tbody tr').length===15;
        }''', timeout=60000)
        assert page.get_by_role('button', name='Load new scenario', exact=True).count() == 1
        assert page.get_by_role('heading', name='Voltage control for ADG', exact=True).count() == 1
        assert 'OLTC tap position:' in page.locator('#controls').inner_text()
        assert page.get_by_text('Username', exact=True).count() == 0
        assert page.locator('.action-table tbody tr').count() == 3
        assert 'ΔP (kW)' in page.locator('.action-table').inner_text()
        assert 'ΔQ (kVar)' in page.locator('.action-table').inner_text()
        assert 'estimate − actual' in page.locator('.voltage-section').inner_text()
        assert 'Higher: use less ΔP' in page.locator('#gain-p').inner_text()
        assert 'Higher: use less ΔQ' in page.locator('#gain-q').inner_text()
        assert 'Higher: stronger voltage correction' in page.locator('#gain-v').inner_text()
        assert 'MLP(P' in page.locator('.cost-function').inner_text()
        check_contrast(page, 'light mode')
        screenshot_log(page)
        for width, height in [(1440,900), (1366,768), (1280,720), (1280,650)]:
            page.set_viewport_size({'width':width, 'height':height})
            page.wait_for_function('''() => {
                const host=document.getElementById('comparison');const gd=host.querySelector('.js-plotly-plot');
                const box=host.getBoundingClientRect();const expected=innerHeight-box.top-12;
                const diagnostics=document.querySelector('.diagnostic-layout');
                return Math.abs(gd.layout.height-expected)<2 && Math.abs(box.height-expected)<2
                    && diagnostics && Math.abs(diagnostics.getBoundingClientRect().height-(expected-18))<2;
            }''')
            result = page.evaluate('''() => {
                const controls=document.getElementById('controls').getBoundingClientRect();
                const graph=document.getElementById('comparison').getBoundingClientRect();
                const diagnostics=document.getElementById('diagnostics-panel').getBoundingClientRect();
                const gd=document.querySelector('#comparison .js-plotly-plot');
                const inputs=[...document.querySelectorAll('#controls input,#controls button,#controls .prose')];
                const table=document.querySelector('.voltage-table');const viewport=table.parentElement.getBoundingClientRect();
                const last=table.querySelector('tbody tr:last-child').getBoundingClientRect();
                return {scroll:document.documentElement.scrollHeight,body:document.body.scrollHeight,
                    controlsBottom:controls.bottom,graphBottom:graph.bottom,diagnosticsBottom:diagnostics.bottom,
                    right:graph.x>=controls.right && diagnostics.x>=graph.right,
                    inputsFit:inputs.every(e=>e.getBoundingClientRect().bottom<=innerHeight),
                    svgHeight:gd._fullLayout.height,hostHeight:graph.height,
                    lastBusVisible:last.bottom<=viewport.bottom+1,
                    costVisible:document.querySelector('.cost-function').getBoundingClientRect().bottom<=innerHeight,
                    horizontalFit:document.querySelector('.diagnostic-layout').scrollWidth<=diagnostics.width,
                    markersClear:[...gd.querySelectorAll('.scatterlayer .points path.point')].every(point=>{
                        const marker=point.getBoundingClientRect();
                        return [...gd.querySelectorAll('.colorbar .cbfill')].every(fill=>{
                            const bar=fill.getBoundingClientRect();
                            return marker.right<=bar.left || marker.left>=bar.right || marker.bottom<=bar.top || marker.top>=bar.bottom;
                        });
                    })};
            }''')
            assert result['scroll'] <= height+1 and result['body'] <= height+1, result
            assert result['right'] and result['inputsFit'], result
            assert max(result['controlsBottom'],result['graphBottom'],result['diagnosticsBottom']) <= height+1, result
            assert abs(result['svgHeight']-result['hostHeight']) < 2, result
            assert result['lastBusVisible'] and result['costVisible'] and result['horizontalFit'], result
            assert result['markersClear'], result
            print(f'Chromium {width}x{height}: full comparison, all 15 bus estimates and cost function, no page scroll: PASS')
        page.evaluate("document.documentElement.classList.add('dark');document.body.classList.add('dark')")
        check_contrast(page, 'dark mode')
        screenshot_log(page)
        # Hold the actual solver request to prove local feedback appears before any response.
        pattern = '**/queue/join**'
        held = []
        page.route(pattern, lambda route: held.append(route))
        previous = page.locator('.diagnostic-layout').get_attribute('data-scenario-id')
        with page.expect_request(pattern, timeout=10000):
            page.get_by_role('button', name='Load new scenario', exact=True).click()
        page.wait_for_function('''() => document.getElementById('new-scenario').disabled
            && document.getElementById('new-scenario').dataset.busy==='true'
            && document.getElementById('comparison').getAttribute('aria-busy')==='true'
            && document.getElementById('scenario-feedback').innerText.includes('Solving')''', timeout=1000)
        assert page.locator('.diagnostic-layout').get_attribute('data-scenario-id') == previous
        assert held, 'The backend request should still be held while loading feedback is visible.'
        screenshot_log(page)
        for route in held:
            route.continue_()
        page.unroute(pattern)
        page.wait_for_function('''old => {
            const d=document.querySelector('.diagnostic-layout');const gd=document.querySelector('#comparison .js-plotly-plot');
            return d.dataset.scenarioId!==old && String(gd.layout.meta.scenario_id)===d.dataset.scenarioId
                && !document.getElementById('new-scenario').disabled
                && document.getElementById('comparison').getAttribute('aria-busy')==='false';
        }''', arg=previous, timeout=30000)
        assert page.locator('.voltage-table tbody tr').count() == 15
        print('Chromium delayed backend: immediate loading label/spinner, disabled duplicate clicks, atomic diagnostics and restored button: PASS')
        browser.close()
