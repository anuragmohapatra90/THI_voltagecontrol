"""Actual Chromium verification of the requested single-screen desktop layout."""
from playwright.sync_api import sync_playwright


def check_browser(url):
    with sync_playwright() as p:
        browser=p.chromium.launch()
        page=browser.new_page(viewport={'width':1440,'height':900})
        page.goto(url,wait_until='networkidle')
        page.wait_for_function("""() => {
            const gd=document.querySelector('#comparison .js-plotly-plot');
            return gd && gd.data && gd.data[0]?.name==='Uncontrolled' && gd.data[2]?.name==='MLP + MPC';
        }""",timeout=60000)
        assert page.get_by_role('button',name='Load new scenario',exact=True).count()==1
        assert page.get_by_text('Username',exact=True).count()==0
        for width,height in [(1440,900),(1366,768),(1280,720),(1280,650)]:
            page.set_viewport_size({'width':width,'height':height})
            page.wait_for_function("""() => {
                const host=document.getElementById('comparison');
                const gd=host.querySelector('.js-plotly-plot');
                return Math.abs(gd.layout.height-host.getBoundingClientRect().height)<2;
            }""")
            result=page.evaluate("""() => {
                const controls=document.getElementById('controls').getBoundingClientRect();
                const graph=document.getElementById('comparison').getBoundingClientRect();
                const gd=document.querySelector('#comparison .js-plotly-plot');
                const items=[...document.querySelectorAll('#controls input,#controls button,#controls .prose')];
                return {scroll:document.documentElement.scrollHeight,body:document.body.scrollHeight,
                    viewport:innerHeight,right:graph.x>=controls.right,
                    controlsBottom:controls.bottom,graphBottom:graph.bottom,
                    inputsFit:items.every(el=>el.getBoundingClientRect().bottom<=innerHeight),
                    svgHeight:gd._fullLayout.height,hostHeight:graph.height};
            }""")
            assert result['scroll']<=height+1 and result['body']<=height+1,result
            assert result['right'] and result['inputsFit'],result
            assert result['controlsBottom']<=height+1 and result['graphBottom']<=height+1,result
            assert abs(result['svgHeight']-result['hostHeight'])<2,result
            print(f'Chromium {width}x{height}: setup left, complete comparison right, no page scroll: PASS')
        previous=page.locator('#controls').inner_text()
        page.get_by_role('button',name='Load new scenario',exact=True).click()
        page.wait_for_function("(text) => document.getElementById('controls').innerText!==text",arg=previous,timeout=30000)
        page.wait_for_function("() => document.querySelector('#comparison .js-plotly-plot').data[2].name==='MLP + MPC'")
        print('Chromium scenario button updates displayed calendar timestamp: PASS')
        browser.close()
