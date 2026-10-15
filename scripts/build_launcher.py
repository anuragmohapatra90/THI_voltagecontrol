"""Generate a thin Colab notebook from the reviewed Git checkout helper."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
nb=nbf.v4.new_notebook()
nb.metadata={'colab':{'name':'Grid_Control_Lab.ipynb','provenance':[]},
             'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
             'language_info':{'name':'python','version':'3.12'}}
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s))
def code(s):cells.append(nbf.v4.new_code_cell(s))
md('''# Voltage control for ADG · GitHub-backed launcher

This notebook loads the project directly from GitHub. Bookmark this GitHub-linked Colab page; future merged changes are fetched by running the checkout cell again. No source modules are embedded here.

**Run all → Open dashboard → load a scenario or adjust PQ.** The original ADG two-feeder grid and synthetic profiles are used. Use a CPU runtime. Keep Colab connected while using the separate dashboard tab.

After a PR is merged, restart the session and run all to load the latest code cleanly. An open dashboard does not update itself in the middle of an experiment. The exact fetched commit is printed for reproducibility.''')
md('''## 1. Fetch current project source
`main` is the stable version. To test a PR before merging, use `refs/pull/123/head`; branches, tags, and commit SHAs are also supported.

For a private repository, set `PRIVATE_REPO=True` and use the Colab secret `GITHUB_TOKEN` or the hidden prompt. No GitHub token is needed for public repositories.

The checkout refuses to overwrite local tracked changes. Close an old dashboard first so the new run starts with consistent source.''')
helper=(ROOT/'scripts/colab_checkout.py').read_text()
code('''REPOSITORY = "anuragmohapatra90/THI_voltagecontrol"
GIT_REF = "main"
PRIVATE_REPO = True

# Close the previous server before refreshing its source.
if 'grid_app' in globals():
    grid_app.close()

'''+helper+'''
import re, base64, getpass
if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', REPOSITORY):
    raise ValueError('REPOSITORY must be owner/name.')
remote_url = f'https://github.com/{REPOSITORY}.git'
git_auth_env = {}
if PRIVATE_REPO:
    token = None
    try:
        from google.colab import userdata
        token = userdata.get('GITHUB_TOKEN')
    except Exception:
        pass
    if not token:
        token = getpass.getpass('Read-only GitHub token for this repository: ')
    basic = base64.b64encode(('x-access-token:' + token).encode()).decode()
    git_auth_env = {
        'GIT_CONFIG_COUNT':'1',
        'GIT_CONFIG_KEY_0':'http.https://github.com/.extraheader',
        'GIT_CONFIG_VALUE_0':'AUTHORIZATION: basic ' + basic,
    }

PROJECT_DIR, SOURCE_COMMIT = checkout_source(
    remote_url, GIT_REF, Path('/content/grid-control-lab'), git_env=git_auth_env
)
git_auth_env.clear()
for secret_name in ['token','basic']:
    globals().pop(secret_name, None)
os.environ['GRID_LAB_SOURCE_REPOSITORY'] = REPOSITORY
os.environ['GRID_LAB_COMMIT'] = SOURCE_COMMIT
print('Repository:', REPOSITORY)
print('Requested ref:', GIT_REF)
print('Fetched commit:', SOURCE_COMMIT)
print('Source directory:', PROJECT_DIR)''')
md('''## 2. Install dependencies from this commit
If you ran the old project in this runtime, restart the session first. If installation changes an already-imported package version, restart and run all again.''')
code('''import subprocess, sys
subprocess.check_call([sys.executable,'-m','pip','install','-q','-r',str(PROJECT_DIR/'requirements.txt')])
for line in (PROJECT_DIR/'requirements.txt').read_text().splitlines():
    if '==' not in line:
        continue
    package,pinned=line.split('==')
    name={'scikit-learn':'sklearn'}.get(package,package)
    imported=sys.modules.get(name)
    if imported is not None and getattr(imported,'__version__',pinned)!=pinned:
        raise RuntimeError(f'{name} was loaded before installation. Restart the session and run all again.')
print('Dependencies ready for commit',SOURCE_COMMIT[:12])''')
md('''## 3. Load the fetched modules and verify the grid
Modules are loaded from the refreshed checkout, including when you reuse an already-open notebook.''')
code('''import importlib, os, sys
os.chdir(PROJECT_DIR)
for name in list(sys.modules):
    if name in ['grid_lab','plots','live_control','scenarios','dashboard','checks'] or name=='active_grid' or name.startswith('active_grid.'):
        sys.modules.pop(name,None)
sys.path.insert(0,str(PROJECT_DIR))
importlib.invalidate_caches()
from grid_lab import GridLab,Config
from checks import run_checks
from dashboard import build_dashboard,launch_dashboard
from live_control import prepare_controller
import pandas as pd
from IPython.display import display
lab=GridLab(Config())
print('Preparing grid and MLP controller…',flush=True)
prepare_controller(lab)
run_checks(lab)
print('Grid and MLP ready. Source commit:',SOURCE_COMMIT)''')
md('''## 4. Open the dashboard in another tab
The server continues in background threads after this cell finishes. Open the printed link directly; there is no dashboard login. Change the load/PV sliders for immediate reaction, or use Load new scenario to generate another synthetic daily load/PV snapshot from the original ADG setup. All three control results appear together.

If a new commit has been merged, restart the session and run all before launching a new comparison. Reopening the same GitHub-Colab link also loads the latest notebook content.''')
code('''from IPython.display import HTML,display
from html import escape
grid_app=build_dashboard(lab)
dashboard_url,_=launch_dashboard(grid_app,share=True)
print('DASHBOARD READY · source commit',SOURCE_COMMIT[:12])
print('Open in another tab:',dashboard_url)
display(HTML('<a href="'+escape(dashboard_url,quote=True)+'" target="_blank" rel="noopener noreferrer" '
             'style="display:inline-block;background:#087f8c;color:white;padding:14px 24px;border-radius:8px;'
             'font-size:18px;text-decoration:none">Open dashboard ↗</a>'))''')
md('''## Development workflow

**Change Python files → open PR → review / merge into main → restart Colab and run all.**

The Colab URL remains the same. Changes do not require downloading or uploading replacement notebooks. Saving a Drive copy creates an independent notebook; use the GitHub-linked version for the shared project.

Open the repository README for source documentation and `MODEL.md` for simulation assumptions. The backend retains the GitHub repository and fetched commit for reproducibility.''')
nb.cells=cells
nbf.validate(nb)
nbf.write(nb,ROOT/'Grid_Control_Lab.ipynb')
print('Launcher generated:',len(nb.cells),'cells')

