"""Validate a thin launcher, safe real Git refresh, and the actual SimBench model."""
from pathlib import Path
import os, sys, subprocess, tempfile, py_compile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import nbformat
from colab_checkout import checkout_source


def check_checkout():
    with tempfile.TemporaryDirectory(dir=ROOT,prefix='checkout_test_') as td:
        root=Path(td); origin=root/'origin';origin.mkdir()
        def git(*args):
            return subprocess.check_output(['git','-C',str(origin),*args],text=True,stderr=subprocess.DEVNULL).strip()
        git('init','-b','main');git('config','user.name','Grid Lab Test');git('config','user.email','test@example.invalid')
        src=origin/'version.txt';src.write_text('v1')
        git('add','version.txt');git('commit','-m','Initial test source')
        target=root/'checkout'
        _,first=checkout_source(str(origin),'main',target)
        assert (target/'version.txt').read_text()=='v1'
        src.write_text('v2');git('add','version.txt');git('commit','-m','Merged update')
        _,second=checkout_source(str(origin),'main',target)
        assert second!=first and (target/'version.txt').read_text()=='v2'
        (target/'version.txt').write_text('local edit')
        try:checkout_source(str(origin),'main',target)
        except RuntimeError as exc:assert 'Local source edits' in str(exc)
        else:raise AssertionError('Refresh overwrote local edits')
        assert (target/'version.txt').read_text()=='local edit'
    print('Real Git refresh, new commit loading, and local-edit protection: PASS')


def main():
    for path in list(ROOT.glob('*.py'))+list((ROOT/'scripts').glob('*.py')):
        py_compile.compile(str(path),doraise=True)
    notebook=nbformat.read(ROOT/'Grid_Control_Lab.ipynb',as_version=4)
    nbformat.validate(notebook)
    assert all(not cell.get('outputs') for cell in notebook.cells if cell.cell_type=='code')
    for cell in notebook.cells:
        if cell.cell_type=='code':compile(cell.source,'launcher_cell','exec')
    helper=(ROOT/'scripts/colab_checkout.py').read_text()
    assert helper in notebook.cells[2].source,'Regenerate the launcher after changing checkout logic.'
    assert not any('%%writefile' in c.source for c in notebook.cells),'Source must come from GitHub.'
    check_checkout()
    os.environ['GRID_LAB_SOURCE_REPOSITORY']='local-validation'
    os.environ['GRID_LAB_COMMIT']='local-test-commit'
    from grid_lab import GridLab
    from checks import run_checks
    lab=GridLab()
    print(run_checks(lab))
    assert lab.metadata()['source_commit']=='local-test-commit'
    print('Thin notebook, source compilation, physical checks, and commit metadata: PASS')

if __name__=='__main__':main()
