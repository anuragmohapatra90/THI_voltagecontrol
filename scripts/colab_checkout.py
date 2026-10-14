"""Refresh a managed checkout without overwriting local tracked edits."""
from pathlib import Path
import os
import subprocess


def checkout_source(remote_url, ref, destination, *, git_env=None):
    destination=Path(destination).expanduser().resolve()
    env=dict(os.environ, GIT_TERMINAL_PROMPT='0')
    if git_env: env.update(git_env)
    def git(*args,capture=True):
        result=subprocess.run(['git','-C',str(destination),*args],env=env,
                              text=True,capture_output=True)
        if result.returncode:
            # Do not echo Git stderr: authentication material must not reach notebook output.
            raise RuntimeError(f'Git {args[0]} failed. Check the repository URL, ref, and access. '
                               'For a private repository, enable PRIVATE_REPO and provide a read-only token in Colab.')
        return result.stdout.strip() if capture else None
    if not (destination/'.git').exists():
        if destination.exists() and any(destination.iterdir()):
            raise RuntimeError('Checkout destination is not empty and is not a managed Git checkout. Choose a different directory.')
        destination.mkdir(parents=True,exist_ok=True)
        git('init','--quiet')
        git('remote','add','origin',remote_url)
    else:
        if git('remote','get-url','origin') != remote_url:
            raise RuntimeError('Checkout belongs to another repository. Choose a different directory.')
        if git('status','--porcelain','--untracked-files=no'):
            raise RuntimeError('Local source edits exist. Commit or save them before refreshing; no files were overwritten.')
    git('fetch','--quiet','--depth','1','origin',ref)
    commit=git('rev-parse','FETCH_HEAD')
    git('checkout','--quiet','--detach',commit)
    return destination,commit
