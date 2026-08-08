"""Root conftest, present so pytest adds the repo root to sys.path regardless
of invocation method, letting tests use absolute imports like `from
replication.optimized import solve` without installing the project as a package.
"""
