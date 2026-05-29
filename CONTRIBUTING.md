# Contributing

## Use of LLMs

Parts of this package have been written with the assistance of large language
models. LLM-generated code is not inherently trustworthy: it can be subtly
wrong, miss edge cases, or introduce regressions. **Human review of every
change is essential** — please read and understand all code before merging,
regardless of how it was produced.

## Code formatting

The project uses [JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl)
with the **Blue** style (configured in `.JuliaFormatter.toml`), pinned to
**version 2.5.1** for now. The examples below invoke it through `jlfmt`, a small
CLI wrapper that avoids paying Julia's startup cost on every run; install the
matching version with:

```bash
julia -e 'using Pkg; Pkg.add(PackageSpec(name="JuliaFormatter", version="=2.5.1"))'
```

Format all source files in-place:

```bash
jlfmt src test
```

Check whether the code is already correctly formatted (exits non-zero if not):

```bash
jlfmt --check src test
```

A **git pre-commit hook** may be installed at `.git-hooks/pre-commit`. It runs
the check automatically before every commit that touches Julia files, so CI never
rejects your change due to formatting. Activate it with:

```bash
git config core.hooksPath .git-hooks
```

If the hook fails, run the formatter, stage the result, and re-commit:

```bash
jlfmt src test
git add -u
git commit ...
```

To bypass the hook on a work-in-progress commit use `git commit --no-verify`.
