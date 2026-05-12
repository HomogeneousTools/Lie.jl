# Contributing

## Use of LLMs

Parts of this package have been written with the assistance of large language
models. LLM-generated code is not inherently trustworthy: it can be subtly
wrong, miss edge cases, or introduce regressions. **Human review of every
change is essential** — please read and understand all code before merging,
regardless of how it was produced.

## Code formatting

The project uses [JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl)
with the **Blue** style (configured in `.JuliaFormatter.toml`).

Format all source files in-place:

```bash
julia --project=. -e 'using JuliaFormatter; format("src"); format("test")'
```

Check whether the code is already correctly formatted (exits non-zero if not):

```bash
julia --project=. -e '
  using JuliaFormatter
  ok = format("src", overwrite=false) && format("test", overwrite=false)
  exit(ok ? 0 : 1)
'
```

A **git pre-commit hook** may be installed at `.git-hooks/pre-commit`. It runs
the check automatically before every commit that touches Julia files, so CI never
rejects your change due to formatting. Activate it with:

```bash
git config core.hooksPath .git-hooks
```

If the hook fails, run the formatter, stage the result, and re-commit:

```bash
julia --project=. -e 'using JuliaFormatter; format("src"); format("test")'
git add -u
git commit ...
```

To bypass the hook on a work-in-progress commit use `git commit --no-verify`.
