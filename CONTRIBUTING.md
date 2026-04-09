# Contributing

## Commit Convention

Format: `<type>(<scope>): <description>`

### Types

| Type | Use for |
|---|---|
| `feat` | New algorithms, models, metrics, environments |
| `fix` | Bug fixes |
| `exp` | Experiment changes — hyperparams, sweep configs, TOML tuning |
| `refactor` | Code restructuring without behavior change |
| `build` | Build system, dependencies, C extensions |
| `chore` | Cleanup, deletions, maintenance |
| `perf` | Performance / efficiency improvements |
| `docs` | Documentation |
| `test` | Tests |

### Scope (optional)

- **Submodules**: `data`, `utils`, `log`, `config`, etc.

### Rules

- Lowercase description, imperative mood, no trailing period
- One type per commit
- Keep description under 60 characters
- Use scope when the change is localized; omit for broad changes
- Breaking changes: add `!` after type/scope
- Unsure temporary commits: add `?` after type/scope

### Examples

```
build: pin jax to 0.4.x
chore: remove unused fm_v0 module
```
