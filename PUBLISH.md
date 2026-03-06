# Publishing TensorDB v0.3.0

## 1. crates.io (Rust)

```bash
# Login once (get token from https://crates.io/settings/tokens)
cargo login

# Publish in dependency order:
cargo publish -p tensordb-native
sleep 30  # wait for crates.io index
cargo publish -p tensordb-core --no-default-features
sleep 30
cargo publish -p tensordb --no-default-features
```

## 2. PyPI (Python)

```bash
# Build wheel
cd crates/tensordb-python
maturin build --release

# Upload (get token from https://pypi.org/manage/account/token/)
twine upload target/wheels/tensordb-0.3.0-*.whl
```

## 3. npm (Node.js)

```bash
# Login once
npm login

# Build and publish
cd crates/tensordb-node
npm run build
npm publish --access public
```

## 4. Hacker News

Title: `Show HN: TensorDB – A Rust Bitemporal Ledger Database with Full SQL`
URL: `https://github.com/tensor-db/TensorDB`

## 5. Reddit r/rust

Title: `TensorDB v0.3.0 – Bitemporal ledger database with MVCC, full SQL, and 276ns reads`
URL: `https://github.com/tensor-db/TensorDB`

## 6. dev.to Blog Post

Saved at: `blog-post-devto.md` in the repo root.
Copy/paste into https://dev.to/new

## 7. Awesome Lists (PRs)

See below for submission commands.
