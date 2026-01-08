# Contributing to GenFrames

Thank you for your interest in contributing to GenFrames!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/Eta06/GenFrames.git
cd GenFrames
```

2. Install in development mode:
```bash
pip install -e ".[dev,all]"
```

3. Run tests:
```bash
pytest tests/
```

## Code Style

We use:
- `black` for code formatting
- `ruff` for linting

Format your code before committing:
```bash
black genframes/
ruff check genframes/
```

## Adding New Models

To add a new frame interpolation model:

1. Create a new file in `genframes/models/`
2. Inherit from `BaseModel`
3. Implement required methods:
   - `download_weights()`
   - `load_model()`
   - `interpolate()`

## Adding New Backends

To add a new computation backend:

1. Create a new file in `genframes/backends/`
2. Inherit from `BaseBackend`
3. Implement required methods:
   - `is_available()`
   - `get_device_name()`
   - `to_tensor()` / `to_numpy()`
   - `load_model()`

## Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Questions?

Open an issue on GitHub!
