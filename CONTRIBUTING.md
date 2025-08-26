# Contributing to Nuke Nodes for ComfyUI

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nuke-nodes-comfyui.git
cd nuke-nodes-comfyui
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install development dependencies:
```bash
pip install pytest flake8
```

## Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and modular
- Use type hints where appropriate

## Adding New Nodes

When adding a new node:

1. Create the node class inheriting from `NukeNodeBase`
2. Implement required methods:
   - `INPUT_TYPES()` class method
   - Main processing function
3. Add comprehensive docstrings
4. Include proper error handling
5. Add the node to appropriate category file
6. Update `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`
7. Add tests for the new functionality
8. Update documentation

## Testing

Run tests before submitting:
```bash
python -m pytest tests/
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Commit with clear, descriptive messages
7. Push to your fork
8. Submit a pull request

## Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Include screenshots/examples if adding visual features
- Ensure CI tests pass
- Update documentation if needed

## Reporting Issues

When reporting bugs:
- Include ComfyUI version
- Provide steps to reproduce
- Include error messages and logs
- Describe expected vs actual behavior

## Code Review Process

All submissions require review. We look for:
- Code quality and style
- Test coverage
- Documentation completeness
- Performance considerations
- Compatibility with ComfyUI

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
