# Real Time Ray Tracing Engine

## Get Started

To compile the project, use the following commands:

```bash
# Configure and build release binaries under `build/Release`
cmake -B build/Release -DCMAKE_BUILD_TYPE=Release
cmake --build build/Release

# Configure and build debug binaries under `build/Debug`
cmake -B build/Debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/Debug
```

### Development

This project uses custom githooks to format .cpp/.hpp files, to change the folder used to run githook scripts from use the following commands:

```bash
# Install dependency clang-format
brew install clang-format

chmod +x .githooks/pre-commit
git config --local core.hooksPath .githooks/
```