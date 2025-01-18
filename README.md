# An out-of-tree MLIR dialect

This is an example of an out-of-tree [MLIR](https://mlir.llvm.org/) dialect along with a finch `opt`-like tool to operate on that dialect.

## Building - Component Build

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-finch
```
To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.

## Building - Monolithic Build

This setup assumes that you build the project as part of a monolithic LLVM build via the `LLVM_EXTERNAL_PROJECTS` mechanism.
To build LLVM, MLIR, the example and launch the tests run
```sh
mkdir build && cd build
cmake -G Ninja `$LLVM_SRC_DIR/llvm` \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS=finch-dialect -DLLVM_EXTERNAL_FINCH_DIALECT_SOURCE_DIR=../
cmake --build . --target check-finch
```
Here, `$LLVM_SRC_DIR` needs to point to the root of the monorepo.


## How to Install (Updated version)

0. Git clone llvm repository
```
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
```

1. Install prerequisites for mlir python bindings

```
# Create python virtual environment
# Make sure your 'python' is what you expect.
which python
python -m venv ~/.venv/mlirdev
source ~/.venv/mlirdev/bin/activate

python -m pip install --upgrade pip
python -m pip install -r mlir/python/requirements.txt
```

2. Build MLIR

```
mkdir build
cd build
cmake -G Ninja ../llvm \                                                                                                                                                  
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DPython3_EXECUTABLE=<YOUR PYTHON PATH (e.g., /Users/jaeyeonwon/.venv/mlirdev/bin/python)>

# Build and Check mlir (I got 88% passed)
cmake --build . --target check-mlir 
```

3. Build Finch-mlir

```
https://github.com/finch-tensor/Finch-mlir.git
cd Finch-mlir
mkdir build && cd build
LLVM_BUILD_DIR=<YOUR LLVM BUILD PATH (e.g., /Users/jaeyeonwon/llvm-project/build>
cmake -G Ninja .. -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$LLVM_BUILD_DIR/bin/llvm-lit

# Build and Check finch-mlir
cmake --build . --target check-finch
```

