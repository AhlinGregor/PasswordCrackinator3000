This is a Java Brute Force Password Cracker. It is my project for the Programming 3 course.
It implements 3 solutions:
  - sequential solution
  - parallel solution
  - cuda accelerated solution

Currently the program runs only the sequential version but at the release stage you will run the program from command line and will choose the version with the arguments (1 being sequential, 2 parallel and 3 cuda).

## Requirements
- Java JDK 17+
- Nvidia GPU with CUDA support if you want to run the 3D accelerated version
- CUDA tooklit (11.x or newer)
- CMake 3.18+
- Visual Studio 2019+ (Windows) or GCC/Clang on Linux
- Intellij IDEA (not required but reccomended)

## Running the project as-is
To run the project as-is without modification to the code just execute the following commands from the project root:
```bash
javac -d build src/main/java/org/example/*.java
java -cp build org.example.Main <version>
```
Replace &lt;version&gt; with the desired number. 1 for single-threaded version, 2 for multi-threaded and 3 for CUDA version.
If for whatever reason project fails to run you can try rebuilding it following the **build instructions**.

## Build instructions
If you modify the code you should probably rebuild the project. Regardles of if you've generated a new JNI header or are using the one provided run the following commands in x64 Native Tools from the project root folder.
```bash
mkdir build
cmake -S . -B build
cmake --build build --config Release
```

## Generating a new JNI header
The JNI header is already provided however if you change anything in the CUDASolution.java it is smart to generate it again. To do so run the command:

```bash
javac -h . src/main/java/org/example/CUDASolution.java
```
from the root directory with x64 Native Tools Command Prompt for VS.
