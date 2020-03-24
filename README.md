# Constrained Filter Feature Selection

Our code is implemented in Java and Python.

For the Python code contained in the Jupyter notebooks, you need to install some additional packages.
Currently you need to check each notebook for what is imported and install these packages manually.

For the Java code, you can import `java_solvers` as a Maven project into Eclipse.

`ChocoDemo` should work directly, its dependency is hosted on Maven Central.

For `Z3Demo`, matters are more complicated.
First, please download a [pre-built version of Z3](https://github.com/Z3Prover/z3/releases) and extract it.
(If that's too easy for you, you can also try to compile it.)
Our project also has a Maven dependency on `Z3`, but the Z3 download only provides a plain JAR.
Thus, extract [Maven](https://maven.apache.org/download.cgi) somewhere on your computer and add it to your `PATH` (optional).
Next, install the Z3 JAR into your local Maven repository (might need to adapt file path and version):

```
mvn install:install-file -Dfile=com.microsoft.z3.jar -DgroupId=com.microsoft -DartifactId=z3 -Dversion=4.8.7 -Dpackaging=jar
```

Furthermore, the JAR depends on DLLs also included in the Z3 download.
To enable access, add the `bin/` directory of your Z3 download to the environment variable `Path`.

For `ORToolsDemo`, the process is similar, download is [here](https://developers.google.com/optimization/install/download) , but you need to build two Maven artifacts.

For the C++ code `z3_demo`, you also need the pre-built version of `Z3`.
You can put the C++ file in a Visual Studio project.
Make sure to adapt (for all configurations, all platforms):

- `Project -> Properties -> Configuration Properties -> C/C++ -> General -> Additional Include Directories` by adding the path to the `include/` directory of the Z3 download.
- `Project -> Properties -> Configuration Properties -> Linker -> General -> Additional Library Directories` by adding the path to the `bin/` directory of the Z3 download.
- `Project -> Properties -> Configuration Properties -> Linker -> Input -> Additional Dependencies` by adding `libz3.lib`.

For the C++ code `gecode_demo`, you also need to [install](https://www.gecode.org/download.html) or compile the software and reference the DLLs in a similar manner.
See the [documentation](https://www.gecode.org/doc-latest/MPG.pdf) for more details.
