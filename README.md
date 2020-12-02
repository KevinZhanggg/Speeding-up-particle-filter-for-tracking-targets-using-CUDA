# Speeding-Up-the-Particle-Filter-Algorithm-forTracking-Multiple-Targets-Using-CUDAProgramming

# Environment requirements:
Operating System:
```
Ubuntu 18.04
```
The following libs should be installed before compiling:

- Opencv 3.4.2 
- gsl
- gflag
- CUDA


# Compile
Before compling the code, you need to edit CMakeLists.txt 
Change CUDA version to the one you installed in your environment
```shell
mkdir build
cd build
cmake..
make 
```

# Run
You can run the prgram with default settings particle number is 300, std is 10, and video file is Ball.avi
```shell
./find-the-bomb
```
Or you can run this program with your own settings:
```shell
./find-the-bomb -file=videofile -ndpp=500 -std=10.0 -showall=true
```

