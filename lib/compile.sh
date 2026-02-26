g++ -shared -O3 -march=native -o map.so map.cpp -fopenmp -fPIC
g++ -shared -O3 -march=native -o profile.so profile.cpp -fPIC
