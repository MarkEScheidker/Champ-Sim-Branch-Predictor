#!/bin/bash
make clean
make -j$(nproc) CXXFLAGS="-I/usr/local/include" LDLIBS="-ltensorflow"