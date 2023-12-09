call bin\clean

mkdir build

g++ src/main/cpp/main.cpp -o build/main -std=c++23 || goto :error
goto :success

:error
call bin\clean
exit /b 1

:success
