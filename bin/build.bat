call bin\clean
mkdir build
g++ src/main/cpp/main.cpp -o build/main -std=c++23 -Ilib/eigen-3.4.0 -Ilib/MiniDNN/include -Wall -Werror || goto :error
goto :success

:error
call bin\clean
exit /b 1

:success
