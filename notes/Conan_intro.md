# Conan Wiki

## Conan Install

'pip install conan'

## Manage remote list

'conan remote list'
'conan search glog -- remote conan-center'
'conan remote add conan-transit https://conan-transit.bintray.com'

## Manage pkg

'conan install glog/0.4.0@bincrafters/stable -r conan-center'

'conan remove glog/o.4.0@bincrafters/stable'

## write conanfile.txt

[requires]
glog/0.4.0@bincrafters/stable

[generators]
cmake

'conan install'

* conaninfo.txt :check the detailed information of the package, including compiler information, system architecture, etc.

* conanbuildinfo.cmake : tell cmake dependencies, such as the reference path of the header file, the reference path of the library, the link of the library and other information;

* conanbuildinfo.txt : easy for check the information above