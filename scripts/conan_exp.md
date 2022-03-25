from conans import ConanFile, CMake, tools 

import os 

import os.path 

import re 

class DicomDataManagementConan(ConanFile): 

    name        = "dicomdatamanagement" 

    version     = "1.0.2" 

    author      = "Carl Zeiss Meditec AG" 

    description = "This package contains all common base functionality of the dicom data management" 

    url         = "git@ssh.dev.azure.com:v3/ZEISSgroup-MED/MUC_Device-Framework/DicomDataManagement" 

    license     = "LicenseRef-CZM-proprietary-license" 

    settings    = "os", "compiler", "build_type", "arch", "arch_build", "ots" 

    generators = "cmake_find_package", "cmake_paths" 

    # the recipe revision of this package will be the git commit ID 

    revision_mode = "scm" 

    scm = { 

        "type": "git", 

        "url": url, 

        "revision": "auto" 

        } 

    build_test = True 

    def requirements(self): 

        # Stable packages should use stable dependencies 

        specific_channel = self.channel if self.channel == "stable" else "testing" 

        requires = ( 

            "applicationcore/1.1.0@zeiss/", 

            ) 

        for require in requires: 

            self.requires( 

                require + specific_channel 

            ) 

        # Linux is built from source. 

        self.requires("czmxmlcpp/3.0.0.117@zeiss/stable") 

        if self.settings.os == "Windows": 

            self.requires("nim/3.0.0@zeiss/stable") 

        else: 

            self.requires("nimlu/3.0.0@zeiss/stable") 

        if (self.settings.os == "Windows" or self.settings.ots != "Native"): 

            self.requires("qt/5.15.2@zeiss/stable") 

    def build_requirements(self): 

        # gtest is windows and linux-x64_86 only, but part of the yocto SDK 

        if (self.settings.os == "Windows" or self.settings.ots != "Native"): 

            self.build_requires("gtest/1.10.0@zeiss/stable") 

    def configure(self): 

        # if the conan pacakge is used we set shared = True 

        if (self.settings.os == "Windows" or self.settings.ots != "Native"): 

            self.options["gtest"].shared = True 

        if (self.settings.os == "Windows" or self.settings.ots != "Native"): 

            self.options["qt"].shared = True 

            self.options["qt"].qtwebengine = False 

            if (self.settings.os == "Linux" and self.settings.ots != "Native"): 

                self.options["qt"].opengl = "desktop" 

    def build(self): 

        cmake = CMake(self) 

        # generated Visual Studio solutions support only one build type at at 

        # time 

        cmake.definitions["CMAKE_CONFIGURATION_TYPES"] = self.settings.build_type 

        cmake.definitions["CMAKE_BUILD_TYPE"] = self.settings.build_type 

        cmake.definitions["TEST"] = "ON" if self.build_test else "OFF" 

        cmake.definitions["OTS"] = "ON" if (self.settings.ots == "Native") else "OFF" 

        cmake.definitions["ARCH_BUILD"] = self.settings.arch_build 

        cmake.definitions["ARCH"] = self.settings.arch 

        semver = re.match(r"(\d+)\.(\d+)\.(\d+)-?.*", self.version) 

        cmake.definitions["VERSION_MAJOR"] = semver[1] 

        cmake.definitions["VERSION_MINOR"] = semver[2] 

        cmake.definitions["VERSION_PATCH"] = semver[3] 

        cmake.definitions["BUILD"] = os.getenv("BUILD_ID", "none") 

        git = tools.Git() 

        cmake.definitions["SOURCEID"] = git.get_commit()[:7] 

        cmake.verbose = True 

        cmake.configure() 

        cmake.build() 

        # check with CTest 

        # cross-compilation: it's not possible to trigger tests on the build system 

        if self.should_test and self.build_test and self.settings.arch == self.settings.arch_build: 

            cmake.test(output_on_failure=True) 

    def package(self): 

        cmake = CMake(self) 

        # conan will automatically set the installation root to the package 

        # build directory 

        cmake.install() 

    def package_info(self): 

        # generators should create CMake packages with nice camel-case names 

        self.cpp_info.name = "DicomDataManagement" 

        ### 

        # COMPONENT "DicomServices" 

        # 

        ### 

        self.cpp_info.components["dicomservices"].name = "DicomServices" 

        self.cpp_info.components["dicomservices"].libs = ["DicomServices"] 

        self.cpp_info.components["dicomservices"].requires = [ 

            "applicationcore::applicationcore", 

            ] 

        self.cpp_info.components["dicomservices"].requires.append("czmxmlcpp::czmxmlcpp") 

        if self.settings.os == "Windows": 

            self.cpp_info.components["dicomservices"].requires.append("nim::nim") 

        else: 

            self.cpp_info.components["dicomservices"].requires.append("nimlu::nimlu") 

        if (self.settings.os == "Windows" or self.settings.ots != "Native"): 

            self.cpp_info.components["dicomservices"].requires.append("qt::qt") 

    def imports(self): 

        # copy dependencies to the respective output directories 

        # 

        # Note: For large dependencies (e.g. Qt) this can take a while. As an 

        # alternative consider using the "virtualrunenv" generator: 

        # https://docs.conan.io/en/latest/reference/generators/virtualrunenv.html 

        # Then the import() function can be removed. 

        if self.settings.build_type == "Debug": 

            output_dir = "debug" 

        else: 

            output_dir = "release" 

        self.copy("*.dll", dst=output_dir, src="bin") 

        self.copy("*.dylib*", dst=output_dir, src="lib") 

        self.copy("*.so*", dst=output_dir, src="lib") 

        self.copy("*.pdb", dst=output_dir, src="bin", root_package="applicationcore") 

        # ensure runtime-dependencies for DicomServicesDemo 

        if self.settings.build_type == "Debug": 

            if self.settings.os == "Windows": 

                demo_resources = [ 

                    ("*.dll", "plugins/platforms", "plugins/platforms", "Qt"), 

                    ("*",     "java",              "config/java",       "nim"), 

                ] 

                for r in demo_resources: 

                    self.copy(r[0], dst=output_dir+"/"+r[1], src=r[2], root_package=r[3]) 