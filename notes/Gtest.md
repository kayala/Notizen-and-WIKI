# Gtest

- [A quick introduction to the Google C++ Testing
  Framework](https://developer.ibm.com/articles/au-googletestingframework/)
- [Unit testing C code with
  gtest](https://notes.eatonphil.com/unit-testing-c-code-with-gtest.html)

For `TEST`, `TEST_F`, `TYPED_TEST`, etc, no test suite or test name can contain
an underscore.

## Libraries

- `gtest`: holds every test
- `gtest_main`: same as `gtest`, but has a default `main()` for testing

`gtest_main` only provides the `main()`. It should be treated as a library. This
is to make things easy by separating the `main()` in different files.

## manual compiling

- `git clone https://github.com/google/googletest.git`
- `unzip googletest-master.zip`
- `cd googletest-master`
- `make`
- `cp *.a /usr/lib`

## quick manual test

```sh
echo "
#include \"gtest/gtest.h\"

TEST(google_self_test, true_cases)
{
    ASSERT_EQ( true, true );
}" > test.cpp
g++ test.cpp -lgtest -lgtest_main -pthread
```

Note that `-pthread` may be unnecessary.

## writing tests in C++

```cpp
#include <gtest/gtest.h>


TEST(test_name, subtest_name) {
	ASSERT_TRUE(1 == 1);
	ASSERT_EQ(1, 1); // and many other
}

// main() declaration is only necessary when using gtest instead of gtest_main
int main(int argc, char *argv[]) {

	testing::InitGoogleTest(&argc, argv);

	return RUN_ALL_TESTS();
}
```

## writing tests in C


```cpp
#include <gtest/gtest.h>

extern "C" {
#include "../src/blockchain.h"
}

TEST(test_name, subtest_name) {
	ASSERT_TRUE(1 == 1);
	ASSERT_EQ(1, 1); // and many other
}

// main() declaration is only necessary when using gtest instead of gtest_main
int main(int argc, char *argv[]) {

	testing::InitGoogleTest(&argc, argv);

	return RUN_ALL_TESTS();
}
```

There are ASSERT and EXPECT:

- ASSERT: Fails fast, aborting the current function
- EXPECT: Continues after the failure

[Since gtest does not use exceptions, you cannot use `ASSERT_*` inside a non-void-returning
method.](https://chenchang.gitbooks.io/googletest_docs/content/googletest/AdvancedGuide.html). You
may either:

- use a reference inout parameter
- use EXPECT instead

## running with color

Use the `--gtest_color=yes` flag when executing the binary.

## customizing diagnostics message

```c++
EXPECT_TRUE(false) << "diagnostic message";
```

## test fixtures

Fixtures are used when the same test initialization is used on separate test
cases. Tests do not share the same instance.

```cpp
class MyStackTest : public ::testing::Test {
protected: // either public or protected, cannot be private
	virtual void SetUp() { // always called on test start
		st.push(34);
		st.push(28);
		st.push(56);
	}
	virtual void TearDown() { // always called on test end
	}
	MyStack st;
}

// test with fixture
TEST_F(MyStackTest, testPop) { // 1st arg must be test fixture class name
	int val = st.pop();
	EXPECT_EQ(56, val);
	EXPECT_EQ(54, val) << "this value must be 56";
	EXPECT_EQ(54, val) << "this value cannot be different from " << val;
}
```

You may use the constructor/destructor from `class MyStackTest`, but should use
`SetUp` and `TearDown`.

## typed test

You have e.g. a class that uses a template as a basis, and you want to test for
multiple types.

```cpp
template <typename T>
class FooTest : public testing::Test {
 public:
  ...
  using List = std::list<T>;
  static T shared_;
  T value_;
};

using MyTypes = ::testing::Types<char, int, unsigned int>;
/* using or typedef is necessary for TYPED_TEST_SUITE */
TYPED_TEST_SUITE(FooTest, MyTypes);
```

Then use `TYPED_TEST` as normal:

```cpp
TYPED_TEST(FooTest, DoesBlah) {
  // Inside a test, refer to the special name TypeParam to get the type
  // parameter. Since we are inside a derived class template, C++ requires
  // us to visit the members of FooTest via 'this'.
  TypeParam n = this->value_;

  // To visit static members of the fixture, add the 'TestFixture::' prefix.
  n += TestFixture::shared_;

  // To refer to typedefs in the fixture, add the 'typename TestFixture::'
  // prefix.  The 'typename' is required to satisfy the compiler.
  typename TestFixture::List values;

  values.push_back(n);
  ...
}

TYPED_TEST(FooTest, HasPropertyA) { ... }
```

## test with parameter

- [gtest param
  test](https://github.com/google/googletest/blob/main/googletest/include/gtest/gtest-param-test.h)
- [Parameterized Testing with
  Gtest](https://www.sandordargo.com/blog/2019/04/24/parameterized-testing-with-gtest)

Instead of writing multiple tests with different values of the parameter, you
can write one test using `TEST_P()` which uses `GetParam()` and can be
instantiated using `INSTANTIATE_TEST_SUITE_P()`. Example:

```c++
class FooTest : public ::testing::TestWithParam<const char*> {
  // implementation of all the fixture members
};

// TEST_P is for test cases using parameters, not the whole test
TEST_P(FooTest, DoesBlah) {
  EXPECT_TRUE(foo.Blah(GetParam()));
}

TEST_P(FooTest, HasBlahBlah) {
  ...
}

INSTANTIATE_TEST_SUITE_P(InstantiationName, // test suite name (!= test name)
                         FooTest, // class name
                         Values("meeny", "miny", "moe") // parameter generator
                         );
```

Notes:
- other parameter generators are: `ValuesIn`, `Bool` and `Combine`.
- you can have: one parameterized class, multiple parameterized test cases, and **multiple test
  suites**, all of they linked by the class name.
- you can use a parameterized class in a `TEST_F` test case.

This will run the following tests:

```
InstantiationName/FooTest.DoesBlah/0 for "meeny"
InstantiationName/FooTest.DoesBlah/1 for "miny"
InstantiationName/FooTest.DoesBlah/2 for "moe"
InstantiationName/FooTest.HasBlahBlah/0 for "meeny"
InstantiationName/FooTest.HasBlahBlah/1 for "miny"
InstantiationName/FooTest.HasBlahBlah/2 for "moe"
```

## mocking

- [Mocking cheatsheet](https://google.github.io/googletest/gmock_cheat_sheet.html)
- [Check invocation of class
  methods](https://stackoverflow.com/questions/42712372/check-invocation-of-class-methods-using-google-test)
