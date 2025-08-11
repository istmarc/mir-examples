import mir.ndslice;
import numir;
import mir.blas: gemv, gemm;
import std.stdio;

void main()
{

	writeln("Uninitialized arrays");
	{
		auto a = uninitSlice!float(3, 3);
		writeln(a);
	}

	{
		writeln("Mir is row major");
		auto a = [1.0f, 2.0f, 3.0f, 4.0f].sliced(2, 2);
		writeln(a);
		writeln("Its stored as in memory as [a[0,0] a[0,1] a[1, 0] a[1, 1]]");
		writeln("a[0,0] = ", a[0, 0]);
		writeln("a[0,1] = ", a[0, 1]);
		writeln("a[1,0] = ", a[1, 0]);
		writeln("a[1,1] = ", a[1, 1]);
	}

	writeln("Vector");
	{
		auto a = [1.0f, 2.0f].sliced(2);
		writeln(a);
	}

	writeln("Matrix");
	{
		auto a = [1.0f, 2.0f, 3.0f, 4.0f].sliced(2, 2);
		writeln(a);
	}

	writeln("Tensors");
	{
		auto a = [1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f].sliced(2, 2, 2);
		writeln(a);
		// Its shape can be acessed with .shape
		writeln(a.shape);
		// Its strides
		writeln(a.strides);
		// Print all the values
		for (auto i = 0; i < a.shape[0]; i++) {
			for (auto j = 0; j < a.shape[1]; j++) {
				for (auto k = 0; k < a.shape[2]; k++) {
					writeln(a[i, j, k]);
				}
			}
		}
	}


	writeln("Empty uninitialized array");
	{
		// Empty vector
		auto a = empty!float(2);
		writeln(a);
		// Empty matrix
		auto b = empty!float(2, 3);
		writeln(b);
		// Empty tensor
		auto c = empty!float(2, 3, 4);
		writeln(c);
	}

	writeln("Arrays of ones");
	{
		auto a = ones!float(2);
		writeln(a);
		auto b = ones!float(2, 3);
		writeln(b);
		auto c = ones!float(2, 3, 4);
		writeln(c);
	}

	writeln("Arrays of zeros");
	{
		auto a = zeros!float(2);
		writeln(a);
		auto b = zeros!float(2, 3);
		writeln(b);
		auto c = zeros!float(2, 3, 4);
		writeln(c);
	}

	writeln("Arrays of range iota");
	{
		auto a = iota(2).as!float();
		writeln(a);
		auto b = iota(2, 3).as!float();
		writeln(b);
		auto c = iota(2, 3, 4).as!float();
		writeln(c);
	}

	writeln("Vector elementwise operations: add, sub, mul and div");
	{
		auto a = iota(10).as!float();
		auto b = iota(10).as!float();
		auto c = a + b;
		writeln(c);
	}
	{
		auto a = iota(10).as!float();
		auto b = iota(10).as!float();
		auto c = a - b;
		writeln(c);
	}
	{
		auto a = iota(10).as!float();
		auto b = iota(10).as!float();
		auto c = a * b;
		writeln(c);
	}
	{
		auto a = iota([10], 1).as!float();
		auto b = iota([10], 1).as!float();
		auto c = a / b;
		writeln(c);
	}

	writeln("Scale by a scalar");
	{
		auto a = iota(10).as!float();
		auto b = 2.0f * a;
		writeln(b);
	}

	writeln("Matrix vector multiplication");
	{
		//auto a = iota([2,2], 1).as!float();
		//auto b = iota([2], 1).as!float();
		auto a = empty!float(2, 2);
		auto b = empty!float(2);
		auto c = empty!float(2);
		//gemv!float(1.0f, a, b, 0.0f, c);
		writeln(c);
	}

	writeln("Matrix matrix multiplication");
	{
		auto a = iota([2, 2], 1).as!float();
		auto b = iota([2, 2], 1).as!float();
		auto c = empty!float(2, 2);
		//gemm!float(1.0f, a, b, 0.0f, c);
		writeln(c);
	}

}
