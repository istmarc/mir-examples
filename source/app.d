import mir.ndslice;
import std.stdio;

void main()
{

	writeln("Uninitialized arrays");
	auto a = uninitSlice!float(3, 3);

}
