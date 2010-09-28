#include <iostream>
#include "TAPN/TAPN.hpp"
#include "boost/smart_ptr.hpp"
#include "boost/make_shared.hpp"
#include "Core/TAPNXmlParser.hpp"

using namespace std;
using namespace VerifyTAPN::TAPN;
using namespace boost;



int main(int argc, const char* argv[]) {
	TimeInvariant ti = TimeInvariant::CreateFor("<= 4");
	cout << ti << endl;

	ti = TimeInvariant::CreateFor("<inf");
	cout << ti << endl;

	TimeInterval interval = TimeInterval::CreateFor("[0,3]");
	cout << interval << endl;

	interval = TimeInterval::CreateFor("[1, 4)");
	cout << interval << endl;

	interval = TimeInterval::CreateFor("(3,5]");
	cout << interval << endl;

	interval = TimeInterval::CreateFor("( 4, inf)");
	cout << interval << endl;

	VerifyTAPN::TAPNXmlParser parser;
	cout << (*parser.Parse("/home/mortenja/Desktop/test.xml"));

	return 0;
}


