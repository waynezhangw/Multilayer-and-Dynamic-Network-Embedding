#include<iostream>
#include<time.h>     //clock_t
#include<cstdlib>    //system("pause")
#include<vector>
#include<string>

#include "getMultiType.h"

#define MY_PI 3.1415926535898

using namespace std;

int main()
{
	clock_t start,end;
	start=clock();    //start time

	getMultiType* try1=new getMultiType();
	try1->run();
	delete try1;
	try1=NULL;

	end=clock();     //end time
	cout<<"running time: "<<(double)(end-start)/CLOCKS_PER_SEC<<"s"<<endl;
	system("pause");
	return 0;
}