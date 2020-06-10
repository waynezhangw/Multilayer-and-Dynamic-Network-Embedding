#include "getMultiType.h"

#include<iostream>
#include<iomanip>
#include<numeric>
#include<string>

using std::pair;
using std::string;
using std::cout;
using std::endl;
using std::istringstream;
using std::ofstream;
using std::setprecision;
using std::fixed;
using std::to_string;

getMultiType::getMultiType(void)
{
	tripDay = 2;
	singleHour=5;                              //有效的开始时间
	durationLast=15;                            //singleHour~（singleHour+durationLast）一般为计算的时间段
}


getMultiType::~getMultiType(void)
{
}

void getMultiType::makeSubwayLineMap() {
	pair<map<string,string>::iterator,bool> Insert_Pair;
	Insert_Pair=subwayLineNumber.insert(pair<string,string>("地铁一号线","-1"));
	insertFailOnMap(Insert_Pair.second);
	Insert_Pair=subwayLineNumber.insert(pair<string,string>("地铁二号线","-2"));
	insertFailOnMap(Insert_Pair.second);
	Insert_Pair=subwayLineNumber.insert(pair<string,string>("地铁三号线","-3"));
	insertFailOnMap(Insert_Pair.second);
	Insert_Pair=subwayLineNumber.insert(pair<string,string>("地铁四号线","-4"));
	insertFailOnMap(Insert_Pair.second);
	Insert_Pair=subwayLineNumber.insert(pair<string,string>("地铁五号线","-5"));
	insertFailOnMap(Insert_Pair.second);
	Insert_Pair=subwayLineNumber.insert(pair<string,string>("地铁七号线","-7"));
	insertFailOnMap(Insert_Pair.second);
	Insert_Pair=subwayLineNumber.insert(pair<string,string>("地铁九号线","-9"));
	insertFailOnMap(Insert_Pair.second);
	Insert_Pair=subwayLineNumber.insert(pair<string,string>("地铁十一号线","-11"));
	insertFailOnMap(Insert_Pair.second);
}
void getMultiType::insertFailOnMap(bool second){
	if(second==false){
		cout<<"subway map insert failed, impossible!!!"<<endl;
	}
}

void getMultiType::readODTrip(ifstream &fin){
	cout<<"reading trip..."<<endl;
	cout<<tripDay<<endl;

	double sOX,sOY,sDX,sDY;                                        //为了与vector的名字区别出来，前面全部加了s，这些都是在读信息的时候用的中间值
	int tempOtime,tempDtime,tempOhour,tempDhour;                   //分别存储转换后的上下车秒数和小时数
	int stops,transfers;                                           //存储这次trip跨了多少个站点和换乘次数
	string OName,DName;                                            //存储trip起点和终点的名字
	string OSubwayLine;                                            //为了读取换乘信息的中间变量，为了确定出起点所在的地铁的线路号
	string DSubwayLine;
	string tempTransField;
	double tempDuration;                                                            //暂时存放逗留时间
	

	string line;
	int times=1;                                                                    //看程序运行的速度
	double tStop=pow(10.0,7);                                                       //控制读多少行
	int countNight=0;                                                               //存储跨凌晨的trip次数
	int timeTooShortCount=0;                                                        //时间间隔太短或太长
	int number=0;                                                                   //累积行数
	int durationInvalid6677=0;                                                      //统计出逗留时间不参与计算的有多少个
	
	while(getline(fin,line))    //一行一行的读 
	{
		if(line=="end,")
		{
			switch(tripDay)
			{
			case 2:
				{
					//说明：在读取工作日时，是按照4月5号、6号、7号来排列的；读取非工作日是按照4月8号、9号、3号、4号来排列的

					ifstream finTrip2("..\\data\\20170406-Trip-statistics-2CN-duration_example.csv");//4月6号是星期四
					cout<<"now is the "<<tripDay<<"day"<<"   case 2 already"<<endl;
					tripDay++;
					readODTrip(finTrip2);	
					finTrip2.close();
					break;
				}
			case 3:
				{
					ifstream finTrip3("D:\\data\\深圳数据\\TRIP20180707\\CN\\CN2\\20170407-Trip-statistics-2CN-duration.csv");//4月7号是星期五
					cout<<"now is the "<<tripDay<<"day"<<"   case 3 already"<<endl;
					tripDay++;
					//readODTrip(finTrip3);
					finTrip3.close();
					break;
				}
			case 4:
				{
					ifstream finTrip4("D:\\data\\深圳数据\\TRIP20180707\\CN\\CN2\\20170404-Trip-statistics-2CN-duration.csv"); //4月8号是星期六
					cout<<"now is the "<<tripDay<<"day"<<"   case 4 already"<<endl;
					tripDay++;
					//readODTrip(finTrip4);
					finTrip4.close();
					break;
				}
			case 5:
				{
					ifstream finTrip5("D:\\data\\深圳数据\\TRIP20180707\\CN\\CN2\\20170409-Trip-statistics-2CN-duration.csv"); //4月9号是星期天
					cout<<"now is the "<<tripDay<<"day"<<"   case 5 already"<<endl;
					tripDay++;
					//readODTrip(finTrip5);
					finTrip5.close();
					break;
				}
			case 6:
				{
					ifstream finTrip6("D:\\data\\深圳数据\\TRIP20180707\\CN\\CN2\\20170404-Trip-statistics-2CN-duration.csv");//4月4号是星期二
					cout<<"now is the "<<tripDay<<"day"<<"   case 6 already"<<endl;
					tripDay++;
					//readODTrip(finTrip6);
					finTrip6.close();
					break;
				}
			case 7:
				{
					ifstream finTrip7("D:\\data\\深圳数据\\TRIP20180707\\CN\\CN2\\20170403-Trip-statistics-2CN-duration.csv"); //4月3号是星期一
					cout<<"now is the "<<tripDay<<"day"<<"   case 7 already"<<endl;                 //一次性把七天的数据都读进来，内存可能放不下
					tripDay++;
					//readODTrip(finTrip7);
					finTrip7.close();
					break;
				}
			default:
				{
					cout<<"reading trip switch fails！！！！！！！！"<<endl;
					break;
				}
			}
			continue;
		}

		istringstream sin(line);   
		string field;
		int i=1;
		while (getline(sin,field,','))   //用vector读取时，这个CSV文件的字段超过5个时可能会出错
		{
			if(i==2)          //只读第row列的数据，然后push_back到vector里
			{
				tempOtime=convertTimeStringToInt(field);     //放Otime进来
				tempOhour=extractTimeHour(field);            
				i++;
				continue;
			}
			else if(i==3)
			{
				OName=field;                           //放起点的站名进来
				i++;
				continue;
			}
			else if(i==4)
			{
				sOX=convertStringToDouble(field);     //放OX进来
				i++;
				continue;
			}
			else if(i==5)
			{
				sOY=convertStringToDouble(field);     //放OY进来
				i++;
				continue;
			}
			else if(i==6)
			{
				tempDtime=convertTimeStringToInt(field);     //放Dtime进来
				tempDhour=extractTimeHour(field);
				i++;
				continue;
			}
			else if(i==7)
			{
				DName=field;                           //放终点的站名进来
				i++;
				continue;
			}
			else if(i==8)
			{
				sDX=convertStringToDouble(field);     //放DX进来
				i++;
				continue;
			}
			else if(i==9)
			{
				sDY=convertStringToDouble(field);     //放DY进来
				i++;
				//break;
				continue;
			}
			else if(i==11)                             //放换乘信息那一列进来
			{
				OSubwayLine=field;    
				OSubwayLine=extractSubwayLine(OSubwayLine); 
				if (subwayLineNumber.count(OSubwayLine)!=0){  //得到换乘信息里第二个位置是不是地铁字样
					OSubwayLine=subwayLineNumber[OSubwayLine];
				}
				else{
					OSubwayLine = "";          //如果不是地铁字样，那就不用加如：-2
				}
				DSubwayLine=field;    
				DSubwayLine=extractLastSubwayLine(DSubwayLine); 
				if (subwayLineNumber.count(DSubwayLine)!=0){  //得到换乘信息里第二个位置是不是地铁字样
					DSubwayLine=subwayLineNumber[DSubwayLine];
				}
				else{
					DSubwayLine = "";          //如果不是地铁字样，那就不用加如：-2
				}
				tempTransField = field;
				i++;
				continue;
			}
			else if(i==13)
			{
				stops=int(convertStringToDouble(field));     
				i++;
				continue;
			}
			else if(i==14)
			{
				transfers=int(convertStringToDouble(field));
				if(transfers>5)     //消除异常换乘次数，如果换乘了超过5次，就默认他换了5次，不能再大了
				{
					transfers=5;
				}
				i++;
				continue;
			}
			else if(i==15)
			{
				tempDuration=convertStringToDouble(field);
				i++;
				break;
			}

			else
			{
				i++;
				continue;
			}	
		}
		number++;                                                             //累积行数
		if(number==times*100000)
		{
			cout<<number<<endl;     
			times++;
		}

		if(tempDhour<tempOhour)                                               //要是有下车时间早于上车时间的trip，就跳过吧，这个还是有一定数量的
		{
			countNight++; 
			cout<<"the "<<number<<" row has boarding time conflicts"<<endl;
			continue;                                                   
		}
		
		//这里是控制读取进来的数据的时间段
		//if(tempOhour>=0 && tempOhour<=24)                                      //所有时段都要
		//if(tempOhour>=singleHour && tempOhour<singleHour+1)                    //这里分时段push_back到vector中,不想分时段就改一下阈值就好
		//if(tempDhour>=singleHour && tempDhour<24 || tempDhour>=0 && tempDhour<2)   //特殊的跨凌晨
		if(tempDhour>=singleHour && tempDhour<singleHour+durationLast)            //这里分时段push_back到vector中,不想分时段就改一下阈值就好
		{
			if(tempDtime-tempOtime<200||tempDtime-tempOtime>12600)       //如果是小于0的时间间隔或者是极短（200s）的时间间隔就不要输进来了
			{                                                            //如果起终点时间超过3.5h那也不要了
				timeTooShortCount++;
				continue;
			}
			if(tempDuration==0.66 || tempDuration==0.77)     //它是多条相同卡号的最后一条且没回家或者它是单独的一条trip
			{
				durationInvalid6677++;
				continue;
			}

			OStopName.push_back(OName+OSubwayLine);                      //把站名也放进来,并且后面加个线路号
			DStopName.push_back(DName+DSubwayLine);
			OGrid_X.push_back(sOX);                                         //把起始站点的坐标放进来
			OGrid_Y.push_back(sOY); 
			DGrid_X.push_back(sDX);      
			DGrid_Y.push_back(sDY);   
			transferField.push_back(tempTransField);                     //把巨大的换乘字段也放进来
			timeDiff.push_back(tempDtime-tempOtime+tempOhour/100.0);     
			ODStopNumber.push_back(stops+transfers/10.0);
			tripDurationTime.push_back(tempDuration);                    //把逗留时间放进来
		}

		if(number>2.0*tStop)     //如果number过大就                                                           
		{
			cout<<"break in advance"<<endl;
			break;
		}
	}

	cout<<"the amount of trip that push_back to vector："<<timeDiff.size()<<"--"<<OStopName.size()<<"    rows："<<number<<endl;
	cout<<"get off time wrong, unable push to trip："<<countNight<<endl;
	cout<<"the boarding duration is either too long or too short, unable push to trip："<<timeTooShortCount<<endl;
	cout<<"duration time is not good："<<durationInvalid6677<<endl;
	cout<<"trip reading finished"<<endl<<endl;

	cout<<"trip start station name based accumulation reading over"<<endl<<endl;
}
string getMultiType::extractSubwayLine(string str1)  //一聪跑的那个trip的第11个字段的换乘信息中是英文，无法判断几号线这种，不适用于这里
{
	string sub;                        //存储最后输出的地铁名字字段
	//参考读时间的方法
	istringstream sin(str1);   
	string field;
	int i=0;
	while (getline(sin,field,'-')) 
	{
		if(i==1)     //因为换乘线路是在换乘字段的第二个
		{
			sub=field;
			break;
		}
		i++;
	}
	return sub;
}
string getMultiType::extractLastSubwayLine(string str1) {
	string sub;                        //存储最后输出的地铁名字字段
	istringstream sin(str1);   
	string field;
	vector<string> vecTemp;            //暂时把每个都放进来，然后取倒数第三个出来
	while (getline(sin,field,'-')) 
	{
		vecTemp.push_back(field);
	}
	sub = vecTemp[vecTemp.size()-3];
	return sub;
}
double getMultiType::convertStringToDouble(string str1)
{
	//尝试string转double
	double n1;
	const char* ch= str1.c_str();
	n1=atof(ch);                   //如果是英文字符串，转过来就是0，而不是一个double的数

	return n1;
}

int getMultiType::convertTimeStringToInt(string str1)
{
	double second=0;
	istringstream sin(str1);   
	string field;
	int i=0;
	while (getline(sin,field,' ')) 
	{
		if(i==1)     //因为时间在第二个位置，第一个位置是年月日
		{
			istringstream sin(field);
			string field2;
			int j=0;
			while (getline(sin,field2,':'))
			{
				if(j==0)
				{
					double seeHour=convertStringToDouble(field2);                     //初步判断读进来的小时数
					if(seeHour<5 && seeHour>=0)
					{
						second=second+(convertStringToDouble(field2)+24)*3600;
					}
					else
					{
						second=second+convertStringToDouble(field2)*3600;   //先读到小时，换算成秒进行累加
					}

					j++;
					continue;
				}
				if(j==1)
				{
					second=second+convertStringToDouble(field2)*60;    //二读分钟，换算成秒进行累加
					j++;
					continue;
				}
				if(j==2)
				{
					second=second+convertStringToDouble(field2);      //直接读到秒
					j++; 
					break;
				}
			}
		}
		i++;
	}
	return int(second);
}
int getMultiType::extractTimeHour(string str1)
{
	int hour;
	istringstream sin(str1);   
	string field;
	int i=0;
	while (getline(sin,field,' ')) 
	{
		if(i==1)
		{
			istringstream sin(field);
			string field2;
			while (getline(sin,field2,':'))
			{		
				hour=int(convertStringToDouble(field2));   //先读到小时，
				break;
			}
		}
		i++;
	}

	return hour;
}
void getMultiType::readBusStop()
{
	string name;        //存站点名字的中间变量
	int ID=1;           //存站点的ID号，这里待读取的数据中没有，一条一条累加即可,从1开始
	double X,Y;         //存坐标的中间变量
	ifstream finBus("..\\data\\merge_BusStation改进dbscan加地铁到前面.csv");   //存有合并后的站点数据
	string line;                                   //当然这里的合并后的站点也存在一定问题：站点在研究区外等
	ifstream &fin=finBus;
	while(getline(fin,line))    //一行一行的读 
	{
		istringstream sin(line);   
		string field;
		int i=1;
		while (getline(sin,field,','))   //用vector读取时，这个CSV文件的字段超过5个时可能会出错
		{
			if(i==1)          //控制读哪一列
			{
				name=field;    
				i++;
				continue;
			}
			else if(i==3)
			{
				X=convertStringToDouble(field);
				i++;
				continue;
			}
			else if(i==4)
			{
				Y=convertStringToDouble(field);
				i++;
				break;
			}
			else
			{
				i++;
				continue;
			}	
		}
		oneWStopID.push_back(ID);
		oneWStopName.push_back(name);
		oneWX.push_back(X);
		oneWY.push_back(Y);
		ID++;
	}
	finBus.close();
	cout<<"read oneW station over"<<endl;
	cout<<ID<<" is it 10626? "<<oneWStopID[oneWStopID.size()-1]<<endl;
}

void getMultiType::makeStopStartCount() {
	pair<map<string,int>::iterator,bool> Insert_Pair;
	for(unsigned int i=0;i<oneWStopName.size();i++)          //for的是一万个站点
	{
		//构建利用站点名索引其所在的位置号，为了在图里读着顺就从1开始吧
		Insert_Pair=oneWStopNameToID.insert(pair<string,int>(oneWStopName[i],i+1));
		if(Insert_Pair.second==false)
		{
			cout<<"oneWStopNameToID insert falied, how come?"<<endl;
		}
	}

	int tripHeadID;                                         //遍历到每一次trip的起点ID
	int tripTailID;
	int tripEdgeType;                                       //trip边的类型
	int noFind=0;                                           //累积那些trip里有但一万个站点里没有的情况
	int yesFind=0;                                          //累积那些trip里有一万个站点里也有的情况
	int times=1;                                            //看执行的速度
	pair<map<EdgeHeadTail,int>::iterator,bool> Insert_Pair2;//为了唯一性存边类型-头节点编号-尾节点编号
	int num = 0;                                            //记录插到map EHTIndex里的条数

	vector<int> durOccurCount;                       //有效的边的有效逗留时间出现的次数
	vector<double> durAccumulation;                  //多次有效的逗留时间的累计

	//以到达某地来算该地的平均逗留时间和平均行程时间，大小是1万个站点的长度
	vector<double> overallAverageTime;
	vector<double> overallAverageDur;
	vector<int> overallTimeCount;                    //以终点为导向的到达次数
	vector<int> overallDurCount;                     //以终点为导向的有效逗留时间约束的到达次数，比上面的值小
	for(unsigned int i=0;i<oneWStopName.size();i++) {     //for的是一万个站点
		overallAverageTime.push_back(0);
		overallAverageDur.push_back(0);
		overallTimeCount.push_back(0);
		overallDurCount.push_back(0);
	}

	cout<<endl<<endl<<"the later ID of trip is based on this number, index start from 0, the largest one is:"<<OStopName.size()<<endl<<endl;
	cout<<"statistic of OD edge type..."<<endl;
	for(unsigned int i=0;i<OStopName.size();i++)           //for的所有的trip，这个就很大了
	{
		tripHeadID = convertStopNameToID(OStopName[i],OGrid_X[i],OGrid_Y[i]);
		tripTailID = convertStopNameToID(DStopName[i],DGrid_X[i],DGrid_X[i]);
		tripEdgeType = convertPairStopToEdgeType(tripHeadID, tripTailID);
		if (tripEdgeType != 0) {
			//累计以终点为导向的行程时间和逗留时间
			overallTimeCount[tripTailID-1] ++;
			overallAverageTime[tripTailID-1] += (timeDiff[i]/60.0);
			if(tripDurationTime[i]>1) {
				overallDurCount[tripTailID-1] ++;
				if(tripDurationTime[i]==180.6) {
					overallAverageDur[tripTailID-1] += 15.0;
				}
				else {
					overallAverageDur[tripTailID-1] += (tripDurationTime[i]/60.0);
				}
			}
			//把边关系-头节点-尾节点-时间-次数，存起来
			EdgeHeadTail temp(tripEdgeType, tripHeadID, tripTailID);
			Insert_Pair2 = EHTIndex.insert(pair<EdgeHeadTail,int>(temp,headID.size()));  //为了让value值从0开始索引
			if(Insert_Pair2.second==false)              //插不进来就代表这种边关系下已经存在过该头节点-尾节点的关系了
			{
				int numNow = EHTIndex[temp];
				//第二次访问到这个边时的行程时间计算
				occurCount[numNow]++;
				timeAccumulation[numNow] += timeDiff[i];
				averageTime[numNow] = timeAccumulation[numNow]/occurCount[numNow];
				//逗留时间的计算
				if(tripDurationTime[i]>1) {
					durOccurCount[numNow]++;
					if(tripDurationTime[i]==180.6) {
						durAccumulation[numNow] += 15;
						averageDuration[numNow] = durAccumulation[numNow]/durOccurCount[numNow];
					}
					else {
						durAccumulation[numNow] += tripDurationTime[i]/60.0;
						averageDuration[numNow] = durAccumulation[numNow]/durOccurCount[numNow];
					}
				}
			}
			else {                                                 //能插进来就代表是新的
				//边类型-头-尾
				edgeType.push_back(tripEdgeType);
				headID.push_back(tripHeadID);
			    tailID.push_back(tripTailID);
				//该边下的行程时间统计
				occurCount.push_back(1);
				timeAccumulation.push_back(timeDiff[i]);
				averageTime.push_back(timeAccumulation[num]/occurCount[num]);
				//该边下的逗留时间统计
				if(tripDurationTime[i]>1) {                 //如果逗留时间大于1，代表是有逗留的或有意义的
					durOccurCount.push_back(1);
					if(tripDurationTime[i]==180.6) {        //180.6是之前跑逗留时间的时候算的那种逗留时间为负值的情况				
						durAccumulation.push_back(15.0);    //默认假设的最短逗留时间为一刻钟
						averageDuration.push_back(durAccumulation[num]/durOccurCount[num]);
					}
					else {
						durAccumulation.push_back(tripDurationTime[i]/60.0);    //正常的逗留时间,以分钟来计
						averageDuration.push_back(durAccumulation[num]/durOccurCount[num]);
					}
				}
				else {
					durOccurCount.push_back(0);
					durAccumulation.push_back(0);    //没有逗留时间
					averageDuration.push_back(0);
				}
				num++;
			}
			yesFind++;
		}
		else {
			noFind++;
		}
		if(i==times*100000)
		{
			times++;
			cout<<i<<endl;
		}
	}
	cout<<"final trip num (million) should be the sum of noFind and yesFind: "<<yesFind+noFind<<endl;
	cout<<"in the pushed trip that oneW also has, yesFind is equal to: "<<yesFind<<endl;
	cout<<"in the pushed trip that oneW does not have，noFind is equal to： "<<noFind<<endl;
	cout<<"trip num is： "<<OStopName.size()<<endl;
	cout<<"check my map -- oneWStopNameToID(10626) is my map broken: "<<oneWStopNameToID.size()<<endl;
	cout<<"check my map -- EHTIndex(same as num) is my map broken："<<EHTIndex.size()<<endl;
	cout<<"  num is at this large："<<num<<"the size of edgeType is："<<edgeType.size()<<endl;

	seeEdgeTypeNum(edgeType);                  //看一下各类别有多少

	//算一下每个以终点为表示的平均逗留时间和行程时间
	for(unsigned int i=0;i<overallAverageTime.size();i++) {
		if (overallTimeCount[i] == 0 || overallDurCount[i] == 0) {
			continue;
		}
		overallAverageTime[i] /= overallTimeCount[i];
		overallAverageDur[i] /= overallDurCount[i];
	}

	ofstream outfile1;
	outfile1.open("..\\result\\20170405_edgeType"+to_string(long long(singleHour))+"-"
		+to_string(long long(singleHour+durationLast))+".csv");
	for(map<EdgeHeadTail,int>::iterator it=EHTIndex.begin(); it!=EHTIndex.end(); it++) //这样遍历确保了输出按照排序输出
	{
		if(edgeType[it->second]==1) {
			outfile1<<1<<",";
		}
		else if(edgeType[it->second]==2 || edgeType[it->second]==3) {
			outfile1<<2<<",";
		}
		else if(edgeType[it->second]==4){
			outfile1<<3<<",";
		}
		else {
			cout<<"impossible!!!!!!!!!!!"<<endl;
		}
		int temp_tail = it->first.tail;
		double ratio_of_travelTime = 1;
		if (overallAverageTime[temp_tail-1]!=0) {
			ratio_of_travelTime = (averageTime[it->second]/60.0)/overallAverageTime[temp_tail-1];
		}
		double ratio_of_travelDur = 1;
		if (overallAverageDur[temp_tail-1]!=0) {
			ratio_of_travelDur = (averageDuration[it->second]/60.0)/overallAverageDur[temp_tail-1];
		}
		double attractiveness = occurCount[it->second] * ratio_of_travelTime * ratio_of_travelDur;
		outfile1<<fixed<<setprecision(6)<<headID[it->second]<<","<<tailID[it->second]<<",";
		outfile1<<fixed<<setprecision(6)<<occurCount[it->second]<<","<<averageTime[it->second]/60.0<<",";
		outfile1<<fixed<<setprecision(6)<<durOccurCount[it->second]<<","<<averageDuration[it->second]<<","<<attractiveness<<endl;
		//为了让出现次数为1，也就是边得权重为1，MNE可能不适合有权重的边
		/*outfile1<<fixed<<setprecision(6)<<1<<","<<occurCount[it->second]<<","
		<<timeAccumulation[it->second]<<","<<averageTime[it->second]<<endl;*/
	}
	outfile1.close();
	cout<<"edgeType is finished!"<<endl;
}
void getMultiType::seeEdgeTypeNum(vector<int> &vec) {
	int A1=0;
	int A2=0;
	int A3=0;
	int A4=0;
	for (unsigned int i=0;i<vec.size();i++) {
		if (vec[i] == 1) A1++;
		if (vec[i] == 2) A2++;
		if (vec[i] == 3) A3++;
		if (vec[i] == 4) A4++;
	}
	cout<<"A1:" <<A1<<"  A2:"<<A2<<"  A3:"<<A3<<"  A4:"<<A4<<endl;
}
bool getMultiType::isSubwayStop(string str1)
{
	//先判断长度够不够，要不然后面会出错
	int lastP=str1.size()-1;                             //定义这个字符串的最后一个位置
	if(str1.size()<3)                                    //如果这个字符串长度过小，下面的就不用考虑了
	{
		return false;
	}
	//先判断最后一个是不是数字
	if(str1[lastP]=='1'|| str1[lastP]=='2' || str1[lastP]=='3' || str1[lastP]=='4' || str1[lastP]=='5' || str1[lastP]=='7'|| str1[lastP]=='9')
	{
		if(str1[lastP-1]=='-' || str1[lastP-1]=='1')     //在最后一个是数字的情况下，倒数第二个还是 - 或者 1 ，那么肯定就是地铁站点了
		{
			return true;
		}
		else
		{
			cout<<"the last one in the string is number, but the one before the last one is not："<<str1<<endl;
			return false;
		}
	}
	else
	{
		return false;
	}
}
string getMultiType::extractSubwayName(string str1)
{
	string finalString;
	int lastP=str1.size()-1;                             //定义这个字符串的最后一个位置
	string tempStr;
	string zhan="站";
	if(str1[lastP-1]=='-')
	{
		tempStr=str1.substr(str1.size()-4,2);                    //提取那个字看是不是“站”字
		if(tempStr==zhan)                                        //如果是站字，那就去掉它
		{
			finalString=str1.substr(0,str1.size()-4);            //保留“站”字之前的内容
			return finalString+str1[lastP];                      //加上那个数字，最后可能从   益田站-3  变成  益田3
		}
		else                        //如果不是站字，那就直接加上线路号后输出
		{
			finalString=str1.substr(0,str1.size()-2);
			return finalString+str1[lastP]; 
		}
	}
	else                           //这里默认不是‘-’就是‘1’，因为前面已经千真万确判断出来这是个地铁站点名字了
	{
		if(str1[lastP-1]!='1')
		{
			cout<<"the one before the last one is not '1'? why："<<str1<<endl;
		}
		tempStr=str1.substr(str1.size()-5,2);                    //提取那个字看是不是“站”字
		if(tempStr==zhan)                                        //如果是站字，那就去掉它
		{
			finalString=str1.substr(0,str1.size()-5);            //保留“站”字之前的内容
			return finalString+str1[lastP-1]+str1[lastP];        //加上那个数字，最后可能从   益田站-11  变成  益田11
		}
		else                        //如果不是站字，那就直接加上线路号后输出
		{
			finalString=str1.substr(0,str1.size()-3);
			return finalString+str1[lastP-1]+str1[lastP]; 
		}
	}
}
int getMultiType::convertStopNameToID(string str1, double tripX, double tripY) {
	int stopID;
	//先判断是不是地铁站点，是就去掉后面“-数字”和“站”字并加一个线路数来用map匹配
	//如果不是地铁站点，那就直接map，map不到就加a和b来map
	if(isSubwayStop(str1))                       //如果是地铁站点的名字
	{
		string tempSub=str1;                     //后面输到函数里去要改变它,存去了“-数字”和“站”字的地铁名字
		tempSub=extractSubwayName(tempSub);              //把地铁站名去掉了站字，并加上了不带-的数字
		if(oneWStopNameToID.count(tempSub)!=0)           //再判断此时那个一万站点的map中有没有这个站点存在
		{
			stopID = oneWStopNameToID[tempSub];                       //类似：用 益田3访问到是一万个中的第几
		}
		else
		{
			//cout<<"出现了明明是trip的地铁站，但是一万个里没有的情况："<<tempSub<<endl;
			stopID = -66;
		}
	}
	else                                                 //如果是公交站点的名字
	{
		if(oneWStopNameToID.count(str1)!=0)               //先在不加a和b的情况下，直接判断一万个站点里有没有这个站点名
		{
			stopID = oneWStopNameToID[str1];                       //类似：用 益田3访问到是一万个中的第几
		}
		else                                                      //如果直接map没有，那就加个a和b来判断
		{
			if(oneWStopNameToID.count(str1+"a")!=0 && oneWStopNameToID.count(str1+"b")!=0)       //理论上来说，+a有，+b肯定也有
			{
				//分别求trip的坐标到名字为站点a的距离和到名字为站点b的距离
				//注意这里减一，oneWStopNameToID索引到的ID从1开始，oneWX从0开始
				double d1=getTwoPointDistance(tripX,tripY,oneWX[oneWStopNameToID[str1+"a"]-1],oneWY[oneWStopNameToID[str1+"a"]-1]);
				double d2=getTwoPointDistance(tripX,tripY,oneWX[oneWStopNameToID[str1+"b"]-1],oneWY[oneWStopNameToID[str1+"b"]-1]);
				if(d1<=d2)               //如果离名字为站点a更近
				{
					stopID = oneWStopNameToID[str1+"a"];
				}
				else
				{
					stopID = oneWStopNameToID[str1+"b"];
				}
			}
			else                             //加a和加b后都没有，表示这个站点可能就在一万站点里没有，可能是比较新的站点
			{
				stopID = -77;
			}
		}
	}
	return stopID;
}
double getMultiType::getTwoPointDistance(double x1,double y1,double x2,double y2)
{
	return sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
}
int getMultiType::convertPairStopToEdgeType(int ID1, int ID2) {
	if (ID1 >= 1 && ID1 <= 10626 && ID2 >= 1 && ID2 <= 10626) {        //确保是在站点范围内得编号
		if (ID1 <= 199) {                                   //确保是地铁开头
			if (ID2 <= 199) {    //地铁开头地铁结尾
				return 1;
			}
			else {               //地铁开头公交结尾
				return 2;
			}
		}
		else {                                              //公交开头
			if (ID2 <= 199) {    //公交开头地铁结尾
				return 3;
			}
			else {               //公交开头公交结尾
				return 4;
			}
		}
	}
	else {                     //那种地铁站点没找到的，或者是太新的trip的公交站点
		return 0;
	}
}

void getMultiType::fillTransferProbMatrix(){
	cout<<"fillTransferProbMatrix"<<endl;
	//先初始化所有站点转移概率为0
	vector<int> temp;                                     //暂时vector
	vector<double> countTransferFront;                            //暂时存储每一行的199以前的次数
	vector<double> countTransferBack;                             //暂时存储每一行的199以后的次数
	vector<double> countAccumulateCost;                           //暂时存储累计总出行时间
	for(unsigned int j=0;j<oneWStopName.size();j++) {
		temp.push_back(0);
		countTransferFront.push_back(0);
		countTransferBack.push_back(0);
		countAccumulateCost.push_back(0);
	}
	for(unsigned int i=0;i<oneWStopName.size();i++) {
		transferProbability.push_back(temp);
	}
	//遍历headID这个数组，也就是边的数量的数组大小，将实际转移次数填充进去
	for(unsigned int i=0;i<headID.size();i++) {
		transferProbability[headID[i]-1][tailID[i]-1]=occurCount[i];
		countAccumulateCost[headID[i]-1] += (timeAccumulation[i]/60.0);
	}
	//累计分母的次数
	cout<<"start storing transfer times file..."<<endl;
	ofstream outfile2;
	outfile2.open("..\\result\\20170405_countTransfer"+to_string(long long(singleHour))+"-"
		+to_string(long long(singleHour+durationLast))+".csv");

	for(unsigned int i=0;i<transferProbability.size();i++) {
		for(unsigned int j=0;j<transferProbability[0].size();j++) {
			if (j<199) {
				countTransferFront[i] += transferProbability[i][j];
			}
			else {
				countTransferBack[i] += transferProbability[i][j];
			}
		}
		int tempSum = int(countTransferFront[i]) + int(countTransferBack[i]);
		if (tempSum > 10000) {
			cout<<"the station transfer times is bigger than 10k is: "<<oneWStopName[i]<<"  the whole times is:"<<tempSum;
			cout<<"went to subway"<<countTransferFront[i]<<" went to bus"<<countTransferBack[i]<<endl;
		}
		outfile2<<fixed<<setprecision(2)<<tempSum<<","<<countAccumulateCost[i]<<endl;
	}
	outfile2.close();
	cout<<"the transfer of O is finished now"<<endl;

	//根据次数除以总次数计算转移概率
	cout<<"start storing transferProbability matrix..."<<endl;
	return;   //因为转移概率直接在python里算了，下面就return跳过吧，用不到了

	/*
	ofstream outfile1;
	outfile1.open("D:\\data\\深圳数据\\MDNE\\output\\20170405_transferProbability.csv");
	int times = 1;
	for(unsigned int i=0;i<transferProbability.size();i++) {
		for(unsigned int j=0;j<transferProbability[0].size();j++) {
			//算概率的话，transferProbability得是double型
			if (j<199 && countTransferFront[i]>0) {
				transferProbability[i][j] /= countTransferFront[i];
			}
			if (j>=199 && countTransferBack[i]>0){
				transferProbability[i][j] /= countTransferBack[i];
			}
			//输出到csv
			if (j == transferProbability[0].size()-1) {
				outfile1<<fixed<<setprecision(4)<<transferProbability[i][j]<<endl;
			}
			else {
				outfile1<<fixed<<setprecision(4)<<transferProbability[i][j]<<",";
			}
		}
		if(i==times*500)
		{
			times++;
			cout<<i<<endl;
		}
	}
	outfile1.close();
	cout<<"OD转移概率输出完毕！！！！"<<endl;
	*/
}


void getMultiType::makeTransferCount() {
	cout<<"start reading transfer type..."<<endl;
	string transferLine;                                                            //暂时存放换乘字段的信息
	int transferTimes = 0;                                                          //统计发生换乘的trip有多少条
	int straightArrive = 0;                                                         //统计直达的次数
	int abnormal = 0;                                                               //统计换乘字段异常
	int tooNew = 0;                                                                 //统计太新的站点
	int disorder = 0;                                                               //统计换乘字段本身出现异常的，不是4、7、10。。。
	int times = 1;                                                                  //看速度
	for (unsigned int i=0;i<transferField.size();i++) {                             //这个就很大了，for的是trip的次数
		transferLine = transferField[i];
		int should_only_run_one_time = extractTransferInfo(transferLine);            //致命错误，如果把函数直接写在if判断中，那会执行多次
		if(should_only_run_one_time==1) {
			transferTimes++;
		}
		else if(should_only_run_one_time==0) {
			straightArrive++;
		}
		else if(should_only_run_one_time==2) {
			abnormal++;
			cout<<"abnormal row is behind: "<<i<<"  it is："<<transferLine<<endl;
		}
		else if(should_only_run_one_time==3) {
			tooNew++;
		}
		else if(should_only_run_one_time==4) {
			disorder++;
		}
		else {
			cout<<"cannot show up!!!"<<endl;
		}
		if(i==times*100000)
		{
			times++;
			cout<<i<<endl;
		}
	}
	seeEdgeTypeNum(transfer_type);                  //看一下各类别有多少
	cout<<"transfer occurence num is "<<transferTimes<<"  straight arrive num is "<<straightArrive<<endl;
	cout<<"weird reading problem(磡) has："<<abnormal<<"  too new to read: "<<tooNew<<"  disordered transfer field: "<<disorder<<endl;
}
int getMultiType::extractTransferInfo(string str1) {
	string head;                        //暂时存储换乘选择的头节点
	string tail;                        //暂时存储换乘选择的尾节点
	istringstream sin(str1);   
	string field;
	vector<string> vecTemp1;             //暂时把每个都放进来，然后取
	string frontHead = "0";              //暂时存放“|”前的站点
	while (getline(sin,field,'|')) {
		vecTemp1.push_back(field);
	}
	if (vecTemp1.size()==1) {           //不存在跨交通工具换乘或者公交换乘
		istringstream sin(vecTemp1[0]);
		vector<string> vecTemp2;
		while (getline(sin,field,'-')) {
			vecTemp2.push_back(field);
		}
		if(vecTemp2.size()==4) {        //如果是直达，中间无换乘
			return 0;
		}
		else if(vecTemp2.size()>4) {                         //不是直达，但是有多个，那肯定是地铁多次乘车
			int tempHere = extractLongTransferInfo(frontHead,vecTemp2);
			if(tempHere==3) {
				return 3;
			}
			if(tempHere==4) {
				return 4;
			}
			return 1;
		}
		else {                                        //为了排除异常
			cout<<"transfer field is less than regular 4"<<endl;
			return 2;
		}
	}
	else {                               //存在跨交通工具或者公交换乘
		for (unsigned int i=0;i<vecTemp1.size();i++) {
			istringstream sin(vecTemp1[i]);
			vector<string> vecTemp2;
			while (getline(sin,field,'-')) {
				vecTemp2.push_back(field);
			}
			if(vecTemp2.size()==4 && i==0) {        //如果是直达，中间无换乘,那把第一段的最后一个节点提出来
				frontHead = getSubwayStationName(vecTemp2[1],vecTemp2[3]);
			}
			else if (vecTemp2.size()==4 && i!=0) {  //如果是直达，中间无换乘,和上一段的最后一个可构成插入
				head = frontHead;
				tail = getSubwayStationName(vecTemp2[1],vecTemp2[0]);
				if(!insertTransfer(head, tail)) {
					//cout<<"太新了："<<head<<"--"<<tail<<endl;
					return 3;
				}
				frontHead = getSubwayStationName(vecTemp2[1],vecTemp2[3]);
			}
			else if (vecTemp2.size()>4){     //不是直达，但是有多个，那肯定是地铁多次乘车,要么是第一段，要么是后面的段
				int temp_longInfo = extractLongTransferInfo(frontHead,vecTemp2);
				if(temp_longInfo==3) {
					return 3;
				}
				if(temp_longInfo==4) {
					return 4;
				}
				frontHead = getSubwayStationName(vecTemp2[vecTemp2.size()-3],vecTemp2[vecTemp2.size()-1]);
			}
			else {                                              //为了排除异常
				cout<<"this transfer filed has big trouble, unable to read 磡"<<endl;       //可能分辨不了“大磡安居苑” 磡字
				/*cout<<vecTemp1[i]<<endl;
				cout<<vecTemp2.size()<<endl;*/
				return 2;
			}
			vecTemp2.clear();
		}
		return 1;
	}
}
int getMultiType::extractLongTransferInfo(string str1, vector<string> vec) {
	string head;                        //暂时存储换乘选择的头节点
	string tail;                        //暂时存储换乘选择的尾节点
	if (str1 != "0") {                  //如果前面有换乘站点
		head = str1;
		tail = getSubwayStationName(vec[1],vec[0]);
		if(!insertTransfer(head, tail)) {
			//cout<<"有“|”时太新了："<<head<<"--"<<tail<<endl;
			return 3;
		}
	}
	if ((vec.size()-4)%3!=0) {
		//cout<<"居然不是4、7、10、13！！！"<<endl;
		return 4;
	}
	int subwayTimes = (vec.size()-4)/3;   //计算换乘信息里有多少次(n-4)/3 + 1，如果是中间的只有4个，那换乘就会在两头   
	int leftLine = 1;
	int middleStation = 3;
	int rightLine = 4;
	for(int i=0;i<subwayTimes;i++) {
		head = getSubwayStationName(vec[leftLine+i*3],vec[middleStation+i*3]);
		tail = getSubwayStationName(vec[rightLine+i*3],vec[middleStation+i*3]);
		if(!insertTransfer(head, tail)) {
			//cout<<"extractLongTransferInfo里太新了："<<head<<"--"<<tail<<endl;
			return 3;
		}
	}
	return 1;
}
string getMultiType::getSubwayStationName(string line, string name) {
	string tempyLine; 
	if (subwayLineNumber.count(line)!=0){  //得到换乘信息里第二个位置是不是地铁字样
		tempyLine=subwayLineNumber[line];
	}
	else{
		tempyLine = "";                             //如果不是地铁字样，那就不用加如：-2
	}
	return name+tempyLine;
}
bool getMultiType::insertTransfer(string head, string tail) {
	int tripHeadID;
	int tripTailID;
	int tripEdgeType;
	pair<map<EdgeHeadTail,int>::iterator,bool> Insert_Pair2;    //为了唯一性存边类型-头节点编号-尾节点编号
	tripHeadID = convertStopNameToID(head,0,0);
	tripTailID = convertStopNameToID(tail,0,0);
	tripEdgeType = convertPairStopToEdgeType(tripHeadID, tripTailID);
	if (tripEdgeType != 0) {
		//把边关系-头节点-尾节点-时间-次数，存起来
		EdgeHeadTail temp(tripEdgeType, tripHeadID, tripTailID);
		Insert_Pair2 = EHT_TransferIndex.insert(pair<EdgeHeadTail,int>(temp,transfer_headID.size()));  //为了让value值从0开始索引
		if(Insert_Pair2.second==false)                          //插不进来就代表这种边关系下已经存在过该头节点-尾节点的关系了
		{
			int numNow = EHT_TransferIndex[temp];
			transfer_occurCount[numNow]++;
		}
		else {                                                 //能插进来就代表是新的
			transfer_type.push_back(tripEdgeType);
			transfer_headID.push_back(tripHeadID);
			transfer_tailID.push_back(tripTailID);
			transfer_occurCount.push_back(1);
		}
		return true;
	}
	else {
		return false;
	}
}
void getMultiType::fillTransferCountMatrix() {
	//根据次数输出文件
	cout<<"start outputing transfer occurence matrix..."<<endl;
	int times = 1;                                        //看输出速度
	ofstream outfile1;
	outfile1.open("..\\result\\20170405_transferCount"+to_string(long long(singleHour))+"-"
		+to_string(long long(singleHour+durationLast))+".csv");

	//因为line模型里输入的是一行一行的数据
	for(map<EdgeHeadTail,int>::iterator it=EHT_TransferIndex.begin(); it!=EHT_TransferIndex.end(); it++) //这样遍历确保了输出按照排序输出
	{
		//MNE读入换乘强度的保存方式
		outfile1<<fixed<<setprecision(6)<<transfer_type[it->second]<<","<<transfer_headID[it->second]<<","<<transfer_tailID[it->second]<<",";
		outfile1<<fixed<<setprecision(6)<<transfer_occurCount[it->second]<<","<<0<<endl;
		//LINE模型的输出方式
		/*outfile1<<fixed<<setprecision(6)<<transfer_headID[it->second]<<" "<<transfer_tailID[it->second]<<" "
		<<transfer_occurCount[it->second]<<endl;*/

	}
	outfile1.close();
	cout<<"transfer occurence matrix output done"<<endl;
	return;
}

void getMultiType::run(){
	makeSubwayLineMap();                                                                      //把那个可以根据string索引的map制作好
	ifstream finTrip("..\\data\\20170405-Trip-statistics-2CN-duration_example.csv");                  //存有所有trip的数据，大概360万条
	readODTrip(finTrip);																	//读取完trip
	finTrip.close();
	readBusStop();                                                                           //读取10626个站点进来
	makeStopStartCount();                                                                    //开始计算边的关系
	fillTransferProbMatrix();                                                                //开始计算转移矩阵概率
	makeTransferCount();                                                                     //先遍历一遍所有trip的换乘字段
	fillTransferCountMatrix();                                                              //必须是在读完所有trip再输出这个换乘联系强度

}
