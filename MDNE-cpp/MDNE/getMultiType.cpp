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
	singleHour=5;                              //��Ч�Ŀ�ʼʱ��
	durationLast=15;                            //singleHour~��singleHour+durationLast��һ��Ϊ�����ʱ���
}


getMultiType::~getMultiType(void)
{
}

void getMultiType::makeSubwayLineMap() {
	pair<map<string,string>::iterator,bool> Insert_Pair;
	Insert_Pair=subwayLineNumber.insert(pair<string,string>("����һ����","-1"));
	insertFailOnMap(Insert_Pair.second);
	Insert_Pair=subwayLineNumber.insert(pair<string,string>("����������","-2"));
	insertFailOnMap(Insert_Pair.second);
	Insert_Pair=subwayLineNumber.insert(pair<string,string>("����������","-3"));
	insertFailOnMap(Insert_Pair.second);
	Insert_Pair=subwayLineNumber.insert(pair<string,string>("�����ĺ���","-4"));
	insertFailOnMap(Insert_Pair.second);
	Insert_Pair=subwayLineNumber.insert(pair<string,string>("���������","-5"));
	insertFailOnMap(Insert_Pair.second);
	Insert_Pair=subwayLineNumber.insert(pair<string,string>("�����ߺ���","-7"));
	insertFailOnMap(Insert_Pair.second);
	Insert_Pair=subwayLineNumber.insert(pair<string,string>("�����ź���","-9"));
	insertFailOnMap(Insert_Pair.second);
	Insert_Pair=subwayLineNumber.insert(pair<string,string>("����ʮһ����","-11"));
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

	double sOX,sOY,sDX,sDY;                                        //Ϊ����vector���������������ǰ��ȫ������s����Щ�����ڶ���Ϣ��ʱ���õ��м�ֵ
	int tempOtime,tempDtime,tempOhour,tempDhour;                   //�ֱ�洢ת��������³�������Сʱ��
	int stops,transfers;                                           //�洢���trip���˶��ٸ�վ��ͻ��˴���
	string OName,DName;                                            //�洢trip�����յ������
	string OSubwayLine;                                            //Ϊ�˶�ȡ������Ϣ���м������Ϊ��ȷ����������ڵĵ�������·��
	string DSubwayLine;
	string tempTransField;
	double tempDuration;                                                            //��ʱ��Ŷ���ʱ��
	

	string line;
	int times=1;                                                                    //���������е��ٶ�
	double tStop=pow(10.0,7);                                                       //���ƶ�������
	int countNight=0;                                                               //�洢���賿��trip����
	int timeTooShortCount=0;                                                        //ʱ����̫�̻�̫��
	int number=0;                                                                   //�ۻ�����
	int durationInvalid6677=0;                                                      //ͳ�Ƴ�����ʱ�䲻���������ж��ٸ�
	
	while(getline(fin,line))    //һ��һ�еĶ� 
	{
		if(line=="end,")
		{
			switch(tripDay)
			{
			case 2:
				{
					//˵�����ڶ�ȡ������ʱ���ǰ���4��5�š�6�š�7�������еģ���ȡ�ǹ������ǰ���4��8�š�9�š�3�š�4�������е�

					ifstream finTrip2("..\\data\\20170406-Trip-statistics-2CN-duration_example.csv");//4��6����������
					cout<<"now is the "<<tripDay<<"day"<<"   case 2 already"<<endl;
					tripDay++;
					readODTrip(finTrip2);	
					finTrip2.close();
					break;
				}
			case 3:
				{
					ifstream finTrip3("D:\\data\\��������\\TRIP20180707\\CN\\CN2\\20170407-Trip-statistics-2CN-duration.csv");//4��7����������
					cout<<"now is the "<<tripDay<<"day"<<"   case 3 already"<<endl;
					tripDay++;
					//readODTrip(finTrip3);
					finTrip3.close();
					break;
				}
			case 4:
				{
					ifstream finTrip4("D:\\data\\��������\\TRIP20180707\\CN\\CN2\\20170404-Trip-statistics-2CN-duration.csv"); //4��8����������
					cout<<"now is the "<<tripDay<<"day"<<"   case 4 already"<<endl;
					tripDay++;
					//readODTrip(finTrip4);
					finTrip4.close();
					break;
				}
			case 5:
				{
					ifstream finTrip5("D:\\data\\��������\\TRIP20180707\\CN\\CN2\\20170409-Trip-statistics-2CN-duration.csv"); //4��9����������
					cout<<"now is the "<<tripDay<<"day"<<"   case 5 already"<<endl;
					tripDay++;
					//readODTrip(finTrip5);
					finTrip5.close();
					break;
				}
			case 6:
				{
					ifstream finTrip6("D:\\data\\��������\\TRIP20180707\\CN\\CN2\\20170404-Trip-statistics-2CN-duration.csv");//4��4�������ڶ�
					cout<<"now is the "<<tripDay<<"day"<<"   case 6 already"<<endl;
					tripDay++;
					//readODTrip(finTrip6);
					finTrip6.close();
					break;
				}
			case 7:
				{
					ifstream finTrip7("D:\\data\\��������\\TRIP20180707\\CN\\CN2\\20170403-Trip-statistics-2CN-duration.csv"); //4��3��������һ
					cout<<"now is the "<<tripDay<<"day"<<"   case 7 already"<<endl;                 //һ���԰���������ݶ����������ڴ���ܷŲ���
					tripDay++;
					//readODTrip(finTrip7);
					finTrip7.close();
					break;
				}
			default:
				{
					cout<<"reading trip switch fails����������������"<<endl;
					break;
				}
			}
			continue;
		}

		istringstream sin(line);   
		string field;
		int i=1;
		while (getline(sin,field,','))   //��vector��ȡʱ�����CSV�ļ����ֶγ���5��ʱ���ܻ����
		{
			if(i==2)          //ֻ����row�е����ݣ�Ȼ��push_back��vector��
			{
				tempOtime=convertTimeStringToInt(field);     //��Otime����
				tempOhour=extractTimeHour(field);            
				i++;
				continue;
			}
			else if(i==3)
			{
				OName=field;                           //������վ������
				i++;
				continue;
			}
			else if(i==4)
			{
				sOX=convertStringToDouble(field);     //��OX����
				i++;
				continue;
			}
			else if(i==5)
			{
				sOY=convertStringToDouble(field);     //��OY����
				i++;
				continue;
			}
			else if(i==6)
			{
				tempDtime=convertTimeStringToInt(field);     //��Dtime����
				tempDhour=extractTimeHour(field);
				i++;
				continue;
			}
			else if(i==7)
			{
				DName=field;                           //���յ��վ������
				i++;
				continue;
			}
			else if(i==8)
			{
				sDX=convertStringToDouble(field);     //��DX����
				i++;
				continue;
			}
			else if(i==9)
			{
				sDY=convertStringToDouble(field);     //��DY����
				i++;
				//break;
				continue;
			}
			else if(i==11)                             //�Ż�����Ϣ��һ�н���
			{
				OSubwayLine=field;    
				OSubwayLine=extractSubwayLine(OSubwayLine); 
				if (subwayLineNumber.count(OSubwayLine)!=0){  //�õ�������Ϣ��ڶ���λ���ǲ��ǵ�������
					OSubwayLine=subwayLineNumber[OSubwayLine];
				}
				else{
					OSubwayLine = "";          //������ǵ����������ǾͲ��ü��磺-2
				}
				DSubwayLine=field;    
				DSubwayLine=extractLastSubwayLine(DSubwayLine); 
				if (subwayLineNumber.count(DSubwayLine)!=0){  //�õ�������Ϣ��ڶ���λ���ǲ��ǵ�������
					DSubwayLine=subwayLineNumber[DSubwayLine];
				}
				else{
					DSubwayLine = "";          //������ǵ����������ǾͲ��ü��磺-2
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
				if(transfers>5)     //�����쳣���˴�������������˳���5�Σ���Ĭ��������5�Σ������ٴ���
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
		number++;                                                             //�ۻ�����
		if(number==times*100000)
		{
			cout<<number<<endl;     
			times++;
		}

		if(tempDhour<tempOhour)                                               //Ҫ�����³�ʱ�������ϳ�ʱ���trip���������ɣ����������һ��������
		{
			countNight++; 
			cout<<"the "<<number<<" row has boarding time conflicts"<<endl;
			continue;                                                   
		}
		
		//�����ǿ��ƶ�ȡ���������ݵ�ʱ���
		//if(tempOhour>=0 && tempOhour<=24)                                      //����ʱ�ζ�Ҫ
		//if(tempOhour>=singleHour && tempOhour<singleHour+1)                    //�����ʱ��push_back��vector��,�����ʱ�ξ͸�һ����ֵ�ͺ�
		//if(tempDhour>=singleHour && tempDhour<24 || tempDhour>=0 && tempDhour<2)   //����Ŀ��賿
		if(tempDhour>=singleHour && tempDhour<singleHour+durationLast)            //�����ʱ��push_back��vector��,�����ʱ�ξ͸�һ����ֵ�ͺ�
		{
			if(tempDtime-tempOtime<200||tempDtime-tempOtime>12600)       //�����С��0��ʱ���������Ǽ��̣�200s����ʱ�����Ͳ�Ҫ�������
			{                                                            //������յ�ʱ�䳬��3.5h��Ҳ��Ҫ��
				timeTooShortCount++;
				continue;
			}
			if(tempDuration==0.66 || tempDuration==0.77)     //���Ƕ�����ͬ���ŵ����һ����û�ؼһ������ǵ�����һ��trip
			{
				durationInvalid6677++;
				continue;
			}

			OStopName.push_back(OName+OSubwayLine);                      //��վ��Ҳ�Ž���,���Һ���Ӹ���·��
			DStopName.push_back(DName+DSubwayLine);
			OGrid_X.push_back(sOX);                                         //����ʼվ�������Ž���
			OGrid_Y.push_back(sOY); 
			DGrid_X.push_back(sDX);      
			DGrid_Y.push_back(sDY);   
			transferField.push_back(tempTransField);                     //�Ѿ޴�Ļ����ֶ�Ҳ�Ž���
			timeDiff.push_back(tempDtime-tempOtime+tempOhour/100.0);     
			ODStopNumber.push_back(stops+transfers/10.0);
			tripDurationTime.push_back(tempDuration);                    //�Ѷ���ʱ��Ž���
		}

		if(number>2.0*tStop)     //���number�����                                                           
		{
			cout<<"break in advance"<<endl;
			break;
		}
	}

	cout<<"the amount of trip that push_back to vector��"<<timeDiff.size()<<"--"<<OStopName.size()<<"    rows��"<<number<<endl;
	cout<<"get off time wrong, unable push to trip��"<<countNight<<endl;
	cout<<"the boarding duration is either too long or too short, unable push to trip��"<<timeTooShortCount<<endl;
	cout<<"duration time is not good��"<<durationInvalid6677<<endl;
	cout<<"trip reading finished"<<endl<<endl;

	cout<<"trip start station name based accumulation reading over"<<endl<<endl;
}
string getMultiType::extractSubwayLine(string str1)  //һ���ܵ��Ǹ�trip�ĵ�11���ֶεĻ�����Ϣ����Ӣ�ģ��޷��жϼ��������֣�������������
{
	string sub;                        //�洢�������ĵ��������ֶ�
	//�ο���ʱ��ķ���
	istringstream sin(str1);   
	string field;
	int i=0;
	while (getline(sin,field,'-')) 
	{
		if(i==1)     //��Ϊ������·���ڻ����ֶεĵڶ���
		{
			sub=field;
			break;
		}
		i++;
	}
	return sub;
}
string getMultiType::extractLastSubwayLine(string str1) {
	string sub;                        //�洢�������ĵ��������ֶ�
	istringstream sin(str1);   
	string field;
	vector<string> vecTemp;            //��ʱ��ÿ�����Ž�����Ȼ��ȡ��������������
	while (getline(sin,field,'-')) 
	{
		vecTemp.push_back(field);
	}
	sub = vecTemp[vecTemp.size()-3];
	return sub;
}
double getMultiType::convertStringToDouble(string str1)
{
	//����stringתdouble
	double n1;
	const char* ch= str1.c_str();
	n1=atof(ch);                   //�����Ӣ���ַ�����ת��������0��������һ��double����

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
		if(i==1)     //��Ϊʱ���ڵڶ���λ�ã���һ��λ����������
		{
			istringstream sin(field);
			string field2;
			int j=0;
			while (getline(sin,field2,':'))
			{
				if(j==0)
				{
					double seeHour=convertStringToDouble(field2);                     //�����ж϶�������Сʱ��
					if(seeHour<5 && seeHour>=0)
					{
						second=second+(convertStringToDouble(field2)+24)*3600;
					}
					else
					{
						second=second+convertStringToDouble(field2)*3600;   //�ȶ���Сʱ�������������ۼ�
					}

					j++;
					continue;
				}
				if(j==1)
				{
					second=second+convertStringToDouble(field2)*60;    //�������ӣ������������ۼ�
					j++;
					continue;
				}
				if(j==2)
				{
					second=second+convertStringToDouble(field2);      //ֱ�Ӷ�����
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
				hour=int(convertStringToDouble(field2));   //�ȶ���Сʱ��
				break;
			}
		}
		i++;
	}

	return hour;
}
void getMultiType::readBusStop()
{
	string name;        //��վ�����ֵ��м����
	int ID=1;           //��վ���ID�ţ��������ȡ��������û�У�һ��һ���ۼӼ���,��1��ʼ
	double X,Y;         //��������м����
	ifstream finBus("..\\data\\merge_BusStation�Ľ�dbscan�ӵ�����ǰ��.csv");   //���кϲ����վ������
	string line;                                   //��Ȼ����ĺϲ����վ��Ҳ����һ�����⣺վ�����о������
	ifstream &fin=finBus;
	while(getline(fin,line))    //һ��һ�еĶ� 
	{
		istringstream sin(line);   
		string field;
		int i=1;
		while (getline(sin,field,','))   //��vector��ȡʱ�����CSV�ļ����ֶγ���5��ʱ���ܻ����
		{
			if(i==1)          //���ƶ���һ��
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
	for(unsigned int i=0;i<oneWStopName.size();i++)          //for����һ���վ��
	{
		//��������վ�������������ڵ�λ�úţ�Ϊ����ͼ�����˳�ʹ�1��ʼ��
		Insert_Pair=oneWStopNameToID.insert(pair<string,int>(oneWStopName[i],i+1));
		if(Insert_Pair.second==false)
		{
			cout<<"oneWStopNameToID insert falied, how come?"<<endl;
		}
	}

	int tripHeadID;                                         //������ÿһ��trip�����ID
	int tripTailID;
	int tripEdgeType;                                       //trip�ߵ�����
	int noFind=0;                                           //�ۻ���Щtrip���е�һ���վ����û�е����
	int yesFind=0;                                          //�ۻ���Щtrip����һ���վ����Ҳ�е����
	int times=1;                                            //��ִ�е��ٶ�
	pair<map<EdgeHeadTail,int>::iterator,bool> Insert_Pair2;//Ϊ��Ψһ�Դ������-ͷ�ڵ���-β�ڵ���
	int num = 0;                                            //��¼�嵽map EHTIndex�������

	vector<int> durOccurCount;                       //��Ч�ıߵ���Ч����ʱ����ֵĴ���
	vector<double> durAccumulation;                  //�����Ч�Ķ���ʱ����ۼ�

	//�Ե���ĳ������õص�ƽ������ʱ���ƽ���г�ʱ�䣬��С��1���վ��ĳ���
	vector<double> overallAverageTime;
	vector<double> overallAverageDur;
	vector<int> overallTimeCount;                    //���յ�Ϊ����ĵ������
	vector<int> overallDurCount;                     //���յ�Ϊ�������Ч����ʱ��Լ���ĵ���������������ֵС
	for(unsigned int i=0;i<oneWStopName.size();i++) {     //for����һ���վ��
		overallAverageTime.push_back(0);
		overallAverageDur.push_back(0);
		overallTimeCount.push_back(0);
		overallDurCount.push_back(0);
	}

	cout<<endl<<endl<<"the later ID of trip is based on this number, index start from 0, the largest one is:"<<OStopName.size()<<endl<<endl;
	cout<<"statistic of OD edge type..."<<endl;
	for(unsigned int i=0;i<OStopName.size();i++)           //for�����е�trip������ͺܴ���
	{
		tripHeadID = convertStopNameToID(OStopName[i],OGrid_X[i],OGrid_Y[i]);
		tripTailID = convertStopNameToID(DStopName[i],DGrid_X[i],DGrid_X[i]);
		tripEdgeType = convertPairStopToEdgeType(tripHeadID, tripTailID);
		if (tripEdgeType != 0) {
			//�ۼ����յ�Ϊ������г�ʱ��Ͷ���ʱ��
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
			//�ѱ߹�ϵ-ͷ�ڵ�-β�ڵ�-ʱ��-������������
			EdgeHeadTail temp(tripEdgeType, tripHeadID, tripTailID);
			Insert_Pair2 = EHTIndex.insert(pair<EdgeHeadTail,int>(temp,headID.size()));  //Ϊ����valueֵ��0��ʼ����
			if(Insert_Pair2.second==false)              //�岻�����ʹ������ֱ߹�ϵ���Ѿ����ڹ���ͷ�ڵ�-β�ڵ�Ĺ�ϵ��
			{
				int numNow = EHTIndex[temp];
				//�ڶ��η��ʵ������ʱ���г�ʱ�����
				occurCount[numNow]++;
				timeAccumulation[numNow] += timeDiff[i];
				averageTime[numNow] = timeAccumulation[numNow]/occurCount[numNow];
				//����ʱ��ļ���
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
			else {                                                 //�ܲ�����ʹ������µ�
				//������-ͷ-β
				edgeType.push_back(tripEdgeType);
				headID.push_back(tripHeadID);
			    tailID.push_back(tripTailID);
				//�ñ��µ��г�ʱ��ͳ��
				occurCount.push_back(1);
				timeAccumulation.push_back(timeDiff[i]);
				averageTime.push_back(timeAccumulation[num]/occurCount[num]);
				//�ñ��µĶ���ʱ��ͳ��
				if(tripDurationTime[i]>1) {                 //�������ʱ�����1���������ж����Ļ��������
					durOccurCount.push_back(1);
					if(tripDurationTime[i]==180.6) {        //180.6��֮ǰ�ܶ���ʱ���ʱ��������ֶ���ʱ��Ϊ��ֵ�����				
						durAccumulation.push_back(15.0);    //Ĭ�ϼ������̶���ʱ��Ϊһ����
						averageDuration.push_back(durAccumulation[num]/durOccurCount[num]);
					}
					else {
						durAccumulation.push_back(tripDurationTime[i]/60.0);    //�����Ķ���ʱ��,�Է�������
						averageDuration.push_back(durAccumulation[num]/durOccurCount[num]);
					}
				}
				else {
					durOccurCount.push_back(0);
					durAccumulation.push_back(0);    //û�ж���ʱ��
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
	cout<<"in the pushed trip that oneW does not have��noFind is equal to�� "<<noFind<<endl;
	cout<<"trip num is�� "<<OStopName.size()<<endl;
	cout<<"check my map -- oneWStopNameToID(10626) is my map broken: "<<oneWStopNameToID.size()<<endl;
	cout<<"check my map -- EHTIndex(same as num) is my map broken��"<<EHTIndex.size()<<endl;
	cout<<"  num is at this large��"<<num<<"the size of edgeType is��"<<edgeType.size()<<endl;

	seeEdgeTypeNum(edgeType);                  //��һ�¸�����ж���

	//��һ��ÿ�����յ�Ϊ��ʾ��ƽ������ʱ����г�ʱ��
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
	for(map<EdgeHeadTail,int>::iterator it=EHTIndex.begin(); it!=EHTIndex.end(); it++) //��������ȷ������������������
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
		//Ϊ���ó��ִ���Ϊ1��Ҳ���Ǳߵ�Ȩ��Ϊ1��MNE���ܲ��ʺ���Ȩ�صı�
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
	//���жϳ��ȹ�������Ҫ��Ȼ��������
	int lastP=str1.size()-1;                             //��������ַ��������һ��λ��
	if(str1.size()<3)                                    //�������ַ������ȹ�С������ľͲ��ÿ�����
	{
		return false;
	}
	//���ж����һ���ǲ�������
	if(str1[lastP]=='1'|| str1[lastP]=='2' || str1[lastP]=='3' || str1[lastP]=='4' || str1[lastP]=='5' || str1[lastP]=='7'|| str1[lastP]=='9')
	{
		if(str1[lastP-1]=='-' || str1[lastP-1]=='1')     //�����һ�������ֵ�����£������ڶ������� - ���� 1 ����ô�϶����ǵ���վ����
		{
			return true;
		}
		else
		{
			cout<<"the last one in the string is number, but the one before the last one is not��"<<str1<<endl;
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
	int lastP=str1.size()-1;                             //��������ַ��������һ��λ��
	string tempStr;
	string zhan="վ";
	if(str1[lastP-1]=='-')
	{
		tempStr=str1.substr(str1.size()-4,2);                    //��ȡ�Ǹ��ֿ��ǲ��ǡ�վ����
		if(tempStr==zhan)                                        //�����վ�֣��Ǿ�ȥ����
		{
			finalString=str1.substr(0,str1.size()-4);            //������վ����֮ǰ������
			return finalString+str1[lastP];                      //�����Ǹ����֣������ܴ�   ����վ-3  ���  ����3
		}
		else                        //�������վ�֣��Ǿ�ֱ�Ӽ�����·�ź����
		{
			finalString=str1.substr(0,str1.size()-2);
			return finalString+str1[lastP]; 
		}
	}
	else                           //����Ĭ�ϲ��ǡ�-�����ǡ�1������Ϊǰ���Ѿ�ǧ����ȷ�жϳ������Ǹ�����վ��������
	{
		if(str1[lastP-1]!='1')
		{
			cout<<"the one before the last one is not '1'? why��"<<str1<<endl;
		}
		tempStr=str1.substr(str1.size()-5,2);                    //��ȡ�Ǹ��ֿ��ǲ��ǡ�վ����
		if(tempStr==zhan)                                        //�����վ�֣��Ǿ�ȥ����
		{
			finalString=str1.substr(0,str1.size()-5);            //������վ����֮ǰ������
			return finalString+str1[lastP-1]+str1[lastP];        //�����Ǹ����֣������ܴ�   ����վ-11  ���  ����11
		}
		else                        //�������վ�֣��Ǿ�ֱ�Ӽ�����·�ź����
		{
			finalString=str1.substr(0,str1.size()-3);
			return finalString+str1[lastP-1]+str1[lastP]; 
		}
	}
}
int getMultiType::convertStopNameToID(string str1, double tripX, double tripY) {
	int stopID;
	//���ж��ǲ��ǵ���վ�㣬�Ǿ�ȥ�����桰-���֡��͡�վ���ֲ���һ����·������mapƥ��
	//������ǵ���վ�㣬�Ǿ�ֱ��map��map�����ͼ�a��b��map
	if(isSubwayStop(str1))                       //����ǵ���վ�������
	{
		string tempSub=str1;                     //�����䵽������ȥҪ�ı���,��ȥ�ˡ�-���֡��͡�վ���ֵĵ�������
		tempSub=extractSubwayName(tempSub);              //�ѵ���վ��ȥ����վ�֣��������˲���-������
		if(oneWStopNameToID.count(tempSub)!=0)           //���жϴ�ʱ�Ǹ�һ��վ���map����û�����վ�����
		{
			stopID = oneWStopNameToID[tempSub];                       //���ƣ��� ����3���ʵ���һ����еĵڼ�
		}
		else
		{
			//cout<<"������������trip�ĵ���վ������һ�����û�е������"<<tempSub<<endl;
			stopID = -66;
		}
	}
	else                                                 //����ǹ���վ�������
	{
		if(oneWStopNameToID.count(str1)!=0)               //���ڲ���a��b������£�ֱ���ж�һ���վ������û�����վ����
		{
			stopID = oneWStopNameToID[str1];                       //���ƣ��� ����3���ʵ���һ����еĵڼ�
		}
		else                                                      //���ֱ��mapû�У��ǾͼӸ�a��b���ж�
		{
			if(oneWStopNameToID.count(str1+"a")!=0 && oneWStopNameToID.count(str1+"b")!=0)       //��������˵��+a�У�+b�϶�Ҳ��
			{
				//�ֱ���trip�����굽����Ϊվ��a�ľ���͵�����Ϊվ��b�ľ���
				//ע�������һ��oneWStopNameToID��������ID��1��ʼ��oneWX��0��ʼ
				double d1=getTwoPointDistance(tripX,tripY,oneWX[oneWStopNameToID[str1+"a"]-1],oneWY[oneWStopNameToID[str1+"a"]-1]);
				double d2=getTwoPointDistance(tripX,tripY,oneWX[oneWStopNameToID[str1+"b"]-1],oneWY[oneWStopNameToID[str1+"b"]-1]);
				if(d1<=d2)               //���������Ϊվ��a����
				{
					stopID = oneWStopNameToID[str1+"a"];
				}
				else
				{
					stopID = oneWStopNameToID[str1+"b"];
				}
			}
			else                             //��a�ͼ�b��û�У���ʾ���վ����ܾ���һ��վ����û�У������ǱȽ��µ�վ��
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
	if (ID1 >= 1 && ID1 <= 10626 && ID2 >= 1 && ID2 <= 10626) {        //ȷ������վ�㷶Χ�ڵñ��
		if (ID1 <= 199) {                                   //ȷ���ǵ�����ͷ
			if (ID2 <= 199) {    //������ͷ������β
				return 1;
			}
			else {               //������ͷ������β
				return 2;
			}
		}
		else {                                              //������ͷ
			if (ID2 <= 199) {    //������ͷ������β
				return 3;
			}
			else {               //������ͷ������β
				return 4;
			}
		}
	}
	else {                     //���ֵ���վ��û�ҵ��ģ�������̫�µ�trip�Ĺ���վ��
		return 0;
	}
}

void getMultiType::fillTransferProbMatrix(){
	cout<<"fillTransferProbMatrix"<<endl;
	//�ȳ�ʼ������վ��ת�Ƹ���Ϊ0
	vector<int> temp;                                     //��ʱvector
	vector<double> countTransferFront;                            //��ʱ�洢ÿһ�е�199��ǰ�Ĵ���
	vector<double> countTransferBack;                             //��ʱ�洢ÿһ�е�199�Ժ�Ĵ���
	vector<double> countAccumulateCost;                           //��ʱ�洢�ۼ��ܳ���ʱ��
	for(unsigned int j=0;j<oneWStopName.size();j++) {
		temp.push_back(0);
		countTransferFront.push_back(0);
		countTransferBack.push_back(0);
		countAccumulateCost.push_back(0);
	}
	for(unsigned int i=0;i<oneWStopName.size();i++) {
		transferProbability.push_back(temp);
	}
	//����headID������飬Ҳ���Ǳߵ������������С����ʵ��ת�ƴ�������ȥ
	for(unsigned int i=0;i<headID.size();i++) {
		transferProbability[headID[i]-1][tailID[i]-1]=occurCount[i];
		countAccumulateCost[headID[i]-1] += (timeAccumulation[i]/60.0);
	}
	//�ۼƷ�ĸ�Ĵ���
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

	//���ݴ��������ܴ�������ת�Ƹ���
	cout<<"start storing transferProbability matrix..."<<endl;
	return;   //��Ϊת�Ƹ���ֱ����python�����ˣ������return�����ɣ��ò�����

	/*
	ofstream outfile1;
	outfile1.open("D:\\data\\��������\\MDNE\\output\\20170405_transferProbability.csv");
	int times = 1;
	for(unsigned int i=0;i<transferProbability.size();i++) {
		for(unsigned int j=0;j<transferProbability[0].size();j++) {
			//����ʵĻ���transferProbability����double��
			if (j<199 && countTransferFront[i]>0) {
				transferProbability[i][j] /= countTransferFront[i];
			}
			if (j>=199 && countTransferBack[i]>0){
				transferProbability[i][j] /= countTransferBack[i];
			}
			//�����csv
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
	cout<<"ODת�Ƹ��������ϣ�������"<<endl;
	*/
}


void getMultiType::makeTransferCount() {
	cout<<"start reading transfer type..."<<endl;
	string transferLine;                                                            //��ʱ��Ż����ֶε���Ϣ
	int transferTimes = 0;                                                          //ͳ�Ʒ������˵�trip�ж�����
	int straightArrive = 0;                                                         //ͳ��ֱ��Ĵ���
	int abnormal = 0;                                                               //ͳ�ƻ����ֶ��쳣
	int tooNew = 0;                                                                 //ͳ��̫�µ�վ��
	int disorder = 0;                                                               //ͳ�ƻ����ֶα�������쳣�ģ�����4��7��10������
	int times = 1;                                                                  //���ٶ�
	for (unsigned int i=0;i<transferField.size();i++) {                             //����ͺܴ��ˣ�for����trip�Ĵ���
		transferLine = transferField[i];
		int should_only_run_one_time = extractTransferInfo(transferLine);            //������������Ѻ���ֱ��д��if�ж��У��ǻ�ִ�ж��
		if(should_only_run_one_time==1) {
			transferTimes++;
		}
		else if(should_only_run_one_time==0) {
			straightArrive++;
		}
		else if(should_only_run_one_time==2) {
			abnormal++;
			cout<<"abnormal row is behind: "<<i<<"  it is��"<<transferLine<<endl;
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
	seeEdgeTypeNum(transfer_type);                  //��һ�¸�����ж���
	cout<<"transfer occurence num is "<<transferTimes<<"  straight arrive num is "<<straightArrive<<endl;
	cout<<"weird reading problem(�|) has��"<<abnormal<<"  too new to read: "<<tooNew<<"  disordered transfer field: "<<disorder<<endl;
}
int getMultiType::extractTransferInfo(string str1) {
	string head;                        //��ʱ�洢����ѡ���ͷ�ڵ�
	string tail;                        //��ʱ�洢����ѡ���β�ڵ�
	istringstream sin(str1);   
	string field;
	vector<string> vecTemp1;             //��ʱ��ÿ�����Ž�����Ȼ��ȡ
	string frontHead = "0";              //��ʱ��š�|��ǰ��վ��
	while (getline(sin,field,'|')) {
		vecTemp1.push_back(field);
	}
	if (vecTemp1.size()==1) {           //�����ڿ罻ͨ���߻��˻��߹�������
		istringstream sin(vecTemp1[0]);
		vector<string> vecTemp2;
		while (getline(sin,field,'-')) {
			vecTemp2.push_back(field);
		}
		if(vecTemp2.size()==4) {        //�����ֱ��м��޻���
			return 0;
		}
		else if(vecTemp2.size()>4) {                         //����ֱ������ж�����ǿ϶��ǵ�����γ˳�
			int tempHere = extractLongTransferInfo(frontHead,vecTemp2);
			if(tempHere==3) {
				return 3;
			}
			if(tempHere==4) {
				return 4;
			}
			return 1;
		}
		else {                                        //Ϊ���ų��쳣
			cout<<"transfer field is less than regular 4"<<endl;
			return 2;
		}
	}
	else {                               //���ڿ罻ͨ���߻��߹�������
		for (unsigned int i=0;i<vecTemp1.size();i++) {
			istringstream sin(vecTemp1[i]);
			vector<string> vecTemp2;
			while (getline(sin,field,'-')) {
				vecTemp2.push_back(field);
			}
			if(vecTemp2.size()==4 && i==0) {        //�����ֱ��м��޻���,�ǰѵ�һ�ε����һ���ڵ������
				frontHead = getSubwayStationName(vecTemp2[1],vecTemp2[3]);
			}
			else if (vecTemp2.size()==4 && i!=0) {  //�����ֱ��м��޻���,����һ�ε����һ���ɹ��ɲ���
				head = frontHead;
				tail = getSubwayStationName(vecTemp2[1],vecTemp2[0]);
				if(!insertTransfer(head, tail)) {
					//cout<<"̫���ˣ�"<<head<<"--"<<tail<<endl;
					return 3;
				}
				frontHead = getSubwayStationName(vecTemp2[1],vecTemp2[3]);
			}
			else if (vecTemp2.size()>4){     //����ֱ������ж�����ǿ϶��ǵ�����γ˳�,Ҫô�ǵ�һ�Σ�Ҫô�Ǻ���Ķ�
				int temp_longInfo = extractLongTransferInfo(frontHead,vecTemp2);
				if(temp_longInfo==3) {
					return 3;
				}
				if(temp_longInfo==4) {
					return 4;
				}
				frontHead = getSubwayStationName(vecTemp2[vecTemp2.size()-3],vecTemp2[vecTemp2.size()-1]);
			}
			else {                                              //Ϊ���ų��쳣
				cout<<"this transfer filed has big trouble, unable to read �|"<<endl;       //���ֱܷ治�ˡ���|����Է�� �|��
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
	string head;                        //��ʱ�洢����ѡ���ͷ�ڵ�
	string tail;                        //��ʱ�洢����ѡ���β�ڵ�
	if (str1 != "0") {                  //���ǰ���л���վ��
		head = str1;
		tail = getSubwayStationName(vec[1],vec[0]);
		if(!insertTransfer(head, tail)) {
			//cout<<"�С�|��ʱ̫���ˣ�"<<head<<"--"<<tail<<endl;
			return 3;
		}
	}
	if ((vec.size()-4)%3!=0) {
		//cout<<"��Ȼ����4��7��10��13������"<<endl;
		return 4;
	}
	int subwayTimes = (vec.size()-4)/3;   //���㻻����Ϣ���ж��ٴ�(n-4)/3 + 1��������м��ֻ��4�����ǻ��˾ͻ�����ͷ   
	int leftLine = 1;
	int middleStation = 3;
	int rightLine = 4;
	for(int i=0;i<subwayTimes;i++) {
		head = getSubwayStationName(vec[leftLine+i*3],vec[middleStation+i*3]);
		tail = getSubwayStationName(vec[rightLine+i*3],vec[middleStation+i*3]);
		if(!insertTransfer(head, tail)) {
			//cout<<"extractLongTransferInfo��̫���ˣ�"<<head<<"--"<<tail<<endl;
			return 3;
		}
	}
	return 1;
}
string getMultiType::getSubwayStationName(string line, string name) {
	string tempyLine; 
	if (subwayLineNumber.count(line)!=0){  //�õ�������Ϣ��ڶ���λ���ǲ��ǵ�������
		tempyLine=subwayLineNumber[line];
	}
	else{
		tempyLine = "";                             //������ǵ����������ǾͲ��ü��磺-2
	}
	return name+tempyLine;
}
bool getMultiType::insertTransfer(string head, string tail) {
	int tripHeadID;
	int tripTailID;
	int tripEdgeType;
	pair<map<EdgeHeadTail,int>::iterator,bool> Insert_Pair2;    //Ϊ��Ψһ�Դ������-ͷ�ڵ���-β�ڵ���
	tripHeadID = convertStopNameToID(head,0,0);
	tripTailID = convertStopNameToID(tail,0,0);
	tripEdgeType = convertPairStopToEdgeType(tripHeadID, tripTailID);
	if (tripEdgeType != 0) {
		//�ѱ߹�ϵ-ͷ�ڵ�-β�ڵ�-ʱ��-������������
		EdgeHeadTail temp(tripEdgeType, tripHeadID, tripTailID);
		Insert_Pair2 = EHT_TransferIndex.insert(pair<EdgeHeadTail,int>(temp,transfer_headID.size()));  //Ϊ����valueֵ��0��ʼ����
		if(Insert_Pair2.second==false)                          //�岻�����ʹ������ֱ߹�ϵ���Ѿ����ڹ���ͷ�ڵ�-β�ڵ�Ĺ�ϵ��
		{
			int numNow = EHT_TransferIndex[temp];
			transfer_occurCount[numNow]++;
		}
		else {                                                 //�ܲ�����ʹ������µ�
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
	//���ݴ�������ļ�
	cout<<"start outputing transfer occurence matrix..."<<endl;
	int times = 1;                                        //������ٶ�
	ofstream outfile1;
	outfile1.open("..\\result\\20170405_transferCount"+to_string(long long(singleHour))+"-"
		+to_string(long long(singleHour+durationLast))+".csv");

	//��Ϊlineģ�����������һ��һ�е�����
	for(map<EdgeHeadTail,int>::iterator it=EHT_TransferIndex.begin(); it!=EHT_TransferIndex.end(); it++) //��������ȷ������������������
	{
		//MNE���뻻��ǿ�ȵı��淽ʽ
		outfile1<<fixed<<setprecision(6)<<transfer_type[it->second]<<","<<transfer_headID[it->second]<<","<<transfer_tailID[it->second]<<",";
		outfile1<<fixed<<setprecision(6)<<transfer_occurCount[it->second]<<","<<0<<endl;
		//LINEģ�͵������ʽ
		/*outfile1<<fixed<<setprecision(6)<<transfer_headID[it->second]<<" "<<transfer_tailID[it->second]<<" "
		<<transfer_occurCount[it->second]<<endl;*/

	}
	outfile1.close();
	cout<<"transfer occurence matrix output done"<<endl;
	return;
}

void getMultiType::run(){
	makeSubwayLineMap();                                                                      //���Ǹ����Ը���string������map������
	ifstream finTrip("..\\data\\20170405-Trip-statistics-2CN-duration_example.csv");                  //��������trip�����ݣ����360����
	readODTrip(finTrip);																	//��ȡ��trip
	finTrip.close();
	readBusStop();                                                                           //��ȡ10626��վ�����
	makeStopStartCount();                                                                    //��ʼ����ߵĹ�ϵ
	fillTransferProbMatrix();                                                                //��ʼ����ת�ƾ������
	makeTransferCount();                                                                     //�ȱ���һ������trip�Ļ����ֶ�
	fillTransferCountMatrix();                                                              //�������ڶ�������trip��������������ϵǿ��

}
