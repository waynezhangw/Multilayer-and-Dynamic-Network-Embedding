#pragma once

#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<map>

using std::string;
using std::vector;
using std::map;
using std::ifstream;

struct EdgeHeadTail {          //����map��keyֵ���ݽṹ
	int edge;
	int head;
	int tail;
	bool operator < (const EdgeHeadTail &o) const{
		return edge < o.edge || (edge == o.edge && head < o.head) || (edge == o.edge && head == o.head && tail < o.tail);
	}
	EdgeHeadTail(int x, int y, int z):edge(x),head(y),tail(z){}
};

class getMultiType
{
public:
	getMultiType(void);
	~getMultiType(void);


	//��subwayLineNumber���map�ֶ�����
	void makeSubwayLineMap(); 
	//map����ʧ�ܵ���ʾ
	void insertFailOnMap(bool second);
	//��ȡ����trip�����յ�����ͳ���ʱ�䡢����ʱ�����Ϣ
	void readODTrip(ifstream &fin);     
	//���뻻����Ϣ�ֶΣ����������Ϣ���������ŵ��Ǹ�������·��������ǵ����������
	string extractSubwayLine(string str1);   
	//���뻻����Ϣ�ֶΣ����������Ϣ���յ���Ǹ�������·��������ǵ����������
	string extractLastSubwayLine(string str1);
	//����������string������ת���ɿɱȽϵ�double����
	double convertStringToDouble(string str1); 
	//��2017-04-05 19:23:38������stringת����int�͵���
	int convertTimeStringToInt(string str1);  
	//��2017-04-05 19:23:38������string�е�19��ȡ����
	int extractTimeHour(string str1);        
	//��ȡ���е���վ��͹���վ����Ϣ,һ��10626��
	void readBusStop();         
	//��ʼ���㲻ͬ�������²�ͬOD��֮��Ĺ�ϵ
	void makeStopStartCount();
	//ͳ���꿴һ�²�ͬ�ı����ͷֱ��ж��ٸ�
	void seeEdgeTypeNum(vector<int> &vec);
	//����һ��վ�������ǲ��ǵ���վ���Ǿͷ�����
	bool isSubwayStop(string str1);       
	//�������վ���֣��Ѻ������·�ű�ʶ�͡�վ����ȥ������ֱ�Ӽ�����·�����ֺţ�Ȼ��map
	string extractSubwayName(string str1);               
	//����tripվ������֡����꣬����վ��ı�š���ΪҪ������ĵ���վ��ȥ��վ���ֺ�-��������վ���a��b
	int convertStopNameToID(string str1, double tripX, double tripY);
	//�������ľ���
	double getTwoPointDistance(double x1,double y1,double x2,double y2);    
	//������ʼ���յ�վ���ID���ж������ֱ�����:����-������1������-������2������-������3������-������4
	int convertPairStopToEdgeType(int ID1, int ID2);
	//����ת�Ƹ��ʾ���ĺ���
	void fillTransferProbMatrix();

	//��ΪҪ���������trip�Ͷ�վ�㣬���Ծ͵ðѻ����ֶηѿռ�Ĵ������������ͳ�ƻ�����ϵǿ��
	void makeTransferCount();
	//���ڶ������ֶ�����߶��ߴ���
	int extractTransferInfo(string str1);
	//������Ϊ4��7��10�ޡ�|�����ֻ��ˣ��磺����-����������-2-����-����ʮһ����-1-������-�����ߺ���-1-��Դ��
	int extractLongTransferInfo(string str1, vector<string> vec);
	//�������������·��������������ߣ�������أ�����2
	string getSubwayStationName(string line, string name);
	//����ǰһ���ͺ�һ�����֣����뵽ȫ�ֱ���transfer_headID��һ������
	bool insertTransfer(string head, string tail);
	//����ת�Ƹ��ʾ���ĺ���
	void fillTransferCountMatrix();

	void run();

private:
	map<string, string> subwayLineNumber;     //��<����������,-2>����������
	int tripDay;                              //����ָʾ���ڶ����ڼ����˵�һ��ȫ�ֱ���
	int singleHour;                           //������һ��Сʱһ��Сʱ�����̵�
	int durationLast;                         //������ʾ��߷����߷�Ŀ�ȵģ�������Сʱ������Сʱ

	//��¼trip����վ�������վ�ǵ��������Ұѵ�������·�ż��뵽վ�������磺��Էվ-2����ʾ����ǵ��������ߵĿ�Էվ
	vector<string> OStopName;
	vector<string> DStopName;
	vector<double> OGrid_X;                   //��ʼվ���������Ϣ�ֿ���
	vector<double> OGrid_Y;
	vector<double> DGrid_X; 
	vector<double> DGrid_Y;
	vector<string> transferField;             //���11�������ֶΣ����ܻ�������ڴ�

	//��¼��Ӧtrip��ʱ������������������int�Ͳ�ֵ��С�������ǳ���ʱ��Сʱ��������1800.13����ʾtrip����1800s���Ҹ�trip��13�㿪ʼ��
	vector<double> timeDiff;
	vector<double> ODStopNumber;              //��¼ÿ��trip���˶��ٸ�վ�㣬С�����ּ�¼���˴���
	vector<double> tripDurationTime;          //trip�Ķ���ʱ��,֮ǰ��õ�

	//����Ҫ��һ���Ǹ��ļ�"merge_BusStation�Ľ�dbscan�ӵ���"
	vector<int> oneWStopID;                       //һ���վ����Ǹ�վ���ID�ţ�����·����͵�����·��ͳ��
	vector<double> oneWX;                         //һ���վ����Ǹ�վ���X��������߶ζ˵��X����
	vector<double> oneWY;                         //һ���վ����Ǹ�վ���Y��������߶ζ˵��Y����
	vector<string> oneWStopName;                  //һ���վ����Ǹ�վ�������
	map<string,int> oneWStopNameToID;             //����һ���վ�������������ڵ�ID��

	//�����������ı߹�ϵ
	vector<int> edgeType;                         //Ŀǰ��ʱ��1��2��3��4�����ֹ�ϵ
	vector<int> headID;                           //ͷ�ڵ��ID
	vector<int> tailID;                           //β�ڵ��ID
	vector<int> occurCount;                       //�߳��ֵĴ���
	vector<double> timeAccumulation;              //����г�ʱ����ۼ�
	vector<double> averageTime;                   //���߹�ϵ���������ӣ�һ��϶��ж�Σ���ƽ���г�ʱ��
	vector<double> averageDuration;               //���߹�ϵ���������ӣ�һ��϶��ж�Σ���ƽ������ʱ��
	map<EdgeHeadTail, int> EHTIndex;              //���ݱ�����-ͷ�ڵ���-β�ڵ������������ǵڼ����������

	vector<vector<int>> transferProbability;    //��СΪ10626*10626�ľ���ÿ��λ�ô���ǰ��վ��ת�Ƶ���ǰ��վ��ĸ���

	//����ͳ�ƻ�����Ϣ
	vector<int> transfer_type;                             //���˵�����
	vector<int> transfer_headID;                           //����ͷ�ڵ��ID
	vector<int> transfer_tailID;                           //����β�ڵ��ID
	vector<int> transfer_occurCount;                       //���˱߳��ֵĴ���
	map<EdgeHeadTail, int> EHT_TransferIndex;              //���ݱ�����-ͷ�ڵ���-β�ڵ������������ǵڼ����������

	//�Ժ�ǧ������ڽӾ���棬̫ռ�ڴ���
	//vector<vector<int>> transferCount;         //��СΪ10626*10626�ľ���ÿ��λ�ô���ǰ��վ�㻻�˵���ǰ��վ��Ĵ���              

};

