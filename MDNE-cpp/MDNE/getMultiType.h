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

struct EdgeHeadTail {          //用作map的key值数据结构
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


	//把subwayLineNumber这个map手动填满
	void makeSubwayLineMap(); 
	//map插入失败的提示
	void insertFailOnMap(bool second);
	//读取出行trip的起终点坐标和出发时间、到达时间等信息
	void readODTrip(ifstream &fin);     
	//输入换乘信息字段，输出换乘信息中起点紧接着的那个地铁线路，如果不是地铁就输出空
	string extractSubwayLine(string str1);   
	//输入换乘信息字段，输出换乘信息中终点的那个地铁线路，如果不是地铁就输出空
	string extractLastSubwayLine(string str1);
	//将读进来的string型坐标转化成可比较的double类型
	double convertStringToDouble(string str1); 
	//将2017-04-05 19:23:38这样的string转化成int型的秒
	int convertTimeStringToInt(string str1);  
	//将2017-04-05 19:23:38这样的string中的19提取出来
	int extractTimeHour(string str1);        
	//读取所有地铁站点和公交站点信息,一共10626个
	void readBusStop();         
	//开始计算不同边类型下不同OD对之间的关系
	void makeStopStartCount();
	//统计完看一下不同的边类型分别有多少个
	void seeEdgeTypeNum(vector<int> &vec);
	//输入一个站名，看是不是地铁站，是就返回真
	bool isSubwayStop(string str1);       
	//输入地铁站名字，把后面的线路号标识和“站”字去掉，并直接加上线路的数字号，然后map
	string extractSubwayName(string str1);               
	//输入trip站点的名字、坐标，返回站点的编号。因为要给输入的地铁站名去“站”字和-，给公交站点加a和b
	int convertStopNameToID(string str1, double tripX, double tripY);
	//求两点间的距离
	double getTwoPointDistance(double x1,double y1,double x2,double y2);    
	//输入起始和终点站点的ID，判断是哪种边类型:地铁-地铁是1，地铁-公交是2，公交-公交是3，公交-地铁是4
	int convertPairStopToEdgeType(int ID1, int ID2);
	//填满转移概率矩阵的函数
	void fillTransferProbMatrix();

	//因为要先运行完读trip和读站点，所以就得把换乘字段费空间的存起来，最后再统计换乘联系强度
	void makeTransferCount();
	//加在读换乘字段那里边读边处理
	int extractTransferInfo(string str1);
	//处理长度为4、7、10无“|”那种换乘，如：燕南-地铁二号线-2-福田-地铁十一号线-1-车公庙-地铁七号线-1-桃源村
	int extractLongTransferInfo(string str1, vector<string> vec);
	//输入地铁名和线路名，如地铁二号线，福田，返回：福田2
	string getSubwayStationName(string line, string name);
	//输入前一个和后一个名字，插入到全局变量transfer_headID的一个函数
	bool insertTransfer(string head, string tail);
	//填满转移概率矩阵的函数
	void fillTransferCountMatrix();

	void run();

private:
	map<string, string> subwayLineNumber;     //用<地铁二号线,-2>这样来索引
	int tripDay;                              //用来指示现在读到第几天了的一个全局变量
	int singleHour;                           //用来用一个小时一个小时间隔算μ的
	int durationLast;                         //用来表示早高峰或晚高峰的跨度的，如两个小时或三个小时

	//记录trip起点的站名，起点站是地铁，并且把地铁的线路号加入到站名后面如：科苑站-2，表示起点是地铁二号线的科苑站
	vector<string> OStopName;
	vector<string> DStopName;
	vector<double> OGrid_X;                   //起始站点的坐标信息分开存
	vector<double> OGrid_Y;
	vector<double> DGrid_X; 
	vector<double> DGrid_Y;
	vector<string> transferField;             //存第11个换乘字段，可能会很消耗内存

	//记录对应trip的时间差，整数部分是秒数的int型差值，小数部分是出发时的小时数，比如1800.13，表示trip花了1800s，且该trip是13点开始的
	vector<double> timeDiff;
	vector<double> ODStopNumber;              //记录每条trip跨了多少个站点，小数部分记录换乘次数
	vector<double> tripDurationTime;          //trip的逗留时间,之前算好的

	//可能要读一下那个文件"merge_BusStation改进dbscan加地铁"
	vector<int> oneWStopID;                       //一万个站点的那个站点的ID号，分马路对面和地铁线路的统计
	vector<double> oneWX;                         //一万个站点的那个站点的X坐标或是线段端点的X坐标
	vector<double> oneWY;                         //一万个站点的那个站点的Y坐标或是线段端点的Y坐标
	vector<string> oneWStopName;                  //一万个站点的那个站点的名字
	map<string,int> oneWStopNameToID;             //根据一万个站名来索引其所在的ID号

	//用来存计算出的边关系
	vector<int> edgeType;                         //目前暂时存1，2，3，4这四种关系
	vector<int> headID;                           //头节点的ID
	vector<int> tailID;                           //尾节点的ID
	vector<int> occurCount;                       //边出现的次数
	vector<double> timeAccumulation;              //多次行程时间的累计
	vector<double> averageTime;                   //这层边关系下这种连接（一般肯定有多次）的平均行程时间
	vector<double> averageDuration;               //这层边关系下这种连接（一般肯定有多次）的平均逗留时间
	map<EdgeHeadTail, int> EHTIndex;              //根据边类型-头节点编号-尾节点编号来索引看是第几个插进来的

	vector<vector<int>> transferProbability;    //大小为10626*10626的矩阵，每个位置代表当前行站点转移到当前列站点的概率

	//用来统计换乘信息
	vector<int> transfer_type;                             //换乘的类型
	vector<int> transfer_headID;                           //换乘头节点的ID
	vector<int> transfer_tailID;                           //换乘尾节点的ID
	vector<int> transfer_occurCount;                       //换乘边出现的次数
	map<EdgeHeadTail, int> EHT_TransferIndex;              //根据边类型-头节点编号-尾节点编号来索引看是第几个插进来的

	//以后千万别用邻接矩阵存，太占内存了
	//vector<vector<int>> transferCount;         //大小为10626*10626的矩阵，每个位置代表当前行站点换乘到当前列站点的次数              

};

