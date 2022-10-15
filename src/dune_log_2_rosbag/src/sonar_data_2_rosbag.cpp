#include "sonar_data_2_rosbag.hpp"

using namespace std;

int main(int argc, char * argv[])
{
	rclcpp::init(argc, argv);
 	
	// Open file
	string fname;
	//cout<<"Enter the csv file name: ";
	//cin>>fname;
	fname = "bags/SonarData.csv";

	fstream file (fname, ios::in);
	if(!file.is_open()) {
		cout<<"Could not open the file\n";
		return 1;
	}



	// Setup writer and open bag
	std::unique_ptr<rosbag2_cpp::Writer> writer = std::make_unique<rosbag2_cpp::Writer>();
	string bag_path, sonar_topic_name;

	//cout << "Enter the path for the bag: ";
	//cin >> bag_path;
	bag_path = "bags/sonar_bag";

	//cout << "Enter the name of the sonar topic: ";
	//cin >> sonar_topic_name;
	sonar_topic_name = "Sonar";

	writer->open(bag_path);

	// Read file, parse data and save msg to bag
	string line, data, header, word;
	CDSSPParser parser;
	float ns;
	brov2_interfaces::msg::Sonar sonar_msg = brov2_interfaces::msg::Sonar();
	rclcpp::Time time_stamp;
	vector<string> headers;

	getline(file, line);
	//line.erase(std::remove(line.begin(), line.end(), '\n'), line.cend());
	stringstream header_ss(line);

	while(getline(header_ss, word, ',')) {
		headers.push_back(word);
		cout << word;
	}

	cout << endl;

	for (vector<string>::iterator t=headers.begin(); t!=headers.end(); ++t) {
        cout<<*t<<endl;
    }
	// Fix for making getline return the last header value
	//header.erase(std::remove(header.begin(), header.end(), '\n'), header.cend());
	//header = header + ",\n";
	//stringstream header_ss_test(header);

	// cout << header;
	// cout << "\n\n";

	//while(getline(header_ss_test, hdr, ',')) {
	//	cout << hdr + "\n";
	//}
	// cout << "\n\n";
	// cout << header;
	// cout << "\n\n";

	// header_ss.clear();
	// stringstream test;
	// test << "cdc,cdcd,cdcd,vrt,yntyn,tn,tyb,t,btb\n";

	// //cout << header_ss.str();

	// while(getline(header_ss_test, hdr, ',')) {
	// 	cout << hdr;
	// }
	
	// cout << "\n\n";

	// header_ss << header;

    

    


	// Get the headers for the data we want
	string timestamp_header, sonar_data_header, garbage;

	cout << "Headers in " + fname + ": " + line + "\n";

	cout << "Enter the header for timestamps: ";
	cin >> timestamp_header;

	cout << "Enter the header for the sonar data: ";
	cin >> sonar_data_header;
	sonar_data_header = headers.back();
	
	while(getline(file, line))
	{
		stringstream line_ss(line);
		vector<string>::iterator hdr = headers.begin(); 
		
		while((getline(line_ss, data, ',')) && (hdr!=headers.end()) ) 
		{	
			if(*hdr == timestamp_header) {
				ns = powf(stof(data), 9.0);
				time_stamp = rclcpp::Time((int) ns);
			} 
			if(*hdr == sonar_data_header) {
				for (uint i=0; i<data.length(); i+=2) {
					string d;
					if(i+2 < data.length()){ //prevent it from throwing out_of_range exception
        				d = data.substr(i,2);
						unsigned int b;   
    					stringstream ss;
						ss << std::hex << d;
						ss >> b;
						if(d.length() == 3) {
							cout << "WTF" << endl;
						}
						if(b > 250) {
							cout << "Should change state: " << b << endl;
						}
						if (parser.Add(b)) {

							char* data0;
							char* data1;
							int size0 = 0;
							int size1 = 0;

							parser.GetChannelData(data0, &size0, data1, &size1); 

							sonar_msg.header.stamp = time_stamp;
							for (int j = 0; j < size0; j++) {
								sonar_msg.data_zero[j] = data0[j];
							}	
							for (int j = 0; j < size1; j++) {
								sonar_msg.data_one[j] = data1[j];
							}
		
							writer->write(
								sonar_msg, 
								sonar_topic_name, 
								time_stamp
							);
							cout << "Message written!\n";
						}
					}
				}


				
			}



				
				//char b;
				// for (uint i=0; i<data.length(); i+=2) {
				// 	string d = "";
				// 	if(i+2 < data.length()){ //prevent it from throwing out_of_range exception
        		// 		d = data.substr(i,2);
				// 	}
					//b = char(stoi(d));
					//int num = stoi(d);
					//char c = static_cast<char>(num);
					
					//cout << std::hex << d << endl;
					//cout << d << endl;
					//cin >> garbage;
					// if (parser.Add(b)) {

					// 	char* data0;
					// 	char* data1;
					// 	int size0 = 0;
					// 	int size1 = 0;

					// 	parser.GetChannelData(data0, &size0, data1, &size1); 

					// 	sonar_msg.header.stamp = time_stamp;
					// 	for (int j = 0; j < size0; j++) {
					// 		sonar_msg.data_zero[j] = data0[j];
					// 	}	
					// 	for (int j = 0; j < size1; j++) {
					// 		sonar_msg.data_one[j] = data1[j];
					// 	}
	
					// 	writer->write(
					// 		sonar_msg, 
					// 		sonar_topic_name, 
					// 		time_stamp
					// 	);
					// 	cout << "Message written!\n";
					// }
				//} 
			//}
			hdr++;
		}
	}

	rclcpp::shutdown();
	return 0;
}