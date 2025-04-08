// #include <bits/stdc++.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include "image.h"
// #include <iostream>

// using namespace std;


// int main()
// {
//   	string meta_file = "./cifar-10-batches-bin/batches.meta.txt";
// 	string data_file = "./cifar-10-batches-bin/data_batch_1.bin";

//     vector<std::string> class_names = loadClassNames(meta_file);
//     vector<Image> images = loadCIFARBatch(data_file, class_names, 10000);

//     cout << "Loaded " << images.size() << " images.\n";

//     // Print first image info
//     if (!images.empty()) {
//         cout << "First image label: " << images[0].label << " ("
//             << images[0].class_name << ")\n";
//         cout << "Pixel (0,0) R: " << (int)images[0].data[0][0][0]
//             << " G: " << (int)images[0].data[1][0][0]
//             << " B: " << (int)images[0].data[2][0][0] << "\n";
//     }

// 	// for(int i = 0; i < 3; i++){
// 	// 	for(int j = 0; j < 32; j++){
// 	// 		for(int k = 0; k < 32; k++){
// 	// 			cout << (int)images[0].data[i][j][k] << " ";
// 	// 		}
// 	// 		cout << endl;
// 	// 	}
// 	// 	cout << endl;
// 	// }

// 	// for(int i = 0; i < images.size(); i++){
// 	// 	cout << "Image " << i << " label: " << images[i].label << " (" << images[i].class_name << ")\n";
// 	// 	cout << "Pixel (0,0) R: " << (int)images[i].data[0][0][0] << " G: " << (int)images[i].data[1][0][0] << " B: " << (int)images[i].data[2][0][0] << "\n";
// 	// }
  	
	
// 	return 0;
// }


//============================================================================================================================================================================


#include "image.h"
#include "logreg.h"
#include "cnn.h"
#include "utils.h"

#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main() {
    std::string train_path = "./cifar-10-batches-bin/data_batch_";
    std::string test_path = "./cifar-10-batches-bin/test_batch.bin";
		std::string meta_file = "./cifar-10-batches-bin/batches.meta.txt";

		std::vector<Image> train_images;
    for(int i = 1; i < 6; i++){
			string path = train_path + to_string(i) + ".bin";
			vector<Image> batch = loadCIFARBatch(path, 1000);
			train_images.insert(train_images.end(), batch.begin(), batch.end());
		}
		
		//std::vector<Image> train_images = loadCIFARBatch(train_path, 1000);  // Training
    std::vector<Image> test_images = loadCIFARBatch(test_path, 1000);    // Testing

    std::cout << "Training Logistic Regression..." << std::endl;
    LogisticRegression model(3072, 10);  // input size, num_classes
	model.train(train_images, 10);       // assume you loaded training as `train_images`

	int correct = 0;
	for (const auto& img : train_images) {
	    int pred = model.predict(img);
	    if (pred == img.label)
        	correct++;
	}
	std::cout << "LogReg Train Accuracy: " << (float)correct / train_images.size() * 100 << "%\n";

	// Evaluate on test set
	correct = 0;
	for (const auto& img : test_images) {
	    int pred = model.predict(img);
			if (pred == img.label){
				correct++;
				std::cout << "Prediction: " << pred << " Actual: " << img.label << "		CORRECT!!\n";
			}
	}
	std::cout << "LogReg Test Accuracy: " << (float)correct / test_images.size() * 100 << "%\n";


    std::cout << "\nTraining CNN..." << std::endl;
    cnnTrainExample(train_images, 5);

    int cnn_correct = 0;
	for (const auto& img : train_images) {
	    std::vector<float> out = cnnForwardExample(img);
	    int pred = argmax(out);
	    if (pred == img.label)
        	cnn_correct++;
	}
	std::cout << "CNN Train Accuracy (multiclass): " << (float)cnn_correct / train_images.size() * 100 << "%\n";


  cnn_correct = 0;
	for (const auto& img : test_images) {
	    std::vector<float> out = cnnForwardExample(img);
	    int pred = argmax(out);
			if (pred == img.label){
				cnn_correct++;
				std::cout << "Prediction: " << pred << " Actual: " << img.label << "		CORRECT!!\n";
			}
	}
	std::cout << "CNN Test Accuracy (multiclass): " << (float)cnn_correct / test_images.size() * 100 << "%\n";


    return 0;
}

