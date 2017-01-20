/*
 * evaluateClassifier.h
 *
 *  Created on: 21 Jan 2017
 *      Author: Zeyi Wen
 */

#ifndef MASCOT_CLASSIFIEREVALUATER_H_
#define MASCOT_CLASSIFIEREVALUATER_H_

#include <vector>
#include <stdio.h>
#include "../SharedUtility/DataType.h"

using std::vector;

class ClassifierEvaluater{
public:
	static vector<float_point> trainingError;
	static vector<float_point> testingError;
public:
	/**
	 * @brief: evaluate sub-classifiers for multi-class classification
	 */
	static void evaluateSubClassifier(const vector<vector<int> > &missLabellingMatrix, vector<float_point> &vErrorRate){
		vErrorRate.clear();

		int row = missLabellingMatrix.size();
		int col = missLabellingMatrix[0].size();
		int totalIns = 0, totalMiss = 0;
		for(int r = 0; r < row; r++){
			totalIns += missLabellingMatrix[r][r];
			for(int c = r + 1; c < col; c++){
				int totalRC = missLabellingMatrix[r][r] + missLabellingMatrix[c][c];
				int rcMissLabelling = missLabellingMatrix[r][c] + missLabellingMatrix[c][r];
				totalMiss += rcMissLabelling;
				//printf("%d and %d accuracy is %f\n", r, c, (float)rcMissLabelling / totalRC);
				vErrorRate.push_back((float)rcMissLabelling / totalRC);
			}
		}
	    printf("classifier incorrect rate = %.2f%%(%d/%d)\n", totalMiss / (float) totalIns * 100,
	    		totalMiss, totalIns);
	}

	/*
	 * @brief: update C to better fit the problem.
	 */
	static vector<float_point> updateC(const vector<float_point> &vOldC){
		vector<float_point> vNewC;
		for(uint i = 0; i < trainingError.size(); i++){
			if(trainingError[i] < testingError[i]){
				vNewC.push_back(vOldC[i] * 0.5);//reduce C
			}else if(trainingError[i] > testingError[i]){
				vNewC.push_back(vOldC[i] * 2);//increase C
			}else{
				vNewC.push_back(vOldC[i]);
			}
		}
		return vNewC;
	}
};



#endif /* MASCOT_CLASSIFIEREVALUATER_H_ */
