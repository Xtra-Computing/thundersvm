/*
 * @brief: define evaluation functions
 * 
*/

#include "classifierEvaluater.h"

vector<real> ClassifierEvaluater::trainingError;
vector<real> ClassifierEvaluater::testingError;

void ClassifierEvaluater::collectSubSVMInfo(SvmModel &model, int insId, int trueLable, int nrClass, const vector<vector<real> > &predictedRes, bool isProbabilistic){
	//update model labeling information
	int rawLabel = trueLable;
	int originalLabel = -1;
	for (int pos = 0; pos < model.label.size(); pos++) {
		if (model.label[pos] == rawLabel)
			originalLabel = pos;
	}
	model.missLabellingMatrix[originalLabel][originalLabel]++; //increase the total occurrence of a label.
	int k = 0;
	for (int i = 0; i < nrClass; ++i) {
		for (int j = i + 1; j < nrClass; ++j) {
			int labelViaBinary = j;
			if(isProbabilistic == true){
				if (predictedRes[i][j] >= 0.5)
					labelViaBinary = i;
			}
			else{
				if (predictedRes[insId][k++] > 0)
					labelViaBinary = i;
			}

			if (i == originalLabel || j == originalLabel) {
				if (labelViaBinary != originalLabel) //miss classification
					model.missLabellingMatrix[originalLabel][labelViaBinary]++;
			}
		}
	}
}

/**
 * @brief: evaluate sub-classifiers for multi-class classification
 */
void ClassifierEvaluater::evaluateSubClassifier(const vector<vector<int> > &missLabellingMatrix, vector<real> &vErrorRate){
	vErrorRate.clear();

	int row = missLabellingMatrix.size();
	int col = missLabellingMatrix[0].size();
	int totalIns = 0, totalMiss = 0;
	int k = 0;
	for(int r = 0; r < row; r++){
		totalIns += missLabellingMatrix[r][r];
		for(int c = r + 1; c < col; c++){
			int totalRC = missLabellingMatrix[r][r] + missLabellingMatrix[c][c];
			int rcMissLabelling = missLabellingMatrix[r][c] + missLabellingMatrix[c][r];
			totalMiss += rcMissLabelling;
			printf("%d v.s %d acc: %.2f%;\t", r, c, (float)(totalRC - rcMissLabelling) / totalRC * 100);
			k++;
			if(k % 5 == 0)printf("\n");
			vErrorRate.push_back((float)rcMissLabelling / totalRC);
		}
	}
}

/*
 * @brief: update C to better fit the problem.
 */
vector<real> ClassifierEvaluater::updateC(const vector<real> &vOldC){
	vector<real> vNewC;
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
