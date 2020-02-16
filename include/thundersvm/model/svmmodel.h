//
// Created by jiashuai on 17-9-21.
//

#ifndef THUNDERSVM_SVMMODEL_H
#define THUNDERSVM_SVMMODEL_H

#include <thundersvm/dataset.h>
#include <thundersvm/svmparam.h>
#include <thundersvm/kernelmatrix.h>
#include <map>

using std::map;

/**
 * @brief Abstract class for different SVM models.
 */
class SvmModel {
public:
    /**
     * train model given dataset and param.
     * @param dataset training dataset
     * @param param param for training
     */
    virtual void train(const DataSet &dataset, SvmParam param) = 0;

    /**
     * predict label given instances.
     * @param instances instances used
     * @param batch_size the number of instances to predict parallel, higher value needs more memory
     * @return label (SVC, NuSVC), real number (SVR, NuSVR), {-1,+1} (OneClassSVC)
     */
    virtual vector<float_type> predict(const DataSet::node2d &instances, int batch_size);

    /**
     * predict decision values.
     * @param [in] instances instances used
     * @param [out] dec_values decision values predicted, #instances \f$times\f$ n_binary_models array
     * @param [in] batch_size the number of instances to predict parallel, higher value needs more memory
     */
    void predict_dec_values(const DataSet::node2d &instances, SyncArray<float_type> &dec_values, int batch_size) const;

    /**
     * performing cross-validation.
     * In \f$k\f$-fold cross_validation, dataset is spilt into \f$k\f$ equal size parts. Then each part is used as
     * testing set, while the remaining \f$k-1\f$ parts are used as training set. The whole dataset will be predicted
     * after \f$k\f$ training and testing.
     * @param dataset training dataset
     * @param param param for cross-validation
     * @param n_fold the number of fold in cross-validation
     * @return the same structure as predict()
     */
    virtual vector<float_type> cross_validation(DataSet dataset, SvmParam param, int n_fold);

    /**
     * save SvmModel to a file, the file format is the same as LIBSVM
     * @param path path and filename of the model file to save
     */
    virtual void save_to_file(string path);

    /**
     * load SvmModel from a file, the file format is the same as LIBSVM.
     * before call this function, one must read the first line of the model file, which contains svm_type. Then construct
     * right SvmModel (SVC, SVR, ...), since SvmModel is an abstract class.
     * @param path path and filename of saved model file
     */
    virtual void load_from_file(string path);

    /**
     * save SvmModel to a string
     */
    virtual string save_to_string();

    /**
     * load SvmModel from a string created by save_to_string.
     * @param data string created by save_to_string
     */
    virtual void load_from_string(string data);

    //return n_total_sv
    int total_sv() const;

    //return n_classes
    int get_n_classes() const;

    //return sv
    const DataSet::node2d &svs() const;

    //return n_sv
    const SyncArray<int> &get_n_sv() const;

    //return coef
    const SyncArray<float_type> &get_coef() const;

    //return linear_coef
    const SyncArray<float_type> &get_linear_coef() const;

    //return rho
    const SyncArray<float_type> &get_rho() const;

    //set max_iter
    void set_max_iter(int iter);

    //return dec_values
    const SyncArray<float_type> &get_dec_value() const;

    //set max_memory_size during training and prediction (input unit MB)
    void set_max_memory_size(size_t size);

	//set max_memory_size during training and prediction (input unit Byte)
	void set_max_memory_size_Byte(size_t size);

    //return n_binary_models
    int get_n_binary_models() const;

    //return prob_predict
    const vector<float> &get_prob_predict() const;

    void compute_linear_coef_single_model(size_t n_feature, const bool zero_based);
    //get the params, for scikit load params
    void get_param(char* kernel_type, int* degree, float* gamma, float* coef0, int* probability);

    //return sv_max_index
    int get_sv_max_index() const;

    //return sv_indices
    const vector<int> &get_sv_ind() const;

    //return label
    const vector<int> &get_label() const;

    //return param.probability
    const bool is_prob() const;

    virtual ~SvmModel() = default;

protected:

    /**
     * called at the begining of train(), do initialization
     * @param dataset
     * @param param
     */
    virtual void model_setup(const DataSet &dataset, SvmParam &param);

    ///calculate a proper working set size for training
    int get_working_set_size(int n_instances, int n_features);

    SvmParam param;
    /**
     * coefficients for each support vector, the structure is the same as LIBSVM. The coefficient is equal to
     * \f$\alpha_iy_i\f$ in SVM dual optimization problem.
     * For one-vs-one multi-class decomposition, \f$k\f$ is the number of classes, and \f$N_{sv}\f$ is the number of
     * support vectors. The size of coef is \f$ (k-1)\times N_{sv}\f$. Each support vector has at most \f$k-1\f$
     * coefficients. For each binary classifier \f$SVM_{i,j}\f$, the coefficients locate in: (1) class i,
     * coef[j-1][sv_start[i]...]; (2) class j, coef[i][sv_start[j]...], where sv_start[i] is the start position of
     * support vectors of class i.
     */

    SyncArray<float_type> coef;


    ///weights assigned to the features. Only available in the case of a linear kernel
    SyncArray<float_type> linear_coef;
    /**
     * support vectors of this model. The support vectors is grouped in classes 0,1,2,.... The sequence of them in each
     * group is as they appear in original dataset. A training instance is saved as a support vector IFF it is a
     * support vector in at least one binary model.
     */

    DataSet::node2d sv;
    ///the indices of support vectors
    vector<int> sv_indices;
    ///the number of support vectors for each class
    SyncArray<int> n_sv;

    ///the number of support vectors for all classes
    int n_total_sv;

    ///the bias term for each binary model
    SyncArray<float_type> rho;

    ///the number of classes
    int n_classes = 2;

    ///the number of binary models, equal to \f$k(k-1)/2\f$, where \f$k\f$ is the number of classes
    size_t n_binary_models;

    ///be used to predict probability for each binary model
    vector<float_type> probA;

    ///be used to predict probability for each binary model
    vector<float_type> probB;

    ///only for SVC, maps logical label (0,1,2,...) to real label in dataset (maybe 2,4,5,...)
    vector<int> label;

    ///Hard limit on iterations within solver, or -1 for no limit.
    int max_iter = -1;

    ///only for SVC and NuSVC, decision values
    SyncArray<float_type> dec_values;

    ///the probability for each class in predict
    vector<float> prob_predict;

    ///the maximum index of support vectors, used for scikit load svs
    int sv_max_index = 0;
};

#endif //THUNDERSVM_SVMMODEL_H
